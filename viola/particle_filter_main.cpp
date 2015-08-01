#include <signal.h>
#include <cstdlib>
#include <string>
#include "project_config.h"

IGNORE_WARNINGS_PUSH

#include <mrpt/gui/CDisplayWindow3D.h>
#include <mrpt/maps/CColouredPointsMap.h>
#include <mrpt/opengl/CGridPlaneXY.h>
#include <mrpt/opengl/stock_objects.h>
#include <mrpt/opengl/CPointCloudColoured.h>

#include <mrpt/gui/CDisplayWindow.h>
#include <mrpt/random.h>
#include <mrpt/bayes/CParticleFilterData.h>
#include <mrpt/obs/CSensoryFrame.h>
#include <mrpt/obs/CObservationImage.h>
#include <mrpt/otherlibs/do_opencv_includes.h>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/threading.h>
#include <libfreenect2/registration.h>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>

IGNORE_WARNINGS_POP

#include "ImageRegistration.h"

#include "CImageParticleFilter.h"
#include "MiscHelpers.h"
#include "GeometryHelpers.h"
#include "EllipseFunctions.h"

#include "ColorModel.h"
#include "FacesDetection.h"
#include "ModelParameters.h"
#include "StateEstimation.h"
#include "MultiTracker.h"
#include "Tracker.h"
#include "EllipseStash.h"

#include "Kinect2Camera.h"
#include "Kinect2VideoReader.h"

using namespace mrpt;
using namespace mrpt::bayes;
using namespace mrpt::gui;
using namespace mrpt::obs;
using namespace mrpt::random;
using namespace mrpt::opengl;
using namespace mrpt::maps;

using namespace std;

void dispDepth(const cv::Mat &in, cv::Mat &out, const float maxValue)
{
    cv::Mat tmp = cv::Mat(in.rows, in.cols, CV_8U);
    const uint32_t maxInt = 255;

    #pragma omp parallel for
    for (int r = 0; r < in.rows; ++r) {
        const uint16_t *itI = in.ptr<uint16_t>(r);
        uint8_t *itO = tmp.ptr<uint8_t>(r);

        for (int c = 0; c < in.cols; ++c, ++itI, ++itO) {
            *itO = (uint8_t)std::min((*itI * maxInt / maxValue), 255.0f);
        }
    }

    cv::applyColorMap(tmp, out, cv::COLORMAP_JET);
}

void combine(const cv::Mat &inC, const cv::Mat &inD, cv::Mat &out)
{
    out = cv::Mat(inC.rows, inC.cols, CV_8UC3);

    #pragma omp parallel for
    for (int r = 0; r < inC.rows; ++r) {
        const cv::Vec3b
        *itC = inC.ptr<cv::Vec3b>(r),
         *itD = inD.ptr<cv::Vec3b>(r);
        cv::Vec3b *itO = out.ptr<cv::Vec3b>(r);

        for (int c = 0; c < inC.cols; ++c, ++itC, ++itD, ++itO) {
            itO->val[0] = (itC->val[0] + itD->val[0]) >> 1;
            itO->val[1] = (itC->val[1] + itD->val[1]) >> 1;
            itO->val[2] = (itC->val[2] + itD->val[2]) >> 1;
        }
    }
}

void create_cloud(const std::vector<Eigen::Vector3f> &points, const float scale, CColouredPointsMap &cloud)
{
    cloud.clear();
    const size_t N = points.size();
    for (size_t i = 0; i < N; i++){
        const Eigen::Vector3f &v = points[i];
        if (isnan(v[2]) || std::abs(v[2] * scale) <= 0.001) {
            continue;
        }
        cloud.insertPoint(-v[2] * scale, v[0] * scale, -v[1] * scale, 0, 1, 0);
    }
}

void create_cloud(const cv::Mat &color, const cv::Mat &depth, const float scale, const ImageRegistration &reg, CColouredPointsMap &cloud)
{
    //const float badPoint = std::numeric_limits<float>::quiet_NaN();
    cloud.clear();
    for (int r = 0; r < depth.rows; ++r) {
        const uint16_t *itD = depth.ptr<uint16_t>(r);
        const cv::Vec3b *itC = color.ptr<cv::Vec3b>(r);
        const float y = reg.lookupY.at<float>(0, r);
        const float *itX = reg.lookupX.ptr<float>();

        for (size_t c = 0; c < (size_t)depth.cols; ++c, ++itD, ++itC, ++itX) {
            register const float depthValue = *itD;
            // Check for invalid measurements
            if (isnan(depthValue) || std::abs(depthValue) <= 0.001) {
                continue;
            }

            const float z_coord = depthValue * scale;
            const float x_coord = (*itX) * depthValue* scale;
            const float y_coord = y * depthValue* scale;
            //cout << "CLOUD " << x_coord << ' ' << y_coord << ' ' << z_coord << endl;
            cloud.insertPoint(-z_coord, x_coord, -y_coord , itC->val[2] / 255.0, itC->val[1] / 255.0, itC->val[0] / 255.0);
        }
    }
}

int particle_filter()
{
#define VIOLA
#ifdef VIOLA
    //Cascades initialization
    //string face_cascade_name = "../cascades/lbpcascade_frontalface.xml";
    string face_cascade_name = "../cascades/haarcascade_frontalface_default.xml";
    string eyes_cascade_name = "../cascades/haarcascade_eye_tree_eyeglasses.xml";
    cv::ocl::OclCascadeClassifier ocl_face_cascade;
    cv::ocl::OclCascadeClassifier ocl_eyes_cascade;
    cv::CascadeClassifier face_cascade;
    cv::CascadeClassifier eyes_cascade;

    if (!face_cascade.load(face_cascade_name)) {
        printf("--(!)Error loading %s\n", face_cascade_name.c_str());
        return -1;
    }

    if (!eyes_cascade.load(eyes_cascade_name)) {
        printf("--(!)Error loading %s\n", eyes_cascade_name.c_str());
        return -1;
    }

    if (!ocl_face_cascade.load(face_cascade_name)) {
        printf("--(!)Error loading %s\n", face_cascade_name.c_str());
        return -1;
    }

    if (!ocl_eyes_cascade.load(eyes_cascade_name)) {
        printf("--(!)Error loading %s\n", eyes_cascade_name.c_str());
        return -1;
    }
#endif

#define DLIB
#ifdef DLIB
    dlib::frontal_face_detector face_detector = dlib::get_frontal_face_detector();
#endif


    //Kinect2Camera video_feed;
    Kinect2VideoReader video_feed = Kinect2VideoReader(std::string("013572345247"), std::string("/run/media/juen/1cf91ca4-036c-44f3-a9b8-35deb7ced99c/videos/video3/video0"), std::string("avi"));

    char *calib_dir = getenv("HOME");
    const std::string calib_path = std::string(calib_dir) + "/kinect2_calib/";

    //Registration initialization
    ImageRegistration reg;
    reg.init(calib_path, video_feed.get_device_serial_number());

    //load or precompute ellipses projections
    EllipseStashLoader ellipses(reg, std::vector<BodyPart> {BodyPart::HEAD, BodyPart::TORSO}, std::vector<std::string>{"ellipses_0.150000x0.250000.bin", "ellipses_0.300000x0.200000.bin"});

    cv::Mat color_frame;
    cv::Mat color_display_frame;
    cv::Mat depth_frame;

    CDisplayWindow image("image");
    //CDisplayWindow registered_color_window("registered_depth_color");
    CDisplayWindow gradient_color_window("gradient_color_window");

    //CDisplayWindow model_candidate_window("model_candidate_window");
    //CDisplayWindow model_candidate_histogram_window("model_candidate_histogram_window");

    CDisplayWindow model_image_window("model-image");
    CDisplayWindow model_histogram_window("model_histogram_window");
    CDisplayWindow model_histogram_torso_window("histogram torso");
    //CDisplayWindow gradient_depth_window("gradient_depth_window");

    //CDisplayWindow model_histogram_window2("2model_histogram_window");
    //CDisplayWindow model_candidate_histogram_window2("2model_candidate_histogram_window");

     //CDisplayWindow image_hist_chest_color_score_window ("chest_color_score_window");
     //CDisplayWindow image_hist_head_color_score_window ("head_color_score_window");
     //CDisplayWindow image_hist_head_fitting_score_window ("head_fitting_score_window");
     //CDisplayWindow image_hist_head_z_score_window ("head_z_score_window");
     //CDisplayWindow image_hist_score_window ("score_window");

//#define VIEW_3D
#ifdef VIEW_3D
    // -------------------3D view stuff------------------
    CDisplayWindow3D win3D("Kinect 3D view", 640, 480);

    win3D.setCameraAzimuthDeg(0);
    win3D.setCameraElevationDeg(0);
    win3D.setCameraZoom(1);
    win3D.setFOV(65);
    win3D.setCameraPointingToPoint(0, 0, 0);

    mrpt::opengl::CPointCloudColouredPtr scene_points = mrpt::opengl::CPointCloudColoured::Create();
    mrpt::opengl::CPointCloudColouredPtr particle_points = mrpt::opengl::CPointCloudColoured::Create();
    mrpt::opengl::CPointCloudColouredPtr model_center_points = mrpt::opengl::CPointCloudColoured::Create();
    scene_points->setPointSize(0.5);
    particle_points->setPointSize(2);
    model_center_points->setPointSize(2);

    {
        mrpt::opengl::COpenGLScenePtr &scene = win3D.get3DSceneAndLock();
        scene->insert(scene_points);
        scene->insert(particle_points);
        scene->insert(model_center_points);
        scene->insert(mrpt::opengl::CGridPlaneXY::Create());
        //scene->insert(mrpt::opengl::stock_objects::CornerXYZSimple());
        win3D.unlockAccess3DScene();
        win3D.repaint();
    }

    CColouredPointsMap scene_points_map;
    CColouredPointsMap particle_points_map;
    CColouredPointsMap model_center_points_map;
    scene_points_map.colorScheme.scheme = CColouredPointsMap::cmFromIntensityImage;
#endif

    //MRPT random generator initialization
    randomGenerator.randomize();
    // Create PF
    // ----------------------
    CParticleFilter::TParticleFilterOptions PF_options;
    PF_options.adaptiveSampleSize = false;
    PF_options.PF_algorithm = CParticleFilter::pfStandardProposal;
    //PF_options.resamplingMethod = CParticleFilter::prSystematic;
    PF_options.resamplingMethod = CParticleFilter::prMultinomial;

    CParticleFilter PF;
    PF.m_options = PF_options;

    MultiTracker<DEPTH_TYPE> trackers(&reg);

    time_t start, end;
    int counter = 0;
    double sec;
    double fps;

    while (!mrpt::system::os::kbhit()) {

        if (counter == 0){
            time(&start);
        }

        // Adquisition
        uint64_t read_kinect_t0 = cv::getTickCount();

        cv::Mat color_mat, depth_mat;

        video_feed.grab(color_mat, depth_mat);

        float read_kinect_t = (cv::getTickCount() - read_kinect_t0) / double(cv::getTickFrequency());
        //std::cout << "TIMES_READ_KINECT " << read_kinect_t << std::endl;

        //Registration
        uint64_t registration_t0 = cv::getTickCount();

        cv::Mat registered_depth;
        reg.register_images(color_mat, depth_mat, registered_depth);
        cv::flip(color_mat, color_mat, 1);

        float registration_t = (cv::getTickCount() - registration_t0) / double(cv::getTickFrequency());
        //std::cout << "TIMES_REGISTRATION " << registration_t << std::endl;

        // Observation building
        //color_frame = color_mat(cv::Rect(140, 0, color_mat.cols - 140 - 290, color_mat.rows));
        //depth_frame = registered_depth(cv::Rect(140, 0, color_mat.cols - 140 - 290, color_mat.rows));
        color_frame = color_mat;
        depth_frame = registered_depth;
        color_display_frame = color_frame.clone();


#define OCL_CONVERSION
#ifdef OCL_CONVERSION
        uint64_t color_conversion_t0 = cv::getTickCount();
        cv::ocl::oclMat ocl_hsv_frame;
        cv::ocl::oclMat ocl_color_frame(color_frame);
        cv::ocl::cvtColor(ocl_color_frame, ocl_hsv_frame, cv::COLOR_BGR2HSV);
        cv::ocl::oclMat ocl_gray_frame;
        cv::ocl::cvtColor(ocl_color_frame, ocl_gray_frame, cv::COLOR_BGR2GRAY);

        cv::Mat hsv_frame = ocl_hsv_frame;
        //cv::Mat gray_frame = ocl_gray_frame;
        float color_conversion_t = (cv::getTickCount() - color_conversion_t0) / double(cv::getTickFrequency());

        uint64_t sobel_t0 = cv::getTickCount();

        cv::Mat gradient_vectors, gradient_magnitude, gradient_magnitude_scaled;
        std::tie(gradient_vectors, gradient_magnitude, gradient_magnitude_scaled) = sobel_operator(ocl_gray_frame);
        float sobel_t = (cv::getTickCount() - sobel_t0) / double(cv::getTickFrequency());
#else
        uint64_t color_conversion_t0 = cv::getTickCount();

        cv::Mat hsv_frame;
        cv::cvtColor(color_frame, hsv_frame, cv::COLOR_BGR2HSV);
        cv::Mat gray_frame;
        cvtColor(color_frame, gray_frame, CV_RGB2GRAY);

        float color_conversion_t = (cv::getTickCount() - color_conversion_t0) / double(cv::getTickFrequency());

        uint64_t sobel_t0 = cv::getTickCount();
        cv::Mat gradient_vectors, gradient_magnitude, gradient_magnitude_scaled;
        std::tie(gradient_vectors, gradient_magnitude, gradient_magnitude_scaled) = sobel_operator(gray_frame);

        float sobel_t = (cv::getTickCount() - sobel_t0) / double(cv::getTickFrequency());
#endif
        //std::cout << "TIMES_COLOR_CONVERSION " << color_conversion_t << std::endl;
        //cv::Mat hsv_frame;
        //cv::cvtColor(color_frame, hsv_frame, cv::COLOR_BGR2HSV);

        CObservationImagePtr obsImage_color = CObservationImage::Create();
        CObservationImagePtr obsImage_hsv = CObservationImage::Create();
        CObservationImagePtr obsImage_depth = CObservationImage::Create();
        CObservationImagePtr obsImage_gradient_vectors = CObservationImage::Create();
        CObservationImagePtr obsImage_gradient_magnitude = CObservationImage::Create();

        const std::unique_ptr<IplImage> ipl_image_color_frame(new IplImage(color_frame));
        const std::unique_ptr<IplImage> ipl_image_hsv_frame(new IplImage(hsv_frame));
        const std::unique_ptr<IplImage> ipl_image_depth_frame( new IplImage(depth_frame));
        const std::unique_ptr<IplImage> ipl_image_gradient_vectors(new IplImage(gradient_vectors));
        const std::unique_ptr<IplImage> ipl_image_gradient_magnitude(new IplImage(gradient_magnitude));

        obsImage_color->image.setFromIplImageReadOnly(ipl_image_color_frame.get());
        obsImage_color->sensorLabel = "color";

        obsImage_hsv->image.setFromIplImageReadOnly(ipl_image_hsv_frame.get());
        obsImage_hsv->sensorLabel = "hsv";

        obsImage_depth->image.setFromIplImageReadOnly(ipl_image_depth_frame.get());
        obsImage_depth->sensorLabel = "depth";

        obsImage_gradient_vectors->image.setFromIplImageReadOnly(ipl_image_gradient_vectors.get());
        obsImage_gradient_vectors->sensorLabel = "gradient_vectors";

        obsImage_gradient_magnitude->image.setFromIplImageReadOnly(ipl_image_gradient_magnitude.get());
        obsImage_gradient_magnitude->sensorLabel = "gradient_magnitude";

        // memory freed by SF.
        CSensoryFrame observation;
        observation.insert(obsImage_color);
        observation.insert(obsImage_hsv);
        observation.insert(obsImage_depth);
        observation.insert(obsImage_gradient_vectors);
        observation.insert(obsImage_gradient_magnitude);

        // Tracking
        std::vector<cv::Vec3f> circles;

        uint64_t viola_t0 = cv::getTickCount();

        cv::ocl::oclMat ocl_gray_frame_upper_half = ocl_gray_frame(cv::Rect(0, 0, ocl_gray_frame.cols, ocl_gray_frame.rows * 0.75));
        std::vector<cv::Rect> faces_roi = viola_faces::detect_faces_dual(ocl_gray_frame_upper_half, ocl_face_cascade, ocl_eyes_cascade, 1, face_detector, color_frame, color_display_frame);

        for (auto &roi : faces_roi){
            circles.push_back(cv::Vec3f(roi.x + roi.width / 2, roi.y + roi.height / 2, roi.width / 2));
        }

        float viola_t = (cv::getTickCount() - viola_t0) / double(cv::getTickFrequency());

        //std::cout << "TIMES_VIOLA " << viola_t << std::endl;

        if (circles.size() != 0) {
            int circle_max = 0;
            double radius_max = circles[0][2];
            for (size_t i = 0; i < circles.size(); i++) {
                cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
                int radius = cvRound(circles[i][2]);
                if (radius_max < radius) {
                    radius_max = radius;
                    circle_max = i;
                }
            }

            std::cout << "OBJECTS FOUND " << circles.size() << std::endl;

            cv::Point center(cvRound(circles[circle_max][0]), cvRound(circles[circle_max][1]));

            //cv::circle(color_display_frame, center, circles[circle_max][2], cv::Scalar(0, 255, 255), -1, 8, 0);

            //TODO CHANGE TO AVERAGE DEPTH
            const DEPTH_TYPE center_depth = depth_frame.at<DEPTH_TYPE>(cvRound(center.y), cvRound(center.x));
            if (center_depth == 0){
                continue;
            }

            trackers.insert_tracker(center, center_depth, hsv_frame, depth_frame, ellipses);
        }

        uint64_t tracking_t0 = cv::getTickCount();

        trackers.tracking_step(hsv_frame, depth_frame, gradient_vectors, observation, PF, ellipses, reg);

        float tracking_t = (cv::getTickCount() - tracking_t0) / double(cv::getTickFrequency());

        //std::cout << "TIMES_TRACKING " << tracking_t << std::endl;

#define VISUALIZATION
#ifdef VISUALIZATION
        uint64_t visualization_t0 = cv::getTickCount();

        for (auto circle : circles){
            cv::Point center(cvRound(circle[0]), cvRound(circle[1]));
            const DEPTH_TYPE center_depth = depth_frame.at<DEPTH_TYPE>(cvRound(center.y), cvRound(center.x));
            if (center_depth == 0){
                continue;
            }

            Eigen::Vector2i torso_center = translate_2D_vector_in_3D_space(center.x, center.y, center_depth, HEAD_TO_TORSE_CENTER_VECTOR, reg.cameraMatrixColor, reg.lookupX, reg.lookupY);

            const cv::Size ellipse_axes = ellipses.get_ellipse_size(BodyPart::TORSO, center_depth);
            const cv::Rect torso_roi = cv::Rect(cvRound(torso_center[0] - ellipse_axes.width * 0.5f),
                                                   cvRound(torso_center[1] - ellipse_axes.height * 0.5f),
                                                   ellipse_axes.width, ellipse_axes.height);

            const bool valid = rect_fits_in_frame(torso_roi, hsv_frame);
            const cv::Scalar color = valid ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255);
            cv::ellipse(color_display_frame, cv::Point(torso_center[0], torso_center[1]), cv::Size(ellipse_axes.width * 0.5, ellipse_axes.height * 0.5), 0, 0, 360, color, -3, 8, 0);
        }

        for(auto &state : trackers.states) {
            const DEPTH_TYPE center_depth = depth_frame.at<DEPTH_TYPE>(cvRound(state.center.y), cvRound(state.center.x));
            if (center_depth == 0){
                continue;
            }

            Eigen::Vector2i torso_center = translate_2D_vector_in_3D_space(state.center.x, state.center.y, center_depth, HEAD_TO_TORSE_CENTER_VECTOR, reg.cameraMatrixColor, reg.lookupX, reg.lookupY);

            const cv::Size ellipse_axes = ellipses.get_ellipse_size(BodyPart::TORSO, center_depth);
            const cv::Rect torso_roi = cv::Rect(cvRound(torso_center[0] - ellipse_axes.width * 0.5f),
                                                   cvRound(torso_center[1] - ellipse_axes.height * 0.5f),
                                                   ellipse_axes.width, ellipse_axes.height);

            const bool valid = rect_fits_in_frame(torso_roi, hsv_frame);
            const cv::Scalar color = valid ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255);
            cv::ellipse(color_display_frame, cv::Point(torso_center[0], torso_center[1]), cv::Size(ellipse_axes.width * 0.5, ellipse_axes.height * 0.5), 0, 0, 360, color, -3, 8, 0);
        }

        trackers.show(color_display_frame, depth_frame);
        trackers.show(gradient_magnitude_scaled, depth_frame);

        std::vector<std::unique_ptr<IplImage>> imp_image_pointers;

        if (trackers.states.size()){
            if (rect_fits_in_frame(trackers.states[0].region, color_frame)){
                imp_image_pointers.push_back(std::unique_ptr<IplImage>(new IplImage(color_frame(trackers.states[0].region))));
                CImage model_image;
                model_image.setFromIplImageReadOnly(imp_image_pointers.back().get());
                model_image_window.showImage(model_image);

                CImage model_histogram_image;
                imp_image_pointers.push_back(std::unique_ptr<IplImage>(new IplImage(histogram_to_image(trackers.states[0].color_model, 10))));
                model_histogram_image.setFromIplImageReadOnly(imp_image_pointers.back().get());
                model_histogram_window.showImage(model_histogram_image);

                CImage model_histogram_torso_image;
                imp_image_pointers.push_back(std::unique_ptr<IplImage>(new IplImage(histogram_to_image(trackers.states[0].torso_color_model, 10))));
                model_histogram_torso_image.setFromIplImageReadOnly(imp_image_pointers.back().get());
                model_histogram_torso_window.showImage(model_histogram_torso_image);
            }
            /*
            //histograms
            {
                auto &tpf = trackers.trackers[0];
                std::vector<double> x;
                std::vector<double> hits;

                std::cout << "image_hist_score" << std::endl;
                CImage image_hist_score;
                tpf.hist_score.getHistogram(x, hits);
                imp_image_pointers.push_back(std::unique_ptr<IplImage>(new IplImage(histogram_to_image(x, hits))));
                image_hist_score.setFromIplImageReadOnly(imp_image_pointers.back().get());
                image_hist_score_window.showImage(image_hist_score);
                x.clear();
                hits.clear();

                std::cout << "image_hist_head_color_score" << std::endl;
                CImage image_hist_head_color_score;
                tpf.hist_head_color_score.getHistogram(x, hits);
                imp_image_pointers.push_back(std::unique_ptr<IplImage>(new IplImage(histogram_to_image(x, hits))));
                image_hist_head_color_score.setFromIplImageReadOnly(imp_image_pointers.back().get());
                image_hist_head_color_score_window.showImage(image_hist_head_color_score);
                x.clear();
                hits.clear();

                std::cout << "image_hist_head_fitting_score" << std::endl;
                CImage image_hist_head_fitting_score;
                tpf.hist_head_fitting_score.getHistogram(x, hits);
                imp_image_pointers.push_back(std::unique_ptr<IplImage>(new IplImage(histogram_to_image(x, hits))));
                image_hist_head_fitting_score.setFromIplImageReadOnly(imp_image_pointers.back().get());
                image_hist_head_fitting_score_window.showImage(image_hist_head_fitting_score);
                x.clear();
                hits.clear();

                std::cout << "image_hist_head_z_score" << std::endl;
                CImage image_hist_head_z_score;
                tpf.hist_head_z_score.getHistogram(x, hits);
                imp_image_pointers.push_back(std::unique_ptr<IplImage>(new IplImage(histogram_to_image(x, hits))));
                image_hist_head_z_score.setFromIplImageReadOnly(imp_image_pointers.back().get());
                image_hist_head_z_score_window.showImage(image_hist_head_z_score);
                x.clear();
                hits.clear();

                std::cout << "image_hist_chest_color_score" << std::endl;
                CImage image_hist_chest_color_score;
                tpf.hist_chest_color_score.getHistogram(x, hits);
                imp_image_pointers.push_back(std::unique_ptr<IplImage>(new IplImage(histogram_to_image(x, hits))));
                image_hist_chest_color_score.setFromIplImageReadOnly(imp_image_pointers.back().get());
                image_hist_chest_color_score_window.showImage(image_hist_chest_color_score);
                x.clear();
                hits.clear();
            }
            */

            /*
            const cv::Mat mask_weight = ellipses->get_ellipse_mask_weights(trackers.states[0].z);
            cv::Mat m;
            mask_weight *= 255;
            mask_weight.convertTo(m, CV_8UC1);
            model_histogram_image.loadFromIplImage(new IplImage(m));
            model_histogram_window.showImage(model_histogram_image);
            */
        }

        //visualization
        //cv::Mat depthDisp;
        //dispDepth(registered_depth, depthDisp, 12000.0f);
        //cv::Mat combined;
        //combine(color_mat, depthDisp, combined);
        //int a = 140;
        //int b = 290;
        //cv::Mat combined2 = combined(cv::Rect(a, 0, combined.cols - a - b, combined.rows));
        //cv::line(combined, cv::Point(color_display_frame.cols * 0.5, 0), cv::Point(color_display_frame.cols * 0.5, color_display_frame.rows - 1), cv::Scalar(0, 0, 255));
        //cv::line(combined, cv::Point(0, color_display_frame.rows * 0.5), cv::Point(color_display_frame.cols - 1, color_display_frame.rows * 0.5), cv::Scalar(0, 255, 0));

        //CImage registered_depth_image;
        //registered_depth_image.loadFromIplImage(new IplImage(combined));
        //registered_color_window.showImage(registered_depth_image);

        CImage gradient_magnitude_image;
        imp_image_pointers.push_back(std::unique_ptr<IplImage>(new IplImage(gradient_magnitude_scaled)));
        gradient_magnitude_image.setFromIplImageReadOnly(imp_image_pointers.back().get());
        gradient_color_window.showImage(gradient_magnitude_image);

        {
            std::ostringstream oss;
            oss << "FPS " << fps;

            int fontFace =  cv::FONT_HERSHEY_PLAIN;
            double fontScale = 2;
            int thickness = 2;

            int baseline = 0;
            cv::Size textSize = cv::getTextSize(oss.str(), fontFace, fontScale, thickness, &baseline);
            cv::Point textOrg(color_display_frame.cols - 100, color_display_frame.cols - textSize.height * 0.5f - 50);
            putText(color_display_frame, oss.str(), textOrg, fontFace, fontScale, cv::Scalar(255, 255, 0), thickness, 8);
        }


        CImage color_display_image;
        imp_image_pointers.push_back(std::unique_ptr<IplImage>(new IplImage(color_display_frame)));
        color_display_image.setFromIplImageReadOnly(imp_image_pointers.back().get());
        image.showImage(color_display_image);

#ifdef VIEW_3D
        //--- 3D view stuff
        create_cloud(color_frame, depth_frame, 1.0/1000.0f, reg, scene_points_map);

        //std::vector<Eigen::Vector2f> particle_vectors(N);
        //std::vector<Eigen::Vector3f> particle_vectors3d(N);
        //std::vector<Eigen::Vector3f> particle_vectors3d_2(N);

        //for (size_t i = 0; i < N; i++) {
            //particle_vectors[i] = Eigen::Vector2f(particles.m_particles[i].d->x, particles.m_particles[i].d->y);
            //particle_vectors3d_2[i] = Eigen::Vector3f(particles.m_particles[i].d->x, particles.m_particles[i].d->y, particles.m_particles[i].d->z);
            //particle_vectors3d[i] = point_3D_reprojection(particle_vectors[i], particles.m_particles[i].d->z, reg.lookupX, reg.lookupY);
        //}

        //std::vector<Eigen::Vector3f> points_3d = points_3D_reprojection<DEPTH_DATA_TYPE>(particle_vectors, depth_frame, reg.lookupX, reg.lookupY);
        //std::vector<Eigen::Vector3f> particle_points_3d = pixel_depth_to_3D_coordiantes(particle_vectors3d_2, reg.cameraMatrixColor);
        //create_cloud(points_3d, particle_points_map);
        //create_cloud(particle_vectors3d, particle_points_map);
        //points_3d[0] = pixel_depth_to_3D_coordiantes(Eigen::Vector3f(x_global, y_global, depth_frame.at<DEPTH_TYPE>(y_global, x_global)), reg.cameraMatrixColor);
        //create_cloud(particle_points_3d, 1.0/1000.0f, particle_points_map);

        win3D.get3DSceneAndLock();
        scene_points->loadFromPointsMap(&scene_points_map);
        //particle_points->loadFromPointsMap(&particle_points_map);
        //model_center_points->loadFromPointsMap(&model_center_points_map);
        win3D.unlockAccess3DScene();
        win3D.repaint();
#endif
        float visualization_t = (cv::getTickCount() - visualization_t0) / double(cv::getTickFrequency());
        //std::cout << "TIMES_VISUALIZATION " << visualization_t << std::endl;
#endif
        counter++;
        if (counter > 30){
            time(&end);
            sec = difftime(end, start);
            fps = counter/sec;
            printf("%.2f fps\n", fps);
        }

        if (counter == (INT_MAX - 1000)){
            counter = 0;
        }


        //cout << "TIME_TOTAL " << read_kinect_t + color_conversion_t + sobel_t + viola_t + tracking_t << std::endl;
    }
    video_feed.close();

    return 0;
}


int main(int argc, char *argv[])
{
    if (argc > 1) {
        NUM_PARTICLES = atof(argv[1]);
    } else {
        NUM_PARTICLES = 1000;
    }

    if (argc > 2) {
        MODEL_TRANSITION_STD_XY = atof(argv[2]);
    } else {
        MODEL_TRANSITION_STD_XY = 10;
    }

    if (argc > 3) {
        MODEL_TRANSITION_STD_VXY  = atof(argv[3]);
    } else {
        MODEL_TRANSITION_STD_VXY  = 10;
    }

    std::cout << "NUM_PARTICLES: " << NUM_PARTICLES << " MODEL_TRANSITION_STD_XY: " << MODEL_TRANSITION_STD_XY << " MODEL_TRANSITION_STD_VXY: " << MODEL_TRANSITION_STD_VXY << std::endl;

    cv::redirectError(handle_OpenCV_error);

    particle_filter();

    return 0;

    /*
    try {
        particle_filter();
        return 0;
    } catch (std::exception &e) {
        std::cout << "MRPT exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        printf("Untyped exception!!");
        return -1;
    }
    */
}

