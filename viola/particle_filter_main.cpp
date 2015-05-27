#include <signal.h>
#include <cstdlib>

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

#include "ImageRegistration.h"

IGNORE_WARNINGS_POP

//#include "KinectCamera.h"
#include "CImageParticleFilter.h"
#include "misc_helpers.h"
#include "geometry_helpers.h"
#include "color_model.h"
#include "faces_detection.h"

using namespace mrpt;
using namespace mrpt::bayes;
using namespace mrpt::gui;
using namespace mrpt::obs;
using namespace mrpt::random;
using namespace mrpt::opengl;
using namespace mrpt::maps;

using namespace std;

double TRANSITION_MODEL_STD_XY   = 0;
double TRANSITION_MODEL_STD_VXY  = 0;
double NUM_PARTICLES             = 0;

libfreenect2::Freenect2Device *dev;
libfreenect2::Freenect2 freenect2;
libfreenect2::SyncMultiFrameListener *listener;
libfreenect2::FrameMap frames_kinect2;
libfreenect2::PacketPipeline *pipeline;

void close_kinect2_handler(void)
{
    dev->stop();
    dev->close();
    listener->release(frames_kinect2);
    delete pipeline;
    delete listener;
}

/*
void grab_kinect2_frames(){
    listener->release(frames_kinect2);
    listener->waitForNewFrame(frames_kinect2);
    const libfreenect2::Frame *rgb = frames_kinect2[libfreenect2::Frame::Color];
    const libfreenect2::Frame *depth = frames_kinect2[libfreenect2::Frame::Depth];
    const libfreenect2::Frame *ir = frames_kinect2[libfreenect2::Frame::Ir];
    cv::Mat color = cv::Mat(rgb->height, rgb->width, CV_8UC3, rgb->data);
    cv::Mat depth = cv::Mat(depth->height, depth->width, CV_32FC1, depth->data);
    cv::Mat ir = cv::Mat(depth->height, depth->width, CV_32FC1, ir->data);
}
*/

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

    //#pragma omp parallel for
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

void create_cloud(const cv::Mat &color, const cv::Mat &depth, const ImageRegistration &reg, CColouredPointsMap &cloud)
{
    //const float badPoint = std::numeric_limits<float>::quiet_NaN();
    cloud.clear();
    //#pragma omp parallel for
    for (int r = 0; r < depth.rows; ++r) {
        const uint16_t *itD = depth.ptr<uint16_t>(r);
        const cv::Vec3b *itC = color.ptr<cv::Vec3b>(r);
        const float y = reg.lookupY.at<float>(0, r);
        const float *itX = reg.lookupX.ptr<float>();

        for (size_t c = 0; c < (size_t)depth.cols; ++c, ++itD, ++itC, ++itX) {
            register const float depthValue = *itD / 1000.0f;
            // Check for invalid measurements
            if (isnan(depthValue) || depthValue <= 0.001) {
                continue;
            }

            const float z_coord = depthValue;
            const float x_coord = (*itX) * depthValue;
            const float y_coord = y * depthValue;
            
            cloud.insertPoint(-z_coord, x_coord, -y_coord , itC->val[2] / 255.0, itC->val[1] / 255.0, itC->val[0] / 255.0);
        }
    }
}

inline Eigen::Vector3f point_3D_reprojection (const Eigen::Vector2f &point, const float depth, const ImageRegistration &reg)
{
    static const float badPoint = std::numeric_limits<float>::quiet_NaN();
    const float x = reg.lookupX.at<float>(0, point[0]);
    const float y = reg.lookupY.at<float>(0, point[1]);
    register const float depthValue = depth / 1000.0f;

    // Check for invalid measurements
    if (isnan(depthValue) || depthValue <= 0.001) {
        return Eigen::Vector3f(badPoint, badPoint, badPoint);
    }
    
    const float z_coord = depthValue;
    const float x_coord = x * depthValue;
    const float y_coord = y * depthValue;

    return Eigen::Vector3f(x_coord, y_coord, z_coord);
}

inline Eigen::Vector3f point_3D_reprojection(const Eigen::Vector2f &point, const cv::Mat &depth, const ImageRegistration &reg)
{
    return point_3D_reprojection(point, depth.at<uint16_t>(point[1], point[0]), reg);
}

std::vector<Eigen::Vector3f> points_3D_reprojection(const std::vector<Eigen::Vector2f> &points, const cv::Mat &depth, const ImageRegistration &reg, CColouredPointsMap &cloud)
{
    cloud.clear();
    std::vector<Eigen::Vector3f> reprojected_points;
    //#pragma omp parallel for
    for (auto vector : points){
        auto point_3d = point_3D_reprojection(vector, depth, reg);
        cloud.insertPoint(-point_3d[2], point_3d[0], -point_3d[1], 0, 1, 0);
        reprojected_points.push_back(point_3d);
    }
    
    return reprojected_points;
}

int particle_filter()
{

    if (freenect2.enumerateDevices() == 0) {
        std::cout << "no device connected!" << std::endl;
        return -1;
    }

    std::string serial = freenect2.getDefaultDeviceSerialNumber();

    char *calib_dir = getenv("HOME");
    std::string calib_path = std::string(calib_dir) + "/kinect2_calib/";

    ImageRegistration reg;
    reg.init(calib_path, serial);

    pipeline = new libfreenect2::OpenCLPacketPipeline();

    const float minDepth = 0;
    const float maxDepth = 12.0;
    const bool bilateral_filter = true;
    const bool edge_aware_filter = true;

    libfreenect2::DepthPacketProcessor::Config config;
    config.EnableBilateralFilter = bilateral_filter;
    config.EnableEdgeAwareFilter = edge_aware_filter;
    config.MinDepth = minDepth;
    config.MaxDepth = maxDepth;

    pipeline->getDepthPacketProcessor()->setConfiguration(config);
    dev = freenect2.openDevice(serial, pipeline);

    if (dev == nullptr) {
        std::cout << "no device connected or failure opening the default one!" << std::endl;
        exit(-1);
    }

    listener = new libfreenect2::SyncMultiFrameListener(libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);
    dev->setColorFrameListener(listener);
    dev->setIrAndDepthFrameListener(listener);
    dev->start();

    struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = [](int) {
        std::cout << "SIGINT" << std::endl;
        close_kinect2_handler();
        std::cout << "SIGINT2" << std::endl;
        exit(0);
    };

    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
    sigaction(SIGQUIT, &sigIntHandler, NULL);
    sigaction(SIGSEGV, &sigIntHandler, NULL);

    //atexit(close_kinect2_handler);
    at_quick_exit(close_kinect2_handler);

    cv::Mat color_frame;
    cv::Mat color_display_frame;
    cv::Mat model_frame;
    cv::Mat depth_frame;

    randomGenerator.randomize();
    CDisplayWindow image("image");

    CDisplayWindow model_window("model");
    CDisplayWindow model_image_window("model-image");
    CDisplayWindow depth_window("depth_window");
    CDisplayWindow registered_color("registered_depth");

    // -------------------3D view stuff------------------
#define _3D
#ifdef _3D
    CDisplayWindow3D win3D("Kinect 3D view", 800, 600);

    win3D.setCameraAzimuthDeg(0);
    win3D.setCameraElevationDeg(0);
    win3D.setCameraZoom(1);
    win3D.setFOV(90);
    win3D.setCameraPointingToPoint(0, 0, 0);

    mrpt::opengl::CPointCloudColouredPtr scene_points = mrpt::opengl::CPointCloudColoured::Create();
    mrpt::opengl::CPointCloudColouredPtr particle_points = mrpt::opengl::CPointCloudColoured::Create();
    scene_points->setPointSize(0.5);
    particle_points->setPointSize(2);

    {
        mrpt::opengl::COpenGLScenePtr &scene = win3D.get3DSceneAndLock();
        scene->insert(scene_points);
        scene->insert(particle_points);
        //scene->insert(mrpt::opengl::CGridPlaneXY::Create());
        //scene->insert(mrpt::opengl::stock_objects::CornerXYZ());
        win3D.unlockAccess3DScene();
        win3D.repaint();
    }

    CColouredPointsMap scene_points_map;
    CColouredPointsMap particle_points_map;
    scene_points_map.colorScheme.scheme = CColouredPointsMap::cmFromIntensityImage;
#endif

    // Create PF
    // ----------------------
    CParticleFilter::TParticleFilterOptions PF_options;
    PF_options.adaptiveSampleSize = false;
    PF_options.PF_algorithm = CParticleFilter::pfStandardProposal;
    //PF_options.resamplingMethod = CParticleFilter::prSystematic;
    PF_options.resamplingMethod = CParticleFilter::prMultinomial;

    CParticleFilter PF;
    PF.m_options = PF_options;

    CImageParticleFilter particles;

    bool init_model = true;

    while (!mrpt::system::os::kbhit()) {
        listener->release(frames_kinect2);
        listener->waitForNewFrame(frames_kinect2);

        libfreenect2::Frame *rgb = frames_kinect2[libfreenect2::Frame::Color];
        libfreenect2::Frame *depth = frames_kinect2[libfreenect2::Frame::Depth];
        libfreenect2::Frame *ir = frames_kinect2[libfreenect2::Frame::Ir];


        cv::Mat color_mat = cv::Mat(rgb->height, rgb->width, CV_8UC3, rgb->data);
        cv::Mat depth_mat = cv::Mat(depth->height, depth->width, CV_32FC1, depth->data);
        //cv::Mat ir_mat = cv::Mat(depth->height, depth->width, CV_32FC1, ir->data);

        cv::Mat registered_depth;
        reg.register_images(color_mat, depth_mat, registered_depth);

        cv::flip(color_mat, color_mat, 1);
        cv::Mat depthDisp;
        dispDepth(registered_depth, depthDisp, 12000.0f);
        cv::Mat combined;
        combine(color_mat, depthDisp, combined);

        CImage registered_depth_image;
        registered_depth_image.loadFromIplImage(new IplImage(combined));
        registered_color.showImage(registered_depth_image);

        color_frame = color_mat;
        depth_frame = registered_depth;
        color_display_frame = color_mat.clone();
        //color_frame = registered_color_mat;

        // Process with PF:

        //TODO: USE CObservationStereoImages?
        CObservationImagePtr obsImage_color = CObservationImage::Create();
        CObservationImagePtr obsImage_depth = CObservationImage::Create();
        obsImage_color->image.loadFromIplImage(new IplImage(color_frame));
        obsImage_depth->image.loadFromIplImage(new IplImage(depth_frame));
        // memory freed by SF.
        CSensoryFrame observation;
        observation.insert(obsImage_color);
        observation.insert(obsImage_depth);

        /*
        cv::Mat gradient = sobel_operator(color_frame);

        double min, max;
        cv::minMaxLoc(depth_frame, &min, &max);
        cv::Mat depth_frame_normalized = (depth_frame * 255)/ max;
        cv::Mat gradient_depth = sobel_operator(depth_frame_normalized);
        cv::Mat gradient_depth_8UC1 = cv::Mat(depth_frame.size(), CV_8UC1);

        gradient_depth.convertTo(gradient_depth_8UC1, CV_8UC1);
        CImage model_image;
        model_image.loadFromIplImage(new IplImage(gradient));
        model_window.showImage(model_image);

        CImage depth_image;
        depth_image.loadFromIplImage(new IplImage(gradient_depth_8UC1));
        depth_window.showImage(depth_image);
        */

        if (init_model) {
            cv::Mat frame_hsv;
            auto circles = viola_faces::detect_circles(color_frame);
            if (circles.size()) {
                int circle_max = 0;
                double radius_max = circles[0][2];
                for (size_t i = 0; i < circles.size(); i++) {
                    cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
                    int radius = cvRound(circles[i][2]);
                    cv::circle(color_frame, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
                    cv::circle(color_frame, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
                    if (radius_max < radius) {
                        radius_max = radius;
                        circle_max = i;
                    }
                }
                cv::Point center(cvRound(circles[circle_max][0]), cvRound(circles[circle_max][1]));

                int radius = cvRound(circles[circle_max][2]);
                cout << "circle " << center.x << ' ' << center.y << ' ' << radius << endl;
                cv::cvtColor(color_frame, frame_hsv, cv::COLOR_BGR2HSV);
                const cv::Rect model_roi(center.x - radius, center.y - radius, 2 * radius, 2 * radius);
                const cv::Mat mask = create_ellipse_mask(model_roi, 1);
                const cv::Mat model = compute_color_model(frame_hsv(model_roi), mask);
                particles.update_color_model(new cv::Mat(model), radius, radius);
                particles.initializeParticles(NUM_PARTICLES, make_pair(center.x, radius), make_pair(center.y,
                                              radius), make_pair(0, 0), make_pair(0, 0), make_pair(0, 0), make_pair(0, 0), &observation);
                init_model = false;
                particles.last_time = cv::getTickCount();

                model_frame = cv::Mat(color_frame(model_roi).size(), color_frame.type());
                const cv::Mat ones = cv::Mat::ones(color_frame(model_roi).size(), color_frame(model_roi).type());
                bitwise_and(color_frame(model_roi), ones, model_frame, mask);

                //cv::Mat gradient = sobel_operator(color_frame(model_roi));
                //model_window.showImage(CImage(new IplImage(gradient)));
                CImage model_frame;
                model_frame.loadFromIplImage(new IplImage(color_frame(model_roi)));
                model_image_window.showImage(model_frame);
            }
        } else {
            // Process in the PF
            static CParticleFilter::TParticleFilterStats stats;
            PF.executeOn(particles, NULL, &observation, &stats);

            // Show PF state:
            cout << "ESS_beforeResample " << stats.ESS_beforeResample << "weightsVariance_beforeResample " << stats.weightsVariance_beforeResample << std::endl;
            cout << "Particle filter ESS: " << particles.ESS() << endl;

            size_t N = particles.m_particles.size();
            for (size_t i = 0; i < N; i++) {
                cv::circle(color_display_frame, cv::Point(particles.m_particles[i].d->x,
                                                  particles.m_particles[i].d->y), 1, cv::Scalar(0, 0, 255), 1, 1, 0);
            }

            float avrg_x, avrg_y, avrg_z, avrg_vx, avrg_vy, avrg_vz;
            particles.get_mean(avrg_x, avrg_y, avrg_z, avrg_vx, avrg_vy, avrg_vz);
            cv::circle(color_display_frame, cv::Point(avrg_x, avrg_y), 20, cv::Scalar(255, 0, 0), 5, 1, 0);
            cv::line(color_display_frame, cv::Point(avrg_x, avrg_y), cv::Point(avrg_x + avrg_vx, avrg_y + avrg_vy),
                     cv::Scalar(0, 255, 0), 5, 1, 0);

            //particles.print_particle_state();
            std::cout << "MEAN " << avrg_x << ' ' << avrg_y << ' ' << avrg_z << ' ' << avrg_vx << ' ' << avrg_vy << ' ' << avrg_vz << std::endl;
        }

        CImage frame_particles;
        frame_particles.loadFromIplImage(new IplImage(color_display_frame));
        image.showImage(frame_particles);

#ifdef _3D
        //--- 3D view stuff
        create_cloud(color_frame, depth_frame, reg, scene_points_map);
        
        size_t N = particles.m_particles.size();
        std::vector<Eigen::Vector2f> particle_vectors;
        for (size_t i = 0; i < N; i++) {
            particle_vectors.push_back(Eigen::Vector2f(particles.m_particles[i].d->x, particles.m_particles[i].d->y));
        }
        points_3D_reprojection(particle_vectors, depth_frame, reg, particle_points_map);


        win3D.get3DSceneAndLock();
        scene_points->loadFromPointsMap(&scene_points_map);
        particle_points->loadFromPointsMap(&particle_points_map);
        win3D.unlockAccess3DScene();
        win3D.repaint();
#endif
        //mrpt::system::sleep(1);
    }

    close_kinect2_handler();
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
        TRANSITION_MODEL_STD_XY = atof(argv[2]);
    } else {
        TRANSITION_MODEL_STD_XY = 10;
    }

    if (argc > 3) {
        TRANSITION_MODEL_STD_VXY  = atof(argv[3]);
    } else {
        TRANSITION_MODEL_STD_VXY  = 10;
    }

    std::cout << "NUM_PARTICLES: " << NUM_PARTICLES << " TRANSITION_MODEL_STD_XY: " << TRANSITION_MODEL_STD_XY << " TRANSITION_MODEL_STD_VXY: " << TRANSITION_MODEL_STD_VXY << std::endl;
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

