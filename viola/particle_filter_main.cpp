#include <signal.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Werror"
#pragma GCC diagnostic ignored "-Wlong-long"

#pragma GCC diagnostic ignored "-pedantic"
#pragma GCC diagnostic ignored "-pedantic-errors"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <mrpt/gui/CDisplayWindow.h>
#include <mrpt/random.h>
#include <mrpt/bayes/CParticleFilterData.h>
#include <mrpt/obs/CSensoryFrame.h>
#include <mrpt/obs/CObservationImage.h>
#include <mrpt/otherlibs/do_opencv_includes.h>

//#define USE_KINECT_2
#ifdef USE_KINECT_2
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#endif

#pragma GCC diagnostic pop

#include "CImageParticleFilter.h"
#include "misc_helpers.h"
#include "geometry_helpers.h"
#include "color_model.h"

using namespace mrpt;
using namespace mrpt::bayes;
using namespace mrpt::gui;
using namespace mrpt::obs;
using namespace mrpt::random;

using namespace std;

double TRANSITION_MODEL_STD_XY   = 0;
double TRANSITION_MODEL_STD_VXY  = 0;
double NUM_PARTICLES             = 0;

#ifndef USE_KINECT_2
cv::VideoCapture capture;
#endif

vector<cv::Vec3f> detect_circles(const cv::Mat &image);

vector<cv::Vec3f> detect_circles(const cv::Mat &image)
{
    using namespace cv;
    Mat src_gray;
    Mat color = image.clone();
    cvtColor(image, src_gray, CV_BGR2GRAY);
    GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);

    vector<Vec3f> circles;
    HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows / 8, 200, 100, 50, 0);
    return circles;
}

void TestBayesianTracking()
{
    cv::Mat color_frame;
    cv::Mat model_frame;
    cv::Mat depth_frame;

#ifdef USE_KINECT_2
    libfreenect2::Freenect2 freenect2;
    std::cout << "kinect2" << std::endl;
    libfreenect2::Freenect2Device *dev = freenect2.openDefaultDevice();

    if (dev == nullptr) {
        std::cout << "no device connected or failure opening the default one!" << std::endl;
        return;
    }

    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color |
            libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);
    libfreenect2::FrameMap frames;

    dev->setColorFrameListener(&listener);
    dev->setIrAndDepthFrameListener(&listener);
    dev->start();

    std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
    std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;
#else
    capture.open(CV_CAP_OPENNI);
    capture.set(CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_SXGA_15HZ);
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = [](int){ capture.release(); exit(0); };
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
    sigaction(SIGQUIT, &sigIntHandler, NULL);
    sigaction(SIGSEGV, &sigIntHandler, NULL);

    if (!capture.isOpened()) {
        return;
    }
#endif

    randomGenerator.randomize();
    CDisplayWindow image("image");
    CDisplayWindow model_window("model");
    CDisplayWindow model_image_window("model-image");
    CDisplayWindow depth_window("depth_window");

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
#ifdef USE_KINECT_2
        listener.waitForNewFrame(frames);
        libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
        libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
        libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];
        color_frame = cv::Mat(rgb->height, rgb->width, CV_8UC3, rgb->data);
        depth_frame = cv::Mat(depth->height, depth->width, CV_32FC1, depth->data);
#else //kinect 1
        capture.grab();
        capture.retrieve(color_frame, CV_CAP_OPENNI_BGR_IMAGE);
        capture.retrieve(depth_frame, CV_CAP_OPENNI_DEPTH_MAP);
        /*
        capture.retrieve(grey_frame, CV_CAP_OPENNI_GRAY_IMAGE);
        capture.retrieve(disparity_map, CV_CAP_OPENNI_DISPARITY_MAP);
        capture.retrieve(depth_frame, CV_CAP_OPENNI_DEPTH_MAP);
        capture.retrieve(valid_depth_pixels, CV_CAP_OPENNI_VALID_DEPTH_MASK);
        */
#endif

        // Process with PF:
        CObservationImagePtr obsImage = CObservationImage::Create();
        CObservationImagePtr obsImage2 = CObservationImage::Create();
        obsImage->image.loadFromIplImage(new IplImage(color_frame));
        obsImage2->image.loadFromIplImage(new IplImage(depth_frame));

        // memory freed by SF.
        CSensoryFrame SF;
        SF.insert(obsImage);
        SF.insert(obsImage2);

        cv::Mat gradient = sobel_operator(color_frame);

        double min, max;
        cv::minMaxLoc(depth_frame, &min, &max);
        cv::Mat depth_frame_normalized = (depth_frame * 255)/ max;
        cv::Mat gradient_depth = sobel_operator(depth_frame_normalized);
        cv::Mat gradient_depth_8UC1 = cv::Mat(depth_frame.size(), CV_8UC1);
        
        gradient_depth.convertTo(gradient_depth_8UC1, CV_8UC1);
        CImage model_image;
        model_image.loadFromIplImage(new IplImage(gradient));
        CImage depth_image;
        depth_image.loadFromIplImage(new IplImage(gradient_depth_8UC1));
        model_window.showImage(model_image);
        depth_window.showImage(depth_image);

        if (init_model) {
            cv::Mat frame_hsv;
            auto circles = detect_circles(color_frame);
            if (circles.size()) {
                int circle_max = 0;
                double radius_max = circles[0][2];
                for (size_t i = 0; i < circles.size(); i++) {
                    cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
                    int radius = cvRound(circles[i][2]);
                    cv::circle(color_frame, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
                    cv::circle(color_frame, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
                    if (radius_max < radius){
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
                                              radius), make_pair(0, 0), make_pair(0, 0), make_pair(0, 0), make_pair(0, 0), &SF);
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
            PF.executeOn(particles, NULL, &SF);

            // Show PF state:
            cout << "Particle filter ESS: " << particles.ESS() << endl;


            size_t N = particles.m_particles.size();
            for (size_t i = 0; i < N; i++) {
                particles.m_particles[i].d->x;
                particles.m_particles[i].d->y;
                cv::circle(color_frame, cv::Point(particles.m_particles[i].d->x,
                                                  particles.m_particles[i].d->y), 1, cv::Scalar(0, 0, 255), 1, 1, 0);
            }

            float avrg_x, avrg_y, avrg_z, avrg_vx, avrg_vy, avrg_vz;
            particles.get_mean(avrg_x, avrg_y, avrg_z, avrg_vx, avrg_vy, avrg_vz);
            cv::circle(color_frame, cv::Point(avrg_x, avrg_y), 20, cv::Scalar(255, 0, 0), 5, 1, 0);
            cv::line(color_frame, cv::Point(avrg_x, avrg_y), cv::Point(avrg_x + avrg_vx, avrg_y + avrg_vy),
                     cv::Scalar(0, 255, 0), 5, 1, 0);
            
            //particles.print_particle_state();
            std::cout << "MEAN " << avrg_x << ' ' << avrg_y << ' ' << avrg_z << ' ' << avrg_vx << ' ' << avrg_vy << ' ' << avrg_vz << std::endl;
        }
    
        CImage frame_particles;
        frame_particles.loadFromIplImage(new IplImage(color_frame));
        image.showImage(frame_particles);
#ifdef USE_KINECT_2
        listener.release(frames);
#endif
    }
}


int main(int argc, char *argv[])
{
    if (argc > 1){
        NUM_PARTICLES = atof(argv[1]);
    }else{
        NUM_PARTICLES = 1000;
    }

    if (argc > 2){
        TRANSITION_MODEL_STD_XY = atof(argv[2]);
    }else{
        TRANSITION_MODEL_STD_XY = 10;
    }

    if (argc > 3){
        TRANSITION_MODEL_STD_VXY  = atof(argv[3]);
    }else{
        TRANSITION_MODEL_STD_VXY  = 10;
    }

    std::cout << "NUM_PARTICLES: " << NUM_PARTICLES << " TRANSITION_MODEL_STD_XY: " << TRANSITION_MODEL_STD_XY << " TRANSITION_MODEL_STD_VXY: " << TRANSITION_MODEL_STD_VXY << std::endl;
    TestBayesianTracking();

    return 0;

    /*
    try {
        TestBayesianTracking();
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
