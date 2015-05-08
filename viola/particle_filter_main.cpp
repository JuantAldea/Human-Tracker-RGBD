#include <signal.h>
#include <cstdlib>

#include "project_config.h"

IGNORE_WARNINGS_PUSH

#include <mrpt/gui/CDisplayWindow.h>
#include <mrpt/random.h>
#include <mrpt/bayes/CParticleFilterData.h>
#include <mrpt/obs/CSensoryFrame.h>
#include <mrpt/obs/CObservationImage.h>
#include <mrpt/otherlibs/do_opencv_includes.h>

IGNORE_WARNINGS_POP

#include "KinectCamera.h"
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

using namespace std;

double TRANSITION_MODEL_STD_XY   = 0;
double TRANSITION_MODEL_STD_VXY  = 0;
double NUM_PARTICLES             = 0;

KinectCamera camera;

void TestBayesianTracking()
{
    camera.open();
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = [](int){std::cout << "SIGINT" << std::endl; camera.close(); std::cout << "SIGINT2" << std::endl;  exit(0); };
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
    sigaction(SIGQUIT, &sigIntHandler, NULL);
    sigaction(SIGSEGV, &sigIntHandler, NULL);

    cv::Mat color_frame;
    cv::Mat model_frame;
    cv::Mat depth_frame;

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
        camera.grabFrames();
        color_frame = camera.frames[KinectCamera::FrameType::COLOR];
        depth_frame = camera.frames[KinectCamera::FrameType::DEPTH];

        // Process with PF:
        CObservationImagePtr obsImage = CObservationImage::Create();
        CObservationImagePtr obsImage2 = CObservationImage::Create();
        obsImage->image.loadFromIplImage(new IplImage(color_frame));
        obsImage2->image.loadFromIplImage(new IplImage(depth_frame));
        // memory freed by SF.
        CSensoryFrame observation;
        observation.insert(obsImage);
        observation.insert(obsImage2);
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
            auto circles = viola_faces::detect_circles(color_frame);
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
    }

    camera.close();
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

