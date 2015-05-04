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
#include <mrpt/gui/CDisplayWindowPlots.h>
#include <mrpt/random.h>
#include <mrpt/system/os.h>
#include <mrpt/system/threads.h>
#include <mrpt/math/wrap2pi.h>
#include <mrpt/math/distributions.h>
#include <mrpt/bayes/CParticleFilterData.h>

#include <mrpt/obs/CSensoryFrame.h>
#include <mrpt/obs/CObservationImage.h>
#include <mrpt/utils/CSerializable.h>

#include <mrpt/otherlibs/do_opencv_includes.h>

#include <tbb/tbb.h>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>


#pragma GCC diagnostic pop

#include "misc_helpers.h"
#include "geometry_helpers.h"
#include "color_model.h"

using namespace mrpt;
using namespace mrpt::bayes;
using namespace mrpt::gui;
using namespace mrpt::math;
using namespace mrpt::obs;
using namespace mrpt::utils;
using namespace mrpt::random;
using namespace std;

double TRANSITION_MODEL_STD_XY   = 0;
double TRANSITION_MODEL_STD_VXY  = 0;
double NUM_PARTICLES             = 0;

#define USE_INTEL_TBB
#ifdef USE_INTEL_TBB
#define TBB_PARTITIONS 8
#endif

vector<cv::Vec3f> detect_circles(const cv::Mat &image);

// ---------------------------------------------------------------
//      Implementation of the system models as a Particle Filter
// ---------------------------------------------------------------
struct CImageParticleData {
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
};

class CImageParticleFilter :
    public mrpt::bayes::CParticleFilterData<CImageParticleData>,
    public mrpt::bayes::CParticleFilterDataImpl < CImageParticleFilter,
    mrpt::bayes::CParticleFilterData<CImageParticleData>::CParticleList >
{
public:
    void update_particles_with_transition_model(double dt);
    void weight_particles_with_model(const CImage &observation);

    void prediction_and_update_pfStandardProposal(
        const mrpt::obs::CActionCollection*,
        const mrpt::obs::CSensoryFrame *observation,
        const bayes::CParticleFilter::TParticleFilterOptions&);

    void initializeParticles(const size_t M, const pair<float, float> x,
                             const pair<float, float> y, const pair<float, float> z,
                             const pair<float, float> v_x, const pair<float, float> v_y, const pair<float, float> v_z);

    void getMean(float &x, float &y, float &z, float &vx, float &vy, float &vz);

    void update_color_model(cv::Mat *model, const int roi_width, const int roi_height);

    int64_t last_time;
private:
    //TODO POTENTIAL LEAK! USE smartptr
    cv::Mat *color_model;
    int roi_width;
    int roi_height;

};


void CImageParticleFilter::update_color_model(cv::Mat *model, const int roi_width,
        const int roi_height)
{
    color_model = model;
    this->roi_width = roi_width;
    this->roi_height = roi_height;
}

void CImageParticleFilter::update_particles_with_transition_model(const double dt)
{
    size_t N = m_particles.size();

#ifndef USE_INTEL_TBB
    for (size_t i = 0; i < N; i++) {
        m_particles[i].d->x += dt * m_particles[i].d->vx + TRANSITION_MODEL_STD_XY *
                               randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->y += dt * m_particles[i].d->vy + TRANSITION_MODEL_STD_XY *
                               randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->z += dt * m_particles[i].d->vz + TRANSITION_MODEL_STD_XY *
                               randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->vx += TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->vy += TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->vz += TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
    }
#else
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N,
    N / TBB_PARTITIONS), [&](const tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i != r.end(); i++) {
            m_particles[i].d->x += dt * m_particles[i].d->vx + TRANSITION_MODEL_STD_XY *
                                   randomGenerator.drawGaussian1D_normalized();
            m_particles[i].d->y += dt * m_particles[i].d->vy + TRANSITION_MODEL_STD_XY *
                                   randomGenerator.drawGaussian1D_normalized();
            m_particles[i].d->z += dt * m_particles[i].d->vz + TRANSITION_MODEL_STD_XY *
                                   randomGenerator.drawGaussian1D_normalized();
            m_particles[i].d->vx += TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
            m_particles[i].d->vy += TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
            m_particles[i].d->vz += TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
        }
    });
#endif
}

void CImageParticleFilter::weight_particles_with_model(const CImage &observation)
{
    const cv::Mat image_mat = cv::Mat(observation.getAs<IplImage>());
    size_t N = m_particles.size();

    //const int roi_width = 100;
    //const int roi_height  = 50;
    // TODO ROI for conversion -> use particle poses to determinate its dimensions
    //cv::Rect particles_roi = cv::Rect(particles_x_min, particles_y_min, particles_x_max - particles_x_min, particles_y_max - particles_y_min);
    //cout << "PARTICLES ROI " << particles_roi.x << ' ' << particles_roi.y << ' ' << particles_roi.width << ' ' << particles_roi.height << endl;
    //cv::Mat particles_roi_img = image_mat(particles_roi);
    cv::Mat frame_hsv;
    cv::cvtColor(image_mat, frame_hsv, cv::COLOR_BGR2HSV);

#ifndef USE_INTEL_TBB
    vector <cv::Mat> particles_color_model(N);
    for (size_t i = 0; i < N; i++) {
        /*
        if (!particles_with_roi_inside_image[i]){
            continue;
        }
        const cv::Rect particle_roi(m_particles[i].d->x - particles_roi.x - roi_width * 0.5, m_particles[i].d->y - particles_roi.y - roi_height * 0.5, roi_width, roi_height);
        */
        const cv::Rect particle_roi(m_particles[i].d->x - roi_width * 0.5,
                                    m_particles[i].d->y - roi_height * 0.5, roi_width, roi_height);
        //cout << particle_roi.x << ' ' << particle_roi.y << ' ' << particle_roi.width << ' ' << particle_roi.height << endl;
        //const cv::Rect particle_roi(100, 100, 100, 50);

        if (particle_roi.x < 0 || particle_roi.y < 0 || particle_roi.width <= 0
                || particle_roi.height <= 0) {
            continue;
        }

        if (particle_roi.x + particle_roi.width >= frame_hsv.cols
                || particle_roi.y + particle_roi.height >= frame_hsv.rows) {
            continue;
        }

        const cv::Mat mask = create_ellipse_mask(particle_roi, 1);
        const cv::Mat particle_roi_img = frame_hsv(particle_roi);

        // THIS NEEDS HEAVY OPTIMIZATION, most of the time is wasted here, in the vector insertion;
        particles_color_model[i] = compute_color_model(particle_roi_img, mask);

        /*
        cout << i << endl;
        cout << particle_roi.x << ' ' << particle_roi.y << ' ' << particle_roi.width << ' ' << particle_roi.height << endl;

        CImage img = CImage(new IplImage(frame_hsv(particle_roi)));
        model_window1.showImage(img);

        cv::Mat particle_histogram = histogram_to_image(particles_color_model[i], 10);
        CImage img2 = CImage(new IplImage(particle_histogram));
        model_window2.showImage(img2);

        cv::Mat model_histogram = histogram_to_image(*color_model, 10);
        CImage img3 = CImage(new IplImage(model_histogram));
        model_window3.showImage(img3);
        cout << cv::compareHist(*color_model, particles_color_model[i], CV_COMP_BHATTACHARYYA);
        int a;
        cin >> a;
        fflush(stdin);
        */
    }
#else
    tbb::concurrent_vector <cv::Mat> particles_color_model(N);
    //tbb::mutex countMutex;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N,
    N / TBB_PARTITIONS), [&](const tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i != r.end(); i++) {
            const cv::Rect particle_roi(m_particles[i].d->x - roi_width * 0.5,
                                        m_particles[i].d->y - roi_height * 0.5, roi_width, roi_height);
            if (particle_roi.x < 0 || particle_roi.y < 0 || particle_roi.width <= 0
                    || particle_roi.height <= 0) {
                continue;
            }

            if (particle_roi.x + particle_roi.width >= frame_hsv.cols
                    || particle_roi.y + particle_roi.height >= frame_hsv.rows) {
                continue;
            }

            const cv::Mat mask = create_ellipse_mask(particle_roi, 1);
            const cv::Mat particle_roi_img = frame_hsv(particle_roi);

            // THIS NEEDS HEAVY OPTIMIZATION, most of the time is wasted here, in the vector insertion;
            particles_color_model[i] = compute_color_model(particle_roi_img, mask);
            /*
            countMutex.lock();
            cout << "RANGE " << r.size() << endl << flush;
            countMutex.unlock();
            */
        }
    });
#endif

/*
#ifndef USE_INTEL_TBB
#else
    tbb::concurrent_vector <cv::Mat> particles_color_model(N);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N,
    N / TBB_PARTITIONS), [&](const tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i != r.end(); i++) {
            const cv::Rect particle_roi(m_particles[i].d->x - roi_width * 0.5,
                                        m_particles[i].d->y - roi_height * 0.5, roi_width, roi_height);
            if (particle_roi.x < 0 || particle_roi.y < 0 || particle_roi.width <= 0
                    || particle_roi.height <= 0) {
                continue;
            }

            if (particle_roi.x + particle_roi.width >= frame_hsv.cols
                    || particle_roi.y + particle_roi.height >= frame_hsv.rows) {
                continue;
            }

            const cv::Mat mask = create_ellipse_mask(particle_roi, 1);
            const cv::Mat particle_roi_img = frame_hsv(particle_roi);

            particles_color_model[i] = compute_color_model(particle_roi_img, mask);
        }
    });

#endif
  */
    //third, weight them using the model
#ifndef USE_INTEL_TBB
    for (size_t i = 0; i < N; i++) {
        if (!particles_color_model[i].empty()) {
            const double score = 1 - cv::compareHist(*color_model, particles_color_model[i],
                                 CV_COMP_BHATTACHARYYA);
            m_particles[i].log_w += log(score);
        } else {
            m_particles[i].log_w += log(0);
        }
        //cout << "SCORE " << exp(m_particles[i].log_w) << endl;
    }
#else
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, N / TBB_PARTITIONS),
        [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); i++) {
                if (!particles_color_model[i].empty()) {
                const double score = 1 - cv::compareHist(*color_model, particles_color_model[i],
                                     CV_COMP_BHATTACHARYYA);
                m_particles[i].log_w += log(score);
            } else {
                m_particles[i].log_w += log(0);
            }
        }
    });
#endif
}

void  CImageParticleFilter::prediction_and_update_pfStandardProposal(
    const mrpt::obs::CActionCollection*,
    const mrpt::obs::CSensoryFrame *observation,
    const bayes::CParticleFilter::TParticleFilterOptions&)
{
    //CDisplayWindow model_window1("model1");
    //CDisplayWindow model_window2("model2");
    //CDisplayWindow model_window3("model3");
    const CObservationImagePtr obs = observation->getObservationByClass<CObservationImage>();
    ASSERT_(obs);
    //ASSERT_(!obs->image.empty());

    const int64_t current_time = cv::getTickCount();
    const double dt = (current_time - last_time) / cv::getTickFrequency();
    last_time = current_time;

    update_particles_with_transition_model(dt);
    weight_particles_with_model(obs->image);

    // Resample is automatically performed by CParticleFilter when required.
}

void CImageParticleFilter::initializeParticles(const size_t M, const pair<float, float> x,
        const pair<float, float> y,
        const pair<float, float> z, const pair<float, float> v_x, const pair<float, float> v_y,
        const pair<float, float> v_z)
{
    clearParticles();

    m_particles.resize(M);

    for (CParticleList::iterator it = m_particles.begin(); it != m_particles.end(); it++) {
        it->d = new CImageParticleData();

        it->d->x  = randomGenerator.drawGaussian1D(x.first, x.second);
        it->d->y  = randomGenerator.drawGaussian1D(y.first, y.second);
        it->d->z  = randomGenerator.drawGaussian1D(z.first, z.second);

        it->d->vx = randomGenerator.drawGaussian1D(v_x.first, v_x.second);
        it->d->vy = randomGenerator.drawGaussian1D(v_y.first, v_y.second);
        it->d->vz = randomGenerator.drawGaussian1D(v_z.first, v_z.second);

        it->log_w   = 0;
    }
}

void CImageParticleFilter::getMean(float &x, float &y, float &z, float &vx, float &vy,
                                   float &vz)
{
    double sumW = 0;
#ifndef USE_INTEL_TBB
    for (CParticleList::iterator it = m_particles.begin(); it != m_particles.end(); it++) {
        sumW += exp(it->log_w);
    }
#else
    sumW = tbb::parallel_reduce(
        tbb::blocked_range<CParticleList::iterator>(m_particles.begin(), m_particles.end(),
        m_particles.size() / TBB_PARTITIONS), 0.f,
        [](const tbb::blocked_range<CParticleList::iterator> &r, double value) -> double {
            return std::accumulate(r.begin(), r.end(), value,
                [](double value, const CParticleData & p) -> double {
                    return exp(p.log_w) + value;
                }
            );
        },
        std::plus<double>()
    );
#endif

    ASSERT_(sumW > 0)

    x = 0;
    y = 0;
    vx = 0;
    vy = 0;

    for (CParticleList::iterator it = m_particles.begin(); it != m_particles.end(); it++) {
        const double w = exp(it->log_w) / sumW;
        x += float(w * it->d->x);
        y += float(w * it->d->y);
        z += float(w * it->d->z);

        vx += float(w * it->d->vx);
        vy += float(w * it->d->vy);
        vz += float(w * it->d->vz);
    }
}



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
#define USE_KINECT_2
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

    cv::VideoCapture capture(CV_CAP_OPENNI);
    capture.set(CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_SXGA_15HZ);

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


    //init color model

    // Init. simulation:
    // -------------------------


    bool init_model = true;

    while (!mrpt::system::os::kbhit()) {
        // make an observation

#ifdef USE_KINECT_2
        listener.waitForNewFrame(frames);
        libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
        libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
        libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];
        color_frame = cv::Mat(rgb->height, rgb->width, CV_8UC3, rgb->data);
        depth_frame = cv::Mat(depth->height, depth->width, CV_32FC1, depth->data);
#else //kinect1
        capture.grab();
        capture.retrieve(color_frame, CV_CAP_OPENNI_BGR_IMAGE);
#endif
        /*
        if (color_frame.empty()) {
            capture.set(CV_CAP_PROP_POS_FRAMES, 0);
            exit(1);
            continue;
        }
        */

        // Process with PF:
        CObservationImagePtr obsImage = CObservationImage::Create();
        obsImage->image = CImage(new IplImage(color_frame));
        cv::Mat gradient = sobel_operator(color_frame);
        cv::Mat gradient_depth = sobel_operator(cv::Mat(depth_frame)/4500.f);
        model_window.showImage(CImage(new IplImage(gradient)));
        //depth_window.showImage(CImage(new IplImage(gradient_depth)));
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
                                              radius), make_pair(0, 0), make_pair(0, 0), make_pair(0, 0), make_pair(0, 0));
                init_model = false;
                particles.last_time = cv::getTickCount();


                model_frame = cv::Mat(color_frame(model_roi).size(), color_frame.type());
                const cv::Mat ones = cv::Mat::ones(color_frame(model_roi).size(), color_frame(model_roi).type());
                bitwise_and(color_frame(model_roi), ones, model_frame, mask);

                //cv::Mat gradient = sobel_operator(color_frame(model_roi));
                //model_window.showImage(CImage(new IplImage(gradient)));

                model_image_window.showImage(CImage(new IplImage(color_frame(model_roi))));
            }
        } else {
            // memory freed by SF.
            CSensoryFrame SF;
            SF.insert(obsImage);
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
            particles.getMean(avrg_x, avrg_y, avrg_z, avrg_vx, avrg_vy, avrg_vz);
            cv::circle(color_frame, cv::Point(avrg_x, avrg_y), 20, cv::Scalar(255, 0, 0), 5, 1, 0);
            cv::line(color_frame, cv::Point(avrg_x, avrg_y), cv::Point(avrg_x + avrg_vx, avrg_y + avrg_vy),
                     cv::Scalar(0, 255, 0), 5, 1, 0);
        }

        CImage frame_particles = CImage(new IplImage(color_frame));
        image.showImage(frame_particles);
#ifdef USE_KINECT_2
        listener.release(frames);
#endif
    }
}

int main(int, char *argv[])
{
    NUM_PARTICLES = atof(argv[1]);
    TRANSITION_MODEL_STD_XY   = atof(argv[2]);
    TRANSITION_MODEL_STD_VXY  = atof(argv[3]);

    TestBayesianTracking();
    return 0;
    /*
    try {
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
