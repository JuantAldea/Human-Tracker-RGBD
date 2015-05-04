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
//#include <opencv2/gpu/gpu.hpp>
//#include <limits>
#include <tbb/tbb.h>
using namespace mrpt;
using namespace mrpt::bayes;
using namespace mrpt::gui;
using namespace mrpt::math;
using namespace mrpt::obs;
using namespace mrpt::utils;
using namespace mrpt::random;
using namespace std;

#define DELTA_TIME                  0.1f

double TRANSITION_MODEL_STD_XY   = 50;
double TRANSITION_MODEL_STD_VXY  = 50;
double NUM_PARTICLES             = 5000;

#define USE_INTEL_TBB
#ifdef USE_INTEL_TBB
#define TBB_PARTITIONS 8
#endif

cv::Mat histogram_to_image(const cv::Mat &histogram, const int scale);
cv::Mat compute_color_model(const cv::Mat &hsv, const cv::Mat &mask);
cv::Mat create_ellipse_mask(const cv::Point &center, const int radi_x, const int radi_y, const int ndims);
cv::Mat create_ellipse_mask(const cv::Rect &rectangle, const int ndims);
inline bool point_within_ellipse(const cv::Point &point, const cv::Point &center, const int radi_x, const int radi_y);
vector<cv::Vec3f> detect_circles(const cv::Mat &image);
cv::Mat sobel_operator(const cv::Mat &img);
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
    void update_particles_with_transition_model();
    void weight_particles_with_model(const CImage &observation);

    void prediction_and_update_pfStandardProposal(
        const mrpt::obs::CActionCollection*,
        const mrpt::obs::CSensoryFrame *observation,
        const bayes::CParticleFilter::TParticleFilterOptions&);

    void initializeParticles(const size_t M, const pair<float, float> x, const pair<float, float> y, const pair<float, float> z,
                             const pair<float, float> v_x, const pair<float, float> v_y, const pair<float, float> v_z);

    void getMean(float &x, float &y, float &z, float &vx, float &vy, float &vz);

    void update_color_model(cv::Mat *model, const int roi_width, const int roi_height);

private:
    //TODO POTENTIAL LEAK! USE smartptr
    cv::Mat *color_model;
    int roi_width;
    int roi_height;
};


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
    randomGenerator.randomize();
    CDisplayWindow image("image");
    CDisplayWindow model_window("model");

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
    float  t = 0;

    cv::VideoCapture capture(CV_CAP_OPENNI);
    capture.set(CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_SXGA_15HZ);
    cv::Mat color_frame;
    cv::Mat frame_color_hsv;

    if (!capture.isOpened()) {
        return;
    }

    bool init_model = true;
    while (!mrpt::system::os::kbhit()) {
        // make an observation
        capture.grab();
        capture.retrieve(color_frame, CV_CAP_OPENNI_BGR_IMAGE);
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

        if (init_model) {
            cv::Mat frame_hsv;
            auto circles = detect_circles(color_frame);
            if (circles.size()){
                for (size_t i = 0; i < circles.size(); i++) {
                    cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
                    int radius = cvRound(circles[i][2]);
                    cv::circle(color_frame, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
                    cv::circle(color_frame, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
                }
                cv::Point center(cvRound(circles[0][0]), cvRound(circles[0][1]));
                int radius = cvRound(circles[0][2]);
                cout << "circle " << center.x << ' ' << center.y << ' ' << radius << endl;
                cv::cvtColor(color_frame, frame_hsv, cv::COLOR_BGR2HSV);
                const cv::Rect particle_roi(center.x - radius, center.y - radius, 2 * radius, 2 * radius);
                const cv::Mat mask = create_ellipse_mask(particle_roi, 1);
                model_window.showImage(CImage(new IplImage(mask)));
                
                const cv::Mat model = compute_color_model(frame_hsv(particle_roi), mask);
                particles.update_color_model(new cv::Mat(model), radius, radius);
                particles.initializeParticles(NUM_PARTICLES, make_pair(center.x, radius), make_pair(center.y, radius), make_pair(0, 0), make_pair(0, 0), make_pair(0, 0), make_pair(0, 0));
                init_model = false;
            }
        }else{
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
                cv::circle(color_frame, cv::Point(particles.m_particles[i].d->x, particles.m_particles[i].d->y), 1, cv::Scalar(0, 0, 255), 1, 1, 0);
            }

            float avrg_x, avrg_y, avrg_z, avrg_vx, avrg_vy, avrg_vz;
            particles.getMean(avrg_x, avrg_y, avrg_z, avrg_vx, avrg_vy, avrg_vz);
            cv::circle(color_frame, cv::Point(avrg_x, avrg_y), 20, cv::Scalar(255, 0, 0), 5, 1, 0);
            cv::line(color_frame, cv::Point(avrg_x, avrg_y), cv::Point(avrg_x + avrg_vx, avrg_y + avrg_vy), cv::Scalar(0, 255, 0), 5, 1, 0);
            t += DELTA_TIME;
        }

        CImage frame_particles = CImage(new IplImage(color_frame));
        image.showImage(frame_particles);
    }
}

void CImageParticleFilter::update_color_model(cv::Mat *model, const int roi_width, const int roi_height)
{
    color_model = model;
    this->roi_width = roi_width;
    this->roi_height = roi_height;
}

void CImageParticleFilter::update_particles_with_transition_model()
{
    size_t N = m_particles.size();

#ifndef USE_INTEL_TBB
    for (size_t i = 0; i < N; i++) {
        m_particles[i].d->x += DELTA_TIME * m_particles[i].d->vx + TRANSITION_MODEL_STD_XY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->y += DELTA_TIME * m_particles[i].d->vy + TRANSITION_MODEL_STD_XY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->z += DELTA_TIME * m_particles[i].d->vz + TRANSITION_MODEL_STD_XY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->vx += TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->vy += TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->vz += TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
    }
#else
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, N / TBB_PARTITIONS), [this](const tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i != r.end(); i++) {
            m_particles[i].d->x += DELTA_TIME * m_particles[i].d->vx + TRANSITION_MODEL_STD_XY * randomGenerator.drawGaussian1D_normalized();
            m_particles[i].d->y += DELTA_TIME * m_particles[i].d->vy + TRANSITION_MODEL_STD_XY * randomGenerator.drawGaussian1D_normalized();
            m_particles[i].d->z += DELTA_TIME * m_particles[i].d->vz + TRANSITION_MODEL_STD_XY * randomGenerator.drawGaussian1D_normalized();
            m_particles[i].d->vx += TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
            m_particles[i].d->vy += TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
            m_particles[i].d->vz += TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
        }
    });
#endif
}


cv::Mat sobel_operator(const cv::Mat &img)
{
    using namespace cv;
    Mat grad;
    Mat src = img.clone();

    GaussianBlur(src, src, Size(3,3), 0, 0, BORDER_DEFAULT);

    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    //Scharr(src, grad_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
    Sobel(src, grad_x, CV_32F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);

    /// Gradient Y
    //Scharr(src, grad_y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
    Sobel(src, grad_y, CV_32F, 0, 1, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);

    /// Total Gradient (approximate)
    magnitude(abs_grad_x, abs_grad_y, grad);

    return grad;
}


void CImageParticleFilter::weight_particles_with_model(const CImage &observation)
{
    const cv::Mat image_mat = cv::Mat(observation.getAs<IplImage>());

    cv::Mat frame_hsv;
    cv::Mat frame_grey;

    cv::cvtColor(image_mat, frame_hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(image_mat, frame_grey, cv::COLOR_BGR2GRAY);


    size_t N = m_particles.size();

#ifndef USE_INTEL_TBB

#else

#endif


#ifndef USE_INTEL_TBB
    vector <cv::Mat> particles_color_model(N);
    for (size_t i = 0; i < N; i++) {
        const cv::Rect particle_roi(m_particles[i].d->x - roi_width * 0.5, m_particles[i].d->y - roi_height * 0.5, roi_width, roi_height);

        if (particle_roi.x < 0 || particle_roi.y < 0 || particle_roi.width <= 0 || particle_roi.height <= 0) {
            continue;
        }

        if (particle_roi.x + particle_roi.width >= frame_hsv.cols || particle_roi.y + particle_roi.height >= frame_hsv.rows) {
            continue;
        }

        const cv::Mat mask = create_ellipse_mask(particle_roi, 1);
        const cv::Mat particle_roi_img = frame_hsv(particle_roi);

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
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, N / TBB_PARTITIONS), 
        [this, &frame_hsv, &particles_color_model](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); i++) {
                const cv::Rect particle_roi(m_particles[i].d->x - roi_width * 0.5, m_particles[i].d->y - roi_height * 0.5, roi_width, roi_height);
                if (particle_roi.x < 0 || particle_roi.y < 0 || particle_roi.width <= 0 || particle_roi.height <= 0) {
                    continue;
                }

                if (particle_roi.x + particle_roi.width >= frame_hsv.cols || particle_roi.y + particle_roi.height >= frame_hsv.rows) {
                    continue;
                }

                const cv::Mat mask = create_ellipse_mask(particle_roi, 1);
                const cv::Mat particle_roi_img = frame_hsv(particle_roi);

                particles_color_model[i] = compute_color_model(particle_roi_img, mask);
            }
    });
#endif

    //third, weight them using the model
#ifndef USE_INTEL_TBB
    for (size_t i = 0; i < N; i++) {
        if (!particles_color_model[i].empty()) {
            const double score = 1 - cv::compareHist(*color_model, particles_color_model[i], CV_COMP_BHATTACHARYYA);
            m_particles[i].log_w += log(score);
        } else {
            m_particles[i].log_w += log(0);
        }
        //cout << "SCORE " << exp(m_particles[i].log_w) << endl;
    }
#else
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, N / TBB_PARTITIONS), [this, &particles_color_model](const tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i != r.end(); i++) {
            if (!particles_color_model[i].empty()) {
                const double score = 1 - cv::compareHist(*color_model, particles_color_model[i], CV_COMP_BHATTACHARYYA);
                //cout << score << endl;
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

    update_particles_with_transition_model();
    weight_particles_with_model(obs->image);


    // Resample is automatically performed by CParticleFilter when required.
}

void CImageParticleFilter::initializeParticles(const size_t M, const pair<float, float> x, const pair<float, float> y,
        const pair<float, float> z, const pair<float, float> v_x, const pair<float, float> v_y, const pair<float, float> v_z)
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

void CImageParticleFilter::getMean(float &x, float &y, float &z, float &vx, float &vy, float &vz)
{
    double sumW = 0;
#ifndef USE_INTEL_TBB
    for (CParticleList::iterator it = m_particles.begin(); it != m_particles.end(); it++) {
        sumW += exp(it->log_w);
    }
#else
    sumW = tbb::parallel_reduce(
        tbb::blocked_range<CParticleList::iterator>(m_particles.begin(), m_particles.end(), m_particles.size() / TBB_PARTITIONS), 0.f,
        [](const tbb::blocked_range<CParticleList::iterator> &r, double value) -> double {
            return std::accumulate(r.begin(), r.end(), value,
                [](double value, const CParticleData &p) -> double {
                    return exp(p.log_w) + value;
            });
        },
        std::plus<double>());
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

cv::Mat compute_color_model(const cv::Mat &hsv, const cv::Mat &mask)
{
    // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    cv::Mat histogram;
    const int hbins = 31;
    const int sbins = 32;

    {
        // hue varies from 0 to 179, it's scaled down by a half
        const int histSize[] = {hbins, sbins};
        // so that it fits in a byte.
        const float hranges[] = {0, 180};
        // saturation varies from 0 (black-gray-white) to
        // 255 (pure spectrum color)
        const float sranges[] = {0, 256};
        const float* ranges[] = {hranges, sranges};
        const int channels[] = {0, 1};
        cv::calcHist(&hsv, 1, channels, mask, histogram, 2, histSize, ranges, true, false);
    }

    cv::Mat histogram_v;

    {
        const int channels[] = {2};
        const int histSize[] = {sbins};
        const float range[] = {0, 256} ;
        const float* histRange = {range};
        cv::calcHist(&hsv, 1, channels, mask, histogram_v, 1, histSize, &histRange, true, false);
    }

    histogram_v = histogram_v.t();
    histogram.push_back(histogram_v);

    double sum = 0;
    for (int h = 0; h < histogram.rows; h++) {
        for (int s = 0; s < histogram.cols; s++) {
            sum += histogram.at<float>(h, s);
        }
    }

    for (int h = 0; h < histogram.rows; h++) {
        for (int s = 0; s < histogram.cols; s++) {
            histogram.at<float>(h, s) /= sum;
        }
    }

    return histogram;
}

inline bool point_within_ellipse(const cv::Point &point, const cv::Point &center, const int radi_x, const int radi_y)
{
    return (((point.x - center.x) * (point.x - center.x)) / float((radi_x * radi_x)) + ((point.y - center.y) * (point.y - center.y)) / float((radi_y * radi_y))) <= 1;
}

cv::Mat create_ellipse_mask(const cv::Rect &rectangle, const int ndims)
{
    return create_ellipse_mask(cv::Point(rectangle.width / 2.f, rectangle.height / 2.f), cvRound(rectangle.width), cvRound(rectangle.height), ndims);
}

cv::Mat create_ellipse_mask(const cv::Point &center, const int radi_x, const int radi_y, const int ndims)
{
    cv::Mat mask;
    mask.create(radi_y, radi_x, CV_8UC1);
    int channels = mask.channels();
    int nRows = mask.rows;
    int nCols = mask.cols * channels;
    for (int i = 0; i < nRows; i++) {
        uchar* mask_row = mask.ptr<uchar>(i);
        for (int j = 0; j < nCols; j++) {
            mask_row[j] = point_within_ellipse(cv::Point(j, i), center, radi_x, radi_y) ? 0xff : 0x0;
        }
    }

    vector<cv::Mat> mask_channels(ndims);
    for (int i = 0; i < ndims; i++) {
        mask_channels[i] = mask;
    }

    cv::Mat mask_ndims;
    cv::merge(mask_channels, mask_ndims);
    return mask_ndims;
}

cv::Mat histogram_to_image(const cv::Mat &histogram, const int scale)
{
    cv::Mat histImg = cv::Mat::zeros(histogram.rows * scale, histogram.cols * scale, CV_8UC1);
    double maxVal = 0;
    cv::minMaxLoc(histogram, 0, &maxVal, 0, 0);
    for (int row = 0; row < histogram.rows; row++) {
        for (int col = 0; col < histogram.cols; col++) {
            const float binVal = histogram.at<float>(row, col);
            const int intensity = cvRound(255 * (binVal / maxVal));
            cv::rectangle(histImg, cv::Point(row * scale, col * scale),
                          cv::Point((row + 1) * scale - 1, (col + 1) * scale - 1),
                          cv::Scalar::all(intensity), CV_FILLED);
        }
    }
    return histImg;
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

