/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2015, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */
// ------------------------------------------------------
//  Refer to the description in the wiki:
//  http://www.mrpt.org/Kalman_Filters
// ------------------------------------------------------


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
#include "CObservationImageWithModel.h"

using namespace mrpt;
using namespace mrpt::bayes;
using namespace mrpt::gui;
using namespace mrpt::math;
using namespace mrpt::obs;
using namespace mrpt::utils;
using namespace mrpt::random;
using namespace std;

#define BEARING_SENSOR_NOISE_STD    DEG2RAD(15.0f)
#define RANGE_SENSOR_NOISE_STD      0.3f
#define DELTA_TIME                  0.1f

#define VEHICLE_INITIAL_X           4.0f
#define VEHICLE_INITIAL_Y           4.0f
#define VEHICLE_INITIAL_V           1.0f
#define VEHICLE_INITIAL_W           DEG2RAD(20.0f)

#define TRANSITION_MODEL_STD_XY     0.03f
#define TRANSITION_MODEL_STD_VXY    0.20f

#define NUM_PARTICLES               2000

// ---------------------------------------------------------------
//      Implementation of the system models as a Particle Filter
// ---------------------------------------------------------------
struct CImageParticleData {
    float x, y, vx, vy; // Vehicle state (position & velocities)
};

class CImageParticleFilter :
    public mrpt::bayes::CParticleFilterData<CImageParticleData>,
    public mrpt::bayes::CParticleFilterDataImpl<CImageParticleFilter, mrpt::bayes::CParticleFilterData<CImageParticleData>::CParticleList>
{
public:
    void  prediction_and_update_pfStandardProposal(
        const mrpt::obs::CActionCollection*,
        const mrpt::obs::CSensoryFrame      *observation,
        const bayes::CParticleFilter::TParticleFilterOptions&);

    void initializeParticles(const size_t M, const pair<float, float> x, const pair<float, float> y, const pair<float, float> v_x, const pair<float, float> v_y);

    void getMean(float &x, float &y, float &vx, float &vy);
};

void TestBayesianTracking()
{
    randomGenerator.randomize();

    CDisplayWindowPlots winPF("Tracking - Particle Filter", 450, 400);
    CDisplayWindow image("image");
    winPF.setPos(480, 10);

    winPF.axis(-2, 20, -10, 10);
    winPF.axis_equal();

    // Create PF
    // ----------------------
    CParticleFilter::TParticleFilterOptions PF_options;
    PF_options.adaptiveSampleSize = false;
    PF_options.PF_algorithm = CParticleFilter::pfStandardProposal;
    PF_options.resamplingMethod = CParticleFilter::prSystematic;

    CParticleFilter PF;
    PF.m_options = PF_options;

    CImageParticleFilter  particles;
    particles.initializeParticles(NUM_PARTICLES, make_pair(0, 10), make_pair(0, 10), make_pair(0, 10), make_pair(0, 10));
    

    // Init. simulation:
    // -------------------------
    float  t = 0;

    cv::VideoCapture capture("output.mpg");

    cv::Mat color_frame;
    cv::Mat frame_color_hsv;

    if (!capture.isOpened()) {
        return;
    }

    
    while (winPF.isOpen() && !mrpt::system::os::kbhit()) {
        // make an observation
        
        //frame.setFromIplImage(new IplImage(cv::Mat(cv::Mat::zeros(100, 100, CV_8UC1))));
        capture.grab();
        capture >> color_frame;
        
        // Process with PF:
        CObservationImageWithModelPtr obsImage =CObservationImageWithModel::Create();
        obsImage->image = CImage(new IplImage(color_frame));
        obsImage->model = cv::Mat::zeros(100, 100, CV_8UC1);
        image.showImage(obsImage->image);
        // memory freed by SF.
        CSensoryFrame SF;
        SF.insert(obsImage);
        // Process in the PF
        PF.executeOn(particles, NULL, &SF);

        // Show PF state:
        cout << "Particle filter ESS: " << particles.ESS() << endl;

        // Draw PF state:
        {
            size_t i, N = particles.m_particles.size();
            vector<float>   parts_x(N), parts_y(N);
            for (i = 0; i < N; i++) {
                parts_x[i] = particles.m_particles[i].d->x;
                parts_y[i] = particles.m_particles[i].d->y;
            }

            winPF.plot(parts_x, parts_y, "b.2", "particles");

            // Draw PF velocities:
            float avrg_x, avrg_y, avrg_vx, avrg_vy;

            particles.getMean(avrg_x, avrg_y, avrg_vx, avrg_vy);

            vector<float> vx(2), vy(2);
            vx[0] = avrg_x;
            vx[1] = vx[0] + avrg_vx * 1;
            vy[0] = avrg_y;
            vy[1] = vy[0] + avrg_vy * 1;
            winPF.plot(vx, vy, "g-4", "velocityPF");
        }

        /*
        // Draw GT:
        winPF.plot(vector<float>(1, x), vector<float>(1, y), "k.8", "plot_GT");

        // Draw noisy observations:
        vector<float>  obs_x(2), obs_y(2);
        winPF.plot(obs_x, obs_y, "r", "plot_obs_ray");
        */
        // Delay:
        mrpt::system::sleep((int)(DELTA_TIME * 1000));
        t += DELTA_TIME;
    }
}

void  CImageParticleFilter::prediction_and_update_pfStandardProposal(
    const mrpt::obs::CActionCollection*,
    const mrpt::obs::CSensoryFrame *observation,
    const bayes::CParticleFilter::TParticleFilterOptions&)
{
    size_t i, N = m_particles.size();

    // Transition model:
    for (i = 0; i < N; i++) {
        m_particles[i].d->x += DELTA_TIME * m_particles[i].d->vx + TRANSITION_MODEL_STD_XY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->y += DELTA_TIME * m_particles[i].d->vy + TRANSITION_MODEL_STD_XY * randomGenerator.drawGaussian1D_normalized();

        m_particles[i].d->vx += TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->vy += TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
    }

    CObservationImagePtr obs = observation->getObservationByClass<CObservationImage>();
    ASSERT_(obs);
    //ASSERT_(!obs->image.empty());
    
    CImage image = obs->image;

    // Update weights
    for (i = 0; i < N; i++) {
        m_particles[i].log_w += 0;
            //log(math::normalPDF(predicted_range - obsRange, 0, RANGE_SENSOR_NOISE_STD)) +
            //log(math::normalPDF(math::wrapToPi(predicted_bearing - obsBearing), 0, BEARING_SENSOR_NOISE_STD));
    }

    // Resample is automatically performed by CParticleFilter when required.
}

void CImageParticleFilter::initializeParticles(const size_t M, const pair<float, float> x, const pair<float, float> y, const pair<float, float> v_x, const pair<float, float> v_y)
{
    clearParticles();
    
    m_particles.resize(M);

    for (CParticleList::iterator it = m_particles.begin(); it != m_particles.end(); it++) {
        it->d = new CImageParticleData();
        
        it->d->x  = randomGenerator.drawGaussian1D(x.first, x.second);
        it->d->y  = randomGenerator.drawGaussian1D(y.first, y.second);
        
        it->d->vx = randomGenerator.drawGaussian1D(v_x.first, v_x.second);
        it->d->vy = randomGenerator.drawGaussian1D(v_y.first, v_y.second);

        it->log_w   = 0;
    }
}

void CImageParticleFilter::getMean(float &x, float &y, float &vx, float &vy)
{
    double sumW = 0;
    for (CParticleList::iterator it = m_particles.begin(); it != m_particles.end(); it++) {
        sumW += exp(it->log_w);
    }

    ASSERT_(sumW > 0)

    x = 0;
    y = 0;
    vx = 0;
    vy = 0;

    for (CParticleList::iterator it = m_particles.begin(); it != m_particles.end(); it++) {
        const double w = exp(it->log_w) / sumW;
        x += float(w * it->d->x);
        y += float(w * it->d->y);
        vx += float(w * it->d->vx);
        vy += float(w * it->d->vy);
    }
}

int main()
{
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
}

