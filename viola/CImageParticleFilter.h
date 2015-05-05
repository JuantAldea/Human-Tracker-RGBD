#pragma once

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

//#include <mrpt/gui/CDisplayWindow.h>
#include <mrpt/random.h>
#include <mrpt/bayes/CParticleFilterData.h>
#include <mrpt/obs/CSensoryFrame.h>
#include <mrpt/obs/CObservationImage.h>
#include <mrpt/otherlibs/do_opencv_includes.h>

#pragma GCC diagnostic pop

using namespace mrpt;
using namespace mrpt::bayes;
using namespace mrpt::obs;
using namespace mrpt::random;
using namespace std;

extern double TRANSITION_MODEL_STD_XY;
extern double TRANSITION_MODEL_STD_VXY;
extern double NUM_PARTICLES;


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
    void update_particles_with_transition_model(double dt, const mrpt::obs::CSensoryFrame * const observation);
    void weight_particles_with_model(const mrpt::obs::CSensoryFrame * const observation);

    void prediction_and_update_pfStandardProposal(
        const mrpt::obs::CActionCollection*,
        const mrpt::obs::CSensoryFrame * const observation,
        const bayes::CParticleFilter::TParticleFilterOptions&);

    void initializeParticles(const size_t M,
                             const pair<float, float> x,
                             const pair<float, float> y,
                             const pair<float, float> z,
                             const pair<float, float> v_x,
                             const pair<float, float> v_y,
                             const pair<float, float> v_z,
                             const mrpt::obs::CSensoryFrame * const observation);


    void update_color_model(cv::Mat *model, const int roi_width, const int roi_height);
    
    void get_mean(float &x, float &y, float &z, float &vx, float &vy, float &vz) const;
    void print_particle_state(void) const;

    int64_t last_time;
private:
    //TODO POTENTIAL LEAK! USE smartptr
    cv::Mat *color_model;
    int roi_width;
    int roi_height;

};
