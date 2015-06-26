#pragma once

#include "project_config.h"

#include <limits>

IGNORE_WARNINGS_PUSH

//#include <mrpt/gui/CDisplayWindow.h>
#include <mrpt/random.h>
#include <mrpt/bayes/CParticleFilterData.h>
#include <mrpt/obs/CSensoryFrame.h>
#include <mrpt/obs/CObservationImage.h>
#include <mrpt/otherlibs/do_opencv_includes.h>

#include <mrpt/gui/CDisplayWindow.h>
using namespace mrpt::gui;

IGNORE_WARNINGS_POP

#include "geometry_helpers.h"
#include "color_model.h"
#include "ellipse_functions.h"
#include "ImageRegistration.h"

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
struct CImageParticleData
{
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
    int object_x_length_pixels;
    int object_y_length_pixels;
};

template<typename DEPTH_TYPE>
class CImageParticleFilter :
    public mrpt::bayes::CParticleFilterData<CImageParticleData>,
    public mrpt::bayes::CParticleFilterDataImpl <CImageParticleFilter<DEPTH_TYPE>,
        mrpt::bayes::CParticleFilterData<CImageParticleData>::CParticleList>
{

public:
    
    using ParticleType = typename decltype(m_particles)::value_type;

    void update_particles_with_transition_model(const double dt, const mrpt::obs::CSensoryFrame * const observation);

    void weight_particles_with_model(const mrpt::obs::CSensoryFrame * const observation);

    void prediction_and_update_pfStandardProposal(
        const mrpt::obs::CActionCollection*,
        const mrpt::obs::CSensoryFrame * const observation,
        const bayes::CParticleFilter::TParticleFilterOptions&);
    void split_particles();
    void sort_particles();
    void initializeParticles(const size_t M,
                             const pair<float, float> &x,
                             const pair<float, float> &y,
                             const pair<float, float> &z,
                             const pair<float, float> &v_x,
                             const pair<float, float> &v_y,
                             const pair<float, float> &v_z,
                             const pair<float, float> &object_semiaxes_lengths,
                             const ImageRegistration &registration_data);


    void update_color_model(const cv::Mat &model);
    float get_mean(float &x, float &y, float &z, float &vx, float &vy, float &vz) const;
    void print_particle_state(void) const;

    int64_t last_time;
    cv::Mat color_model;

#ifdef DEBUG
    CDisplayWindow particle_window;
    CImage particle_image;
#endif


private:
    vector<reference_wrapper<typename decltype(m_particles)::value_type>> particles_valid_roi;
    vector<reference_wrapper<typename decltype(m_particles)::value_type>> particles_invalid_roi;

    ImageRegistration registration_data;
    float object_x_length;
    float object_y_length;
};


#include "CImageParticleFilter.cpp"
