#pragma once

#include "project_config.h"

#include <limits>

#include <boost/math/distributions/normal.hpp>

IGNORE_WARNINGS_PUSH

//#include <mrpt/gui/CDisplayWindow.h>
#include <mrpt/random.h>
#include <mrpt/bayes/CParticleFilterData.h>
#include <mrpt/obs/CSensoryFrame.h>
#include <mrpt/obs/CObservationImage.h>
#include <mrpt/otherlibs/do_opencv_includes.h>

#include <mrpt/gui/CDisplayWindow.h>
#include <mrpt/math/CHistogram.h>
using namespace mrpt::gui;

IGNORE_WARNINGS_POP

#include "GeometryHelpers.h"
#include "ColorModel.h"
#include "EllipseFunctions.h"
#include "ImageRegistration.h"
#include "EllipseStash.h"

using namespace mrpt;
using namespace mrpt::math;
using namespace mrpt::bayes;
using namespace mrpt::obs;
using namespace mrpt::random;
using namespace std;

using normal_dist = boost::math::normal_distribution<float>;

extern double TRANSITION_MODEL_STD_XY;
extern double TRANSITION_MODEL_STD_VXY;
extern double NUM_PARTICLES;

// ---------------------------------------------------------------
//      Implementation of the system models as a Particle Filter
// ---------------------------------------------------------------

struct ParticleData
{
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
    bool valid;
};

template<typename DEPTH_TYPE>
class CImageParticleFilter :
    public mrpt::bayes::CParticleFilterData<ParticleData>,
    public mrpt::bayes::CParticleFilterDataImpl <CImageParticleFilter<DEPTH_TYPE>,
        mrpt::bayes::CParticleFilterData<ParticleData>::CParticleList>
{

public:
    CHistogram hist_chest_color_score;
    CHistogram hist_head_color_score;
    CHistogram hist_head_fitting_score;
    CHistogram hist_head_z_score;
    CHistogram hist_score;

    static double WEIGHT_INVALID;
    CImageParticleFilter(EllipseStash *ellipses, const ImageRegistration * const reg, const normal_dist * const depth_distribution);
    using ParticleType = typename decltype(m_particles)::value_type;

    void update_particles_with_transition_model(const double dt, const mrpt::obs::CSensoryFrame * const observation);

    void weight_particles_with_model(const mrpt::obs::CSensoryFrame * const observation);

    void prediction_and_update_pfStandardProposal(
        const mrpt::obs::CActionCollection*,
        const mrpt::obs::CSensoryFrame * const observation,
        const bayes::CParticleFilter::TParticleFilterOptions&);

    void split_particles();

    void init_particles(const size_t M,
                        const pair<float, float> &x,
                        const pair<float, float> &y,
                        const pair<float, float> &z,
                        const pair<float, float> &v_x,
                        const pair<float, float> &v_y,
                        const pair<float, float> &v_z
    );

    void set_object_missing();
    void set_object_found();
    bool get_object_found();


    void set_head_color_model(const cv::Mat &model);
    const cv::Mat &get_head_color_model() const;

    void set_torso_color_model(const cv::Mat &model);
    const cv::Mat &get_torso_color_model() const;

    void set_shape_model(const vector<Eigen::Vector2f> &normal_vectors);
    float get_mean(float &x, float &y, float &z, float &vx, float &vy, float &vz) const;
    void print_particle_state(void) const;

    float last_distance;
    int64_t last_time;

    int object_times_missing;

    

#ifdef DEBUG
    CDisplayWindow particle_window;
    CImage particle_image;
#endif

    float transition_model_std_xy;
    float missing_uncertaincy_multipler;
protected:
    int64_t last_seen;
    bool object_found;

    vector<reference_wrapper<typename decltype(m_particles)::value_type>> particles_valid_roi;
    vector<reference_wrapper<typename decltype(m_particles)::value_type>> particles_invalid_roi;

    cv::Mat head_color_model;
    cv::Mat torso_color_model;

    const vector<Eigen::Vector2f> *shape_model;
    EllipseStash *ellipses;
    const ImageRegistration *registration;
    const boost::math::normal_distribution<float> *depth_normal_distribution;
};


//templated methods should have its definition in the same compilation unit
//of its declaration... hence this dirty trick.

#include "CImageParticleFilter.cpp"
