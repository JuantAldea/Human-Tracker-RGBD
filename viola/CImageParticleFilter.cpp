
#include <limits>
#include "CImageParticleFilter.h"

#include "misc_helpers.h"
#include "geometry_helpers.h"
#include "color_model.h"

#define USE_INTEL_TBB
#ifdef USE_INTEL_TBB
#include <tbb/tbb.h>
#define TBB_PARTITIONS 8
#endif

void CImageParticleFilter::print_particle_state(void) const
{
    size_t N = m_particles.size();
    for (size_t i = 0; i < N; i++) {
        std::cout << i << ' ' << m_particles[i].d->x
            << ' ' << m_particles[i].d->y << ' ' << m_particles[i].d->z
            << ' ' << m_particles[i].d->vx<< ' ' << m_particles[i].d->vy
            << ' ' << m_particles[i].d->vz << std::endl;
    }
}

void CImageParticleFilter::update_color_model(cv::Mat *model, const int roi_width,
        const int roi_height)
{
    color_model = model;
    this->roi_width = roi_width;
    this->roi_height = roi_height;
}

void CImageParticleFilter::update_particles_with_transition_model(const double dt, const mrpt::obs::CSensoryFrame * const observation)
{
    const CObservationImagePtr obs_image = observation->getObservationByClass<CObservationImage>(0);
    const CObservationImagePtr obs_depth = observation->getObservationByClass<CObservationImage>(1);
    
    ASSERT_(obs_image);
    ASSERT_(obs_depth);

    const cv::Mat image_mat = cv::Mat(obs_image->image.getAs<IplImage>());
    const cv::Mat depth_mat = cv::Mat(obs_depth->image.getAs<IplImage>());

    auto update_particle = [&](int i) {
        m_particles[i].d->x += dt * m_particles[i].d->vx + TRANSITION_MODEL_STD_XY *
                               randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->y += dt * m_particles[i].d->vy + TRANSITION_MODEL_STD_XY *
                               randomGenerator.drawGaussian1D_normalized();

        const double old_z = m_particles[i].d->z;
        const int x = cvRound((m_particles[i].d->x * depth_mat.cols) / float(image_mat.cols));
        const int y = cvRound((m_particles[i].d->y * depth_mat.rows) / float(image_mat.rows));
        m_particles[i].d->z = depth_mat.at<unsigned short>(y, x);

        m_particles[i].d->vx += TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->vy += TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->vz = (m_particles[i].d->z - old_z) / dt;
    };

    size_t N = m_particles.size();
#ifndef USE_INTEL_TBB
    for (size_t i = 0; i < N; i++) {
        update_particle(i);
    }
#else
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, N / TBB_PARTITIONS),
        [&update_particle](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); i++) {
                update_particle(i);
            }
    });
#endif
}

void CImageParticleFilter::weight_particles_with_model(const mrpt::obs::CSensoryFrame * const observation)
{
    const CObservationImagePtr obs_image = observation->getObservationByClass<CObservationImage>(0);
    
    ASSERT_(obs_image);

    const cv::Mat image_mat = cv::Mat(obs_image->image.getAs<IplImage>());

    cv::Mat frame_hsv;
    cv::cvtColor(image_mat, frame_hsv, cv::COLOR_BGR2HSV);

    size_t N = m_particles.size();

#ifndef USE_INTEL_TBB
    vector <cv::Mat> particles_color_model(N);
    for (size_t i = 0; i < N; i++) {
        const cv::Rect particle_roi(m_particles[i].d->x - roi_width * 0.5,
                                    m_particles[i].d->y - roi_height * 0.5,
                                    roi_width, roi_height);

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
#else
    tbb::concurrent_vector <cv::Mat> particles_color_model(N);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, N / TBB_PARTITIONS), 
        [this, &frame_hsv, &particles_color_model](const tbb::blocked_range<size_t> &r) {
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
        }
    );
#endif

    //third, weight them
    auto weight_particle = [this, &particles_color_model] (size_t i){
        if (!particles_color_model[i].empty()) {
            const double score = 1 - cv::compareHist(*color_model, particles_color_model[i],
                                 CV_COMP_BHATTACHARYYA);
            m_particles[i].log_w += log(score);
        } else {
            m_particles[i].log_w += log(std::numeric_limits<double>::min());
        }
    };
#ifndef USE_INTEL_TBB
    for (size_t i = 0; i < N; i++) {
        weight_particle(i);
    }
#else
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, N / TBB_PARTITIONS),
        [this, &particles_color_model, &weight_particle](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); i++) {
                weight_particle(i);
            }
        }
    );
#endif
}

void CImageParticleFilter::prediction_and_update_pfStandardProposal(
    const mrpt::obs::CActionCollection*,
    const mrpt::obs::CSensoryFrame * const observation,
    const bayes::CParticleFilter::TParticleFilterOptions&)
{
    //TODO take a look at mrpt::utils::CTicTac
    const int64_t current_time = cv::getTickCount();

    const double dt = (current_time - last_time) / cv::getTickFrequency();
    last_time = current_time;

    update_particles_with_transition_model(dt, observation);
    weight_particles_with_model(observation);

    // Resample is automatically performed by CParticleFilter when required.
}

void CImageParticleFilter::initializeParticles(const size_t M, const pair<float, float> x,
        const pair<float, float> y, const pair<float, float> z, const pair<float, float> v_x,
        const pair<float, float> v_y, const pair<float, float> v_z,
        const mrpt::obs::CSensoryFrame * const observation)
{
    clearParticles();

    const CObservationImagePtr obs_depth = observation->getObservationByClass<CObservationImage>(1);
    ASSERT_(obs_depth);
    
    const cv::Mat depth_mat = cv::Mat(obs_depth->image.getAs<IplImage>());

    m_particles.resize(M);

    for (CParticleList::iterator it = m_particles.begin(); it != m_particles.end(); it++) {
        it->d = new CImageParticleData();

        it->d->x  = randomGenerator.drawGaussian1D(x.first, x.second);
        it->d->y  = randomGenerator.drawGaussian1D(y.first, y.second);

        it->d->vx = randomGenerator.drawGaussian1D(v_x.first, v_x.second);
        it->d->vy = randomGenerator.drawGaussian1D(v_y.first, v_y.second);

        if (observation != nullptr){
            it->d->z  = depth_mat.at<float>(cvRound(it->d->y), cvRound(it->d->x));
            it->d->vz = 0;
        } else{
            it->d->z  = randomGenerator.drawGaussian1D(z.first, z.second);
            it->d->vz = randomGenerator.drawGaussian1D(v_z.first, v_z.second);;
        }

        it->log_w = 0;
    }
}

void CImageParticleFilter::get_mean(float &x, float &y, float &z, float &vx, float &vy,
                                   float &vz) const
{
    double sumW = 0;
#ifndef USE_INTEL_TBB
    for (CParticleList::iterator it = m_particles.begin(); it != m_particles.end(); it++) {
        sumW += exp(it->log_w);
    }
#else
    sumW = tbb::parallel_reduce(
        tbb::blocked_range<CParticleList::const_iterator>(m_particles.begin(), m_particles.end(),
            m_particles.size() / TBB_PARTITIONS), 0.f,
                [](const tbb::blocked_range<CParticleList::const_iterator> &r, double value) -> double {
                    return std::accumulate(r.begin(), r.end(), value,
                        [](double value, const CParticleData &p) -> double {
                            return exp(p.log_w) + value;
                        });
                },
            std::plus<double>()
        );
#endif

    ASSERT_(sumW > 0)

    x = 0;
    y = 0;
    z = 0;
    vx = 0;
    vy = 0;
    vz = 0;

    for (CParticleList::const_iterator it = m_particles.begin(); it != m_particles.end(); it++) {
        const double w = exp(it->log_w) / sumW;
        x += float(w * it->d->x);
        y += float(w * it->d->y);
        z += float(w * it->d->z);

        vx += float(w * it->d->vx);
        vy += float(w * it->d->vy);
        vz += float(w * it->d->vz);
    }
}
