#pragma once
#include <cmath>
#include "model_parameters.h"

/*
#include <limits>

#include "CImageParticleFilter.h"
#include "geometry_helpers.h"
#include "color_model.h"
*/

template<typename DEPTH_TYPE>
void CImageParticleFilter<DEPTH_TYPE>::print_particle_state(void) const
{
    size_t N = m_particles.size();
    for (size_t i = 0; i < N; i++) {
        std::cout << i << ' '            
            << std::exp(m_particles[i].log_w) << ' '
            << m_particles[i].d->x  << ' '
            << m_particles[i].d->y  << ' '
            << m_particles[i].d->z  << ' '
            << m_particles[i].d->vx << ' '
            << m_particles[i].d->vy << ' '
            << m_particles[i].d->vz << ' '
            << m_particles[i].d->object_x_length_pixels << ' '
            << m_particles[i].d->object_y_length_pixels << ' '
            << std::endl;
    }
}

template<typename DEPTH_TYPE>
void CImageParticleFilter<DEPTH_TYPE>::update_color_model(const cv::Mat &model)
{
    color_model = model.clone();
}

template<typename DEPTH_TYPE>
void CImageParticleFilter<DEPTH_TYPE>::update_particles_with_transition_model(const double dt, const mrpt::obs::CSensoryFrame * const observation)
{

    const CObservationImagePtr image_color = observation->getObservationBySensorLabelAs<CObservationImagePtr>("color");
    const CObservationImagePtr image_depth = observation->getObservationBySensorLabelAs<CObservationImagePtr>("depth");
    
    ASSERT_(image_color);
    ASSERT_(image_depth);

    const cv::Mat image_mat = cv::Mat(image_color->image.getAs<IplImage>());
    const cv::Mat depth_mat = cv::Mat(image_depth->image.getAs<IplImage>());
    
    auto update_particle = [&](const size_t i) {
        const double old_z = m_particles[i].d->z;
        const float old_x = m_particles[i].d->x;
        const float old_y = m_particles[i].d->y;
        
        
        m_particles[i].d->x += dt * m_particles[i].d->vx + TRANSITION_MODEL_STD_XY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->y += dt * m_particles[i].d->vy + TRANSITION_MODEL_STD_XY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->z = 0;
        
        if (m_particles[i].d->x >= 0 && m_particles[i].d->x < depth_mat.cols && m_particles[i].d->y >= 0 && m_particles[i].d->y < depth_mat.rows){
            m_particles[i].d->z = depth_mat.at<DEPTH_TYPE>(cvRound(m_particles[i].d->y), cvRound(m_particles[i].d->x));
        }
        const double inv_dt = 1.0 / dt;
        m_particles[i].d->vx = (m_particles[i].d->x - old_x) * inv_dt + TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->vy = (m_particles[i].d->y - old_y) * inv_dt + TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->vz = (m_particles[i].d->z - old_z) * inv_dt + TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();

        /*
        m_particles[i].d->vx = TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->vy = TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->vz = TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
        */
        
        m_particles[i].d->object_x_length_pixels = 0;
        m_particles[i].d->object_y_length_pixels = 0;
        
        if(m_particles[i].d->z != 0){
            Eigen::Vector2i top_corner, bottom_corner;
            std::tie(top_corner, bottom_corner) = project_model(Eigen::Vector2f(m_particles[i].d->x, m_particles[i].d->y), m_particles[i].d->z,
                Eigen::Vector2f(this->object_x_length * 0.5, this->object_y_length * 0.5),
                registration_data.cameraMatrixColor, registration_data.lookupX, registration_data.lookupY);
            
            if (top_corner[0] >= 0 && top_corner[1] > 0 && bottom_corner[0] < depth_mat.cols && bottom_corner[1] < depth_mat.rows){
                const Eigen::Vector2i pixel_lengths = bottom_corner - top_corner;
                m_particles[i].d->object_x_length_pixels = pixel_lengths[0];
                m_particles[i].d->object_y_length_pixels = pixel_lengths[1];
            }
        }
    };

    size_t N = m_particles.size();
#ifdef USE_INTEL_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, N / TBB_PARTITIONS),
        [&update_particle](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); i++) {
                update_particle(i);
            }
        }
    );
#else
    for (size_t i = 0; i < N; i++) {
        update_particle(i);
    }
#endif
}

template<typename DEPTH_TYPE>
void CImageParticleFilter<DEPTH_TYPE>::sort_particles()
{
    auto m_particles_filtered = m_particles;
    std::sort(m_particles_filtered.begin(), m_particles_filtered.end(), 
        [this](decltype(m_particles_filtered)::value_type &a, decltype(m_particles_filtered)::value_type &b)
            { 
                return a.d->object_x_length_pixels * a.d->object_y_length_pixels > b.d->object_x_length_pixels * b.d->object_y_length_pixels;
            }
    );
}

template<typename DEPTH_TYPE>
void CImageParticleFilter<DEPTH_TYPE>::split_particles()
{
    particles_valid_roi.clear();
    particles_invalid_roi.clear();
    for (typename decltype(m_particles)::iterator it = m_particles.begin(); it != m_particles.end(); it++){
        if (it->d->object_x_length_pixels > 0) {
            particles_valid_roi.push_back(*it);
        } else {
            particles_invalid_roi.push_back(*it);
        }
    }
}


template<typename DEPTH_TYPE>
void CImageParticleFilter<DEPTH_TYPE>::weight_particles_with_model(const mrpt::obs::CSensoryFrame * const observation)
{
    const CObservationImagePtr image_hsv = observation->getObservationBySensorLabelAs<CObservationImagePtr>("hsv");
    const CObservationImagePtr image_gradient_vectors = observation->getObservationBySensorLabelAs<CObservationImagePtr>("gradient_vectors");
    const CObservationImagePtr image_gradient_magnitude = observation->getObservationBySensorLabelAs<CObservationImagePtr>("gradient_magnitude");

    ASSERT_(image_hsv);
    ASSERT_(image_gradient_vectors);
    ASSERT_(image_gradient_magnitude);

    const cv::Mat frame_hsv = cv::Mat(image_hsv->image.getAs<IplImage>());
    const cv::Mat gradient_vectors = cv::Mat(image_gradient_vectors->image.getAs<IplImage>());
    const cv::Mat gradient_magnitude = cv::Mat(image_gradient_magnitude->image.getAs<IplImage>());

    split_particles();
    assert(particles_invalid_roi.size() + particles_valid_roi.size() == m_particles.size());

    size_t N = particles_valid_roi.size();

    vector<cv::Mat> particles_color_model(N);
    vector<float> particles_ellipse_fitting(N);

    auto compute_valid_particle_color_model = [&](size_t i){
        const ParticleType &particle = particles_valid_roi[i];
        const cv::Rect particle_roi(
            particle.d->x - particle.d->object_x_length_pixels * 0.5,
            particle.d->y - particle.d->object_y_length_pixels * 0.5,
            particle.d->object_x_length_pixels, particle.d->object_y_length_pixels
        );

        const cv::Mat mask = fast_create_ellipse_mask(particle_roi, 1);
        const cv::Mat particle_roi_img = frame_hsv(particle_roi);
        
        //TODO DEBUG CODE
        if (i == 0){
            particle_image.loadFromIplImage(new IplImage(particle_roi_img));
            particle_window.showImage(particle_image);
        }

        //particles_color_model[i] = compute_color_model2(particle_roi_img, mask);
        particles_color_model[i] = compute_color_model(particle_roi_img, mask);
    };

#ifdef USE_INTEL_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, N / TBB_PARTITIONS), 
        [this, &frame_hsv, &particles_color_model, &compute_valid_particle_color_model,
            &gradient_vectors, &gradient_magnitude, &particles_ellipse_fitting](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); i++) {                
                compute_valid_particle_color_model(i);
                
                const ParticleType &particle = particles_valid_roi[i];
                particles_ellipse_fitting[i] = ellipse_contour_test(
                    cv::Point(particle.d->x, particle.d->y),
                        particle.d->object_x_length_pixels * 0.5,
                        particle.d->object_y_length_pixels * 0.5,
                        ELLIPSE_FITTING_ANGLE_STEP, gradient_vectors, gradient_magnitude, nullptr
                );
                //TODO DEBUG CODE
                if (i == 0){
                    std::cout << "FITTING 0 " << particles_ellipse_fitting[i] << std::endl;
                }
            }
        }
    );
#else
    for (size_t i = 0; i < N; i++) {
        compute_valid_particle_color_model(i);
        const ParticleType &particle = particles_valid_roi[i];
        particles_ellipse_fitting[i] = ellipse_contour_test(
            cv::Point(particle.d->x, particle.d->y),
                particle.d->object_x_length_pixels * 0.5,
                particle.d->object_y_length_pixels * 0.5,
                ELLIPSE_FITTING_ANGLE_STEP, gradient_vectors, gradient_magnitude, nullptr
        );
    }
#endif

    double sum_gradient_fitting = 0;
#ifdef USE_INTEL_TBB
    sum_gradient_fitting = tbb::parallel_reduce(
        tbb::blocked_range<decltype(particles_ellipse_fitting)::const_iterator>(particles_ellipse_fitting.begin(), particles_ellipse_fitting.end(),
            particles_ellipse_fitting.size() / TBB_PARTITIONS), 0.f,
                [](const tbb::blocked_range<vector<float>::const_iterator> &r, double value) -> double {
                    return std::accumulate(r.begin(), r.end(), value,
                        [](double value, const float fitting) -> double {
                            return fitting + value;
                        }
                    );
                },
            std::plus<double>()
    );
#else
    for (size_t i = 0; i < N; i++) {
        sum_gradient_fitting += particles_ellipse_fitting[i];
    }
#endif

float max_fitting = particles_ellipse_fitting[0];
float min_fitting = particles_ellipse_fitting[0];
for (size_t i = 0; i < N; i++) {
    min_fitting = std::min(min_fitting, particles_ellipse_fitting[i]);
    max_fitting = std::max(max_fitting, particles_ellipse_fitting[i]);
}

float inv_range_fitting = 1.0f / (max_fitting - min_fitting);

#ifdef USE_INTEL_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, N / TBB_PARTITIONS),
        [this, &particles_ellipse_fitting, &sum_gradient_fitting, &min_fitting, &inv_range_fitting](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); i++) {
                particles_ellipse_fitting[i] = (particles_ellipse_fitting[i] - min_fitting) * inv_range_fitting;
            }
        }
    );
#else
    for (size_t i = 0; i < N; i++) {
        particles_ellipse_fitting[i] = (particles_ellipse_fitting[i] - min_fitting) * inv_range_fitting;
    }
#endif

/*
#ifdef USE_INTEL_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, N / TBB_PARTITIONS),
        [this, &particles_ellipse_fitting, sum_gradient_fitting](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); i++) {
                particles_ellipse_fitting[i] /= sum_gradient_fitting;
            }
        }
    );
#else
    for (size_t i = 0; i < N; i++) {
        particles_ellipse_fitting[i] /= sum_gradient_fitting;
    }
#endif
*/
    //third, weight them
    auto weight_valid_particle = [this, &particles_color_model, &particles_ellipse_fitting] (size_t i){
        const double distance_hist = cv::compareHist(color_model, particles_color_model[i], CV_COMP_BHATTACHARYYA);
        double score = 1;
        score *= (1 - distance_hist);
        score *= particles_ellipse_fitting[i];
        particles_valid_roi[i].get().log_w += log(score);
    };

#ifdef USE_INTEL_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, N / TBB_PARTITIONS),
        [this, &particles_color_model, &weight_valid_particle](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); i++) {
                weight_valid_particle(i);
            }
        }
    );
#else
    for (size_t i = 0; i < N; i++) {
        weight_valid_particle(i);
    }
#endif
    
    const size_t N_invalids = particles_invalid_roi.size();
    constexpr double w_invalid = log(std::numeric_limits<double>::min());
    for (size_t i = 0; i < N_invalids; i++) {
        particles_invalid_roi[i].get().log_w += w_invalid;
    }
}

template<typename DEPTH_TYPE>
void CImageParticleFilter<DEPTH_TYPE>::prediction_and_update_pfStandardProposal(
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
    //print_particle_state();
    // Resample is automatically performed by CParticleFilter when required.
}

template<typename DEPTH_TYPE>
void CImageParticleFilter<DEPTH_TYPE>::initializeParticles(const size_t M, const pair<float, float> &x,
        const pair<float, float> &y, const pair<float, float> &z, const pair<float, float> &v_x,
        const pair<float, float> &v_y, const pair<float, float> &v_z, const pair<float, float> &object_axes_length,
        const ImageRegistration &registration_data)
{
    clearParticles();
    m_particles.resize(M);
    
    this->registration_data = registration_data;

    for (CParticleList::iterator it = m_particles.begin(); it != m_particles.end(); it++) {
        it->d = new CImageParticleData();
        
        it->d->x = randomGenerator.drawGaussian1D(x.first, x.second);
        it->d->y = randomGenerator.drawGaussian1D(y.first, y.second);
        it->d->z = randomGenerator.drawGaussian1D(z.first, z.second);
        
        /*
        if (observation != nullptr){
            it->d->z  = depth_mat.at<DEPTH_TYPE>(cvRound(it->d->y), cvRound(it->d->x));
            it->d->vz = 0;
        } else{
            it->d->z  = randomGenerator.drawGaussian1D(z.first, z.second);
            it->d->vz = randomGenerator.drawGaussian1D(v_z.first, v_z.second);;
        }
        */

        it->d->vx = randomGenerator.drawGaussian1D(v_x.first, v_x.second);
        it->d->vy = randomGenerator.drawGaussian1D(v_y.first, v_y.second);
        it->d->vz = randomGenerator.drawGaussian1D(v_z.first, v_z.second);

        it->log_w = 0;
        
        std::tie(this->object_x_length, this->object_y_length) = object_axes_length;

        Eigen::Vector2i top_corner, bottom_corner;
        std::tie(top_corner, bottom_corner) = project_model(Eigen::Vector2f(it->d->x, it->d->y), it->d->z,
            Eigen::Vector2f(this->object_x_length * 0.5, this->object_y_length * 0.5),
            registration_data.cameraMatrixColor, registration_data.lookupX, registration_data.lookupY);
        
        it->d->object_x_length_pixels = cvRound((bottom_corner - top_corner)[0]);
        it->d->object_y_length_pixels = cvRound((bottom_corner - top_corner)[1]);
    }
}

template<typename DEPTH_TYPE>
float CImageParticleFilter<DEPTH_TYPE>::get_mean(float &x, float &y, float &z, float &vx, float &vy, float &vz) const
{
    auto m_particles_filtered = m_particles;
    /*
    std::sort(m_particles_filtered.begin(), m_particles_filtered.end(), 
        [this](decltype(m_particles_filtered)::value_type &a, decltype(m_particles_filtered)::value_type &b)
            { 
                return a.log_w > b.log_w;
            }
    );
    m_particles_filtered.resize(size_t(m_particles_filtered.size() * 0.20));
    */
    
    double sumW = 0;
#ifdef USE_INTEL_TBB
    sumW = tbb::parallel_reduce(
        tbb::blocked_range<CParticleList::const_iterator>(m_particles_filtered.begin(), m_particles_filtered.end(),
            m_particles_filtered.size() / TBB_PARTITIONS), 0.f,
                [](const tbb::blocked_range<CParticleList::const_iterator> &r, double value) -> double {
                    return std::accumulate(r.begin(), r.end(), value,
                        [](double value, const CParticleData &p) -> double {
                            return exp(p.log_w) + value;
                        }
                    );
                },
            std::plus<double>()
        );
#else
    //std::cout << "SORTED ########################################################" << std::endl;
    for (CParticleList::iterator it = m_particles_filtered.begin(); it != m_particles_filtered.end(); it++) {
        //std::cout << "SORTED " << exp(it->log_w) << std::endl;
        sumW += exp(it->log_w);
    }
#endif

    std::cout << "MEAN WEIGHT " << sumW / m_particles.size() << std::endl;
    ASSERT_(sumW > 0)

    x = 0;
    y = 0;
    z = 0;
    vx = 0;
    vy = 0;
    vz = 0;
    const double inv_sumW = 1.0 / sumW;
    for (CParticleList::const_iterator it = m_particles_filtered.begin(); it != m_particles_filtered.end(); it++) {
        const double w = exp(it->log_w) * inv_sumW;
        x += float(w * it->d->x);
        y += float(w * it->d->y);
        z += float(w * it->d->z);

        vx += float(w * it->d->vx);
        vy += float(w * it->d->vy);
        vz += float(w * it->d->vz);
    }

    /*
    double max_w = std::numeric_limits<double>::min();
    CParticleList::const_iterator max_it;
    for (CParticleList::const_iterator it = m_particles_filtered.begin(); it != m_particles_filtered.end(); it++) {
        const double w = exp(it->log_w) * inv_sumW;
        if (w > max_w){
            max_it = it;
        }
    }

    x = float(max_it->d->x);
    y = float(max_it->d->y);
    z = float(max_it->d->z);

    vx = float(max_it->d->vx);
    vy = float(max_it->d->vy);
    vz = float(max_it->d->vz);
    */


    cout << "PARTICLES USED " << m_particles_filtered.size() << endl;
    return sumW / m_particles.size();
}
