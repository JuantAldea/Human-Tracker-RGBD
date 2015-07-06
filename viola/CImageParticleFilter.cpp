#pragma once

#include <cmath>
#include <limits>

template<typename DEPTH_TYPE>
CImageParticleFilter<DEPTH_TYPE>::CImageParticleFilter(EllipseStash *ellipses, ImageRegistration *reg, normal_dist *normal_distribution) :
    ellipses(ellipses),
    registration_data(reg),
    depth_normal_distribution(normal_distribution)
{
    object_found = true;
    object_times_missing = 0;
    transition_model_std_xy = MODEL_TRANSITION_STD_XY;
    missing_uncertaincy_multipler = MODEL_MISSING_UNCERTAINCY_MULTIPLER;
}

template<typename DEPTH_TYPE>
void CImageParticleFilter<DEPTH_TYPE>::print_particle_state(void) const
{
    size_t N = m_particles.size();
    for (size_t i = 0; i < N; i++) {
        std::cout << i << ' '
            << std::exp(m_particles[i].log_w) << ' '
            << m_particles[i].d->x << ' '
            << m_particles[i].d->y << ' '
            << m_particles[i].d->z << ' '
            << m_particles[i].d->vx << ' '
            << m_particles[i].d->vy << ' '
            << m_particles[i].d->vz << ' '
            << m_particles[i].d->valid << ' '
            << std::endl;
    }
}

template<typename DEPTH_TYPE>
void CImageParticleFilter<DEPTH_TYPE>::set_color_model(const cv::Mat &model)
{
    color_model = model.clone();
}

template<typename DEPTH_TYPE>
const cv::Mat & CImageParticleFilter<DEPTH_TYPE>::get_color_model() const
{
    return color_model;
}

template<typename DEPTH_TYPE>
void CImageParticleFilter<DEPTH_TYPE>::set_object_found()
{
    object_found = true;
    object_times_missing = 0;
    transition_model_std_xy = MODEL_TRANSITION_STD_XY;
}

template<typename DEPTH_TYPE>
void CImageParticleFilter<DEPTH_TYPE>::set_object_missing()
{
    object_found = false;
    object_times_missing += 1;
    transition_model_std_xy *= missing_uncertaincy_multipler;
}

template<typename DEPTH_TYPE>
void CImageParticleFilter<DEPTH_TYPE>::set_shape_model(const vector<Eigen::Vector2f> &normal_vectors)
{
    std::cout << "NORMALS: " << normal_vectors.size() << std::endl;
    shape_model = const_cast<vector<Eigen::Vector2f>*>(std::addressof(normal_vectors));
}

template<typename DEPTH_TYPE>
void CImageParticleFilter<DEPTH_TYPE>::update_particles_with_transition_model(const double dt, const mrpt::obs::CSensoryFrame * const observation)
{
    const CObservationImagePtr image_depth = observation->getObservationBySensorLabelAs<CObservationImagePtr>("depth");

    ASSERT_(image_depth);

    const cv::Mat depth_mat = cv::Mat(image_depth->image.getAs<IplImage>());

    auto update_particle = [&](const size_t i) {
        const double old_z = m_particles[i].d->z;
        const float old_x = m_particles[i].d->x;
        const float old_y = m_particles[i].d->y;


        m_particles[i].d->x += dt * m_particles[i].d->vx + transition_model_std_xy * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->y += dt * m_particles[i].d->vy + transition_model_std_xy * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->z = 0;

        if (point_in_mat(m_particles[i].d->x, m_particles[i].d->y, depth_mat)) {
            m_particles[i].d->z = depth_mat.at<DEPTH_TYPE>(cvRound(m_particles[i].d->y), cvRound(m_particles[i].d->x));
        }

        const double inv_dt = 1.0 / dt;
        m_particles[i].d->vx = object_found * ((m_particles[i].d->x - old_x) * inv_dt + MODEL_TRANSITION_STD_VXY * randomGenerator.drawGaussian1D_normalized());
        m_particles[i].d->vy = object_found * ((m_particles[i].d->y - old_y) * inv_dt + MODEL_TRANSITION_STD_VXY * randomGenerator.drawGaussian1D_normalized());
        m_particles[i].d->vz = object_found * ((m_particles[i].d->z - old_z) * inv_dt + MODEL_TRANSITION_STD_VXY * randomGenerator.drawGaussian1D_normalized());

        m_particles[i].d->valid = false;

        if(m_particles[i].d->z != 0){
            const cv::Size ellipse_axes = ellipses->get_ellipse_size(BodyPart::HEAD, m_particles[i].d->z);

            const cv::Rect particle_roi = cv::Rect(cvRound(m_particles[i].d->x - ellipse_axes.width * 0.5f),
                                                   cvRound(m_particles[i].d->y - ellipse_axes.height * 0.5f),
                                                   ellipse_axes.width, ellipse_axes.height);

             m_particles[i].d->valid = rect_fits_in_frame(particle_roi, depth_mat);
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
void CImageParticleFilter<DEPTH_TYPE>::split_particles()
{
    particles_valid_roi.clear();
    particles_invalid_roi.clear();
    for (typename decltype(m_particles)::iterator it = m_particles.begin(); it != m_particles.end(); it++){
        if (it->d->valid) {
            particles_valid_roi.push_back(*it);
        } else {
            particles_invalid_roi.push_back(*it);
        }
    }
}

template<typename DEPTH_TYPE>
cv::Mat compute_valid_particle_color_model(const typename CImageParticleFilter<DEPTH_TYPE>::ParticleType &particle,
                                           const cv::Mat &frame_hsv, const EllipseStash &ellipses)
{
    const cv::Mat &mask = ellipses.get_ellipse_mask_1D(BodyPart::HEAD, particle.d->z);
    const cv::Mat &mask_weights = ellipses.get_ellipse_mask_weights(BodyPart::HEAD, particle.d->z);

    const cv::Rect particle_roi = cv::Rect(
        particle.d->x - mask.cols * 0.5,
        particle.d->y - mask.rows * 0.5,
        mask.cols, mask.rows);

    const cv::Mat particle_roi_img = frame_hsv(particle_roi);

    return compute_color_model2(particle_roi_img, mask_weights);
}
template<typename DEPTH_TYPE>
float compute_valid_particle_ellipse_fitting(const typename CImageParticleFilter<DEPTH_TYPE>::ParticleType &particle,
                                             const cv::Mat &gradient_vectors, const cv::Mat &gradient_magnitude,
                                             const vector<Eigen::Vector2f> &shape_model, const EllipseStash &ellipses)
{
    const cv::Size ellipse_axes = ellipses.get_ellipse_size(BodyPart::HEAD, particle.d->z);
    //TODO CHANGE THIS TO DO THE TEST OVER A ROI
    const float fitting_score = ellipse_contour_test(cv::Point(particle.d->x, particle.d->y),
                                                     ellipse_axes.width * 0.5,
                                                     ellipse_axes.height * 0.5,
                                                     shape_model, gradient_vectors,
                                                     gradient_magnitude, nullptr);
    return fitting_score;
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

    if (!particles_valid_roi.size()){
        throw;
    }

    size_t N = particles_valid_roi.size();

    vector<cv::Mat> particles_color_model(N);
    vector<float> particles_ellipse_fitting(N);

    auto compute_particle_color_model = [&](const ParticleType &particle){
        const cv::Mat &mask = ellipses->get_ellipse_mask_1D(BodyPart::HEAD, particle.d->z);
        const cv::Mat &mask_weights = ellipses->get_ellipse_mask_weights(BodyPart::HEAD, particle.d->z);

        const cv::Rect particle_roi = cv::Rect(
            particle.d->x - mask.cols * 0.5,
            particle.d->y - mask.rows * 0.5,
            mask.cols, mask.rows);

        const cv::Mat particle_roi_img = frame_hsv(particle_roi);
        const cv::Mat color_model = compute_color_model2(particle_roi_img, mask_weights);

#ifdef DEBUG
        if (i == 0){
            particle_image.loadFromIplImage(new IplImage(particle_roi_img));
            particle_window.showImage(particle_image);
        }
#endif
        //return compute_color_model(particle_roi_img, mask);
        return color_model;
    };

    auto compute_particle_ellipse_fitting = [&](const ParticleType &particle){
        const cv::Size ellipse_axes = ellipses->get_ellipse_size(BodyPart::HEAD, particle.d->z);
        //TODO CHANGE THIS TO DO THE TEST OVER A ROI
        const float fitting = ellipse_contour_test(cv::Point(particle.d->x, particle.d->y),
                                             ellipse_axes.width * 0.5,
                                             ellipse_axes.height * 0.5,
                                             *shape_model, gradient_vectors,
                                             gradient_magnitude, nullptr);
#ifdef DEBUG
        if (i == 0){
            std::cout << "FITTING 0 " << particles_ellipse_fitting[i] << std::endl;
        }
#endif
        return fitting;
    };

#ifdef USE_INTEL_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, N / TBB_PARTITIONS),
        [this, &frame_hsv, &particles_color_model, &compute_valid_particle_color_model,
            &gradient_vectors, &gradient_magnitude, &particles_ellipse_fitting](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); i++) {
                const ParticleType &particle = particles_valid_roi[i];
                /*
                particles_color_model[i] = compute_particle_color_model(particle, frame_hsv);
                particles_ellipse_fitting[i] = compute_valid_particle_ellipse_fitting(particle, gradient_vectors,
                    gradient_magnitude, *shape_model);
                */
                particles_color_model[i] = compute_particle_color_model(particle);
                particles_ellipse_fitting[i] = compute_particle_ellipse_fitting(particle);
            }
        }
    );
#else
    for (size_t i = 0; i < N; i++) {
        const ParticleType &particle = particles_valid_roi[i];
        particles_color_model[i] = compute_particle_color_model(particle);
        particles_ellipse_fitting[i] = compute_particle_ellipse_fitting(particle);
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


float max_fitting = std::numeric_limits<float>::min();
float min_fitting = std::numeric_limits<float>::max();;

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
    auto weight_valid_particle = [this, &particles_color_model, &particles_ellipse_fitting] (const size_t i){
        const float distance_hist = cv::compareHist(color_model, particles_color_model[i], CV_COMP_BHATTACHARYYA);
        const float color_score  = (1 - distance_hist);
        const float fitting_score = particles_ellipse_fitting[i];
        const float z_score = 1 - (2 * cdf(*depth_normal_distribution, std::abs(particles_valid_roi[i].get().d->z - last_distance) * 0.001) - 1);
        //printf("%f %f\n", std::abs(particles_valid_roi[i].get().d->z - last_distance) * 0.001, z_score);

        float score = 1;
        score *= color_score;
        score *= fitting_score;
        score *= z_score;

        if (!object_found){
            score = score > 0.2 ? score : 0;
        }

        particles_valid_roi[i].get().log_w += log(score);
        //printf("%f %f %f = %f (%f)\n", color_score, fitting_score, z_score, score, particles_valid_roi[i].get().log_w);
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
    //constexpr double w_invalid = log(std::numeric_limits<double>::min());
    constexpr double w_invalid = log(0.001);
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
    const int64_t current_time = cv::getTickCount();

    const double dt = (current_time - last_time) / cv::getTickFrequency();
    last_time = current_time;

    update_particles_with_transition_model(dt, observation);

    weight_particles_with_model(observation);
    //print_particle_state();
    // Resample is automatically performed by CParticleFilter when required.
}

template<typename DEPTH_TYPE>
void CImageParticleFilter<DEPTH_TYPE>::init_particles(const size_t M, const pair<float, float> &x,
        const pair<float, float> &y, const pair<float, float> &z, const pair<float, float> &v_x,
        const pair<float, float> &v_y, const pair<float, float> &v_z)
{
    clearParticles();
    m_particles.resize(M);

    for (CParticleList::iterator it = m_particles.begin(); it != m_particles.end(); it++) {
        it->d = new ParticleData();

        it->d->x = randomGenerator.drawGaussian1D(x.first, x.second);
        it->d->y = randomGenerator.drawGaussian1D(y.first, y.second);
        it->d->z = randomGenerator.drawGaussian1D(z.first, z.second);

        it->d->vx = randomGenerator.drawGaussian1D(v_x.first, v_x.second);
        it->d->vy = randomGenerator.drawGaussian1D(v_y.first, v_y.second);
        it->d->vz = randomGenerator.drawGaussian1D(v_z.first, v_z.second);

        it->log_w = 0;
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
    for (CParticleList::iterator it = m_particles_filtered.begin(); it != m_particles_filtered.end(); it++) {
        sumW += exp(it->log_w);
    }
#endif

    //std::cout << "MEAN WEIGHT " << sumW / m_particles.size() << std::endl;
    //ASSERT_(sumW > 0)

    if (sumW <= 0){
        throw;
    }

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

    return sumW / m_particles.size();
}
