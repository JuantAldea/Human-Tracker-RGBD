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
    const CObservationImagePtr obs_image = observation->getObservationByClass<CObservationImage>(0);
    const CObservationImagePtr obs_depth = observation->getObservationByClass<CObservationImage>(1);
    
    ASSERT_(obs_image);
    ASSERT_(obs_depth);

    const cv::Mat image_mat = cv::Mat(obs_image->image.getAs<IplImage>());
    const cv::Mat depth_mat = cv::Mat(obs_depth->image.getAs<IplImage>());
    auto update_particle = [&](const size_t i) {
        //TODO can x, y go outside of the frame?
        /*
        const double old_z = m_particles[i].d->z;
        const float old_x = m_particles[i].d->x;
        const float old_y = m_particles[i].d->y;
        */
        const double new_z = depth_mat.at<DEPTH_TYPE>(cvRound(m_particles[i].d->y), cvRound(m_particles[i].d->x));
        
        m_particles[i].d->x += dt * m_particles[i].d->vx + TRANSITION_MODEL_STD_XY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->y += dt * m_particles[i].d->vy + TRANSITION_MODEL_STD_XY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->z = new_z;
        //m_particles[i].d->z += dt * m_particles[i].d->vz + TRANSITION_MODEL_STD_XY * randomGenerator.drawGaussian1D_normalized();
        
        /*
        m_particles[i].d->vx = (m_particles[i].d->x - old_x) / dt + TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->vy = (m_particles[i].d->y - old_y) / dt + TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->vz = (m_particles[i].d->z - old_z) / dt + TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
        */

        m_particles[i].d->vx = TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->vy = TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
        m_particles[i].d->vz = TRANSITION_MODEL_STD_VXY * randomGenerator.drawGaussian1D_normalized();
        
        Eigen::Vector2i top_corner, bottom_corner;
        std::tie(top_corner, bottom_corner) = project_model(Eigen::Vector2f(m_particles[i].d->x, m_particles[i].d->y), m_particles[i].d->z,
            Eigen::Vector2f(this->object_x_length * 0.5, this->object_y_length * 0.5),
            registration_data.cameraMatrixColor, registration_data.lookupX, registration_data.lookupY);
        
        m_particles[i].d->object_x_length_pixels = cvRound((bottom_corner - top_corner)[0]);
        m_particles[i].d->object_y_length_pixels = cvRound((bottom_corner - top_corner)[1]);
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
void CImageParticleFilter<DEPTH_TYPE>::weight_particles_with_model(const mrpt::obs::CSensoryFrame * const observation)
{
    const CObservationImagePtr obs_image = observation->getObservationByClass<CObservationImage>(0);
    
    ASSERT_(obs_image);

    const cv::Mat image_mat = cv::Mat(obs_image->image.getAs<IplImage>());
    cv::Mat frame_hsv;
    cv::cvtColor(image_mat, frame_hsv, cv::COLOR_BGR2HSV);
    
    cv::Mat gradient_vectors, gradient_magnitude, gradient_magnitude_scaled;
    std::tie(gradient_vectors, gradient_magnitude, gradient_magnitude_scaled) = sobel_operator(image_mat);
    

    size_t N = m_particles.size();

    vector <cv::Mat> particles_color_model(N);
    vector <float> particles_ellipse_fitting(N);

    auto compute_particles_color_model = [&](size_t i){
        const cv::Rect particle_roi(
            m_particles[i].d->x - m_particles[i].d->object_x_length_pixels * 0.5,
            m_particles[i].d->y - m_particles[i].d->object_y_length_pixels * 0.5,
            m_particles[i].d->object_x_length_pixels, m_particles[i].d->object_y_length_pixels
        );
        
        if (particle_roi.x < 0 || particle_roi.y < 0 || particle_roi.width <= 0
                || particle_roi.height <= 0) {
            return;
        }

        if (particle_roi.x + particle_roi.width >= frame_hsv.cols
                || particle_roi.y + particle_roi.height >= frame_hsv.rows) {
            return;
        }

        const cv::Mat mask = create_ellipse_mask(particle_roi, 1);
        const cv::Mat particle_roi_img = frame_hsv(particle_roi);
        const cv::Mat roi_img(particle_roi_img.size(), particle_roi_img.type());
        cv::Mat mask_3C;
        cv::merge(std::vector<cv::Mat>{mask, mask, mask}, mask_3C);
        bitwise_and(image_mat(particle_roi), mask_3C, roi_img);

        if (i == 0){
            particle_image.loadFromIplImage(new IplImage(particle_roi_img));
            particle_window.showImage(particle_image);
            //std::cout << "m_particles[i].d->object_x_length_pixels " << m_particles[i].d->object_x_length_pixels << std::endl;
            //std::cout << "MASK: " << mask.rows << ' ' << mask.cols << std::endl;
            //std::cout << "MASKROI: " << particle_roi_img.rows << ' ' << particle_roi_img.cols << std::endl;
        }

        particles_color_model[i] = compute_color_model(particle_roi_img, mask);
    };

#ifdef USE_INTEL_TBB
    //tbb::concurrent_vector <cv::Mat> particles_color_model(N);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, N / TBB_PARTITIONS), 
        [this, &frame_hsv, &particles_color_model, &compute_particles_color_model, &gradient_vectors, &gradient_magnitude, &particles_ellipse_fitting](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); i++) {
                compute_particles_color_model(i);
                if (particles_color_model[i].empty()){
                    continue;
                }
                //continue;
                if (m_particles[i].d->object_x_length_pixels == 0 ||  m_particles[i].d->object_y_length_pixels == 0){
                   continue;
                }
                particles_ellipse_fitting[i] = ellipse_contour_test(
                    cv::Point(m_particles[i].d->x, m_particles[i].d->y),
                    m_particles[i].d->object_x_length_pixels * 0.5, m_particles[i].d->object_y_length_pixels * 0.5, ELLIPSE_FITTING_ANGLE_STEP, gradient_vectors, gradient_magnitude);

                if (i == 0){
                    std::cout << "FITTING 0 " << particles_ellipse_fitting[i] << std::endl;
                    //std::cout << "m_particles[i].d->object_x_length_pixels " << m_particles[i].d->object_x_length_pixels << std::endl;
                    //std::cout << "MASK: " << mask.rows << ' ' << mask.cols << std::endl;
                    //std::cout << "MASKROI: " << particle_roi_img.rows << ' ' << particle_roi_img.cols << std::endl;
                }
            }
        }
    );
#else
    for (size_t i = 0; i < N; i++) {
        compute_particles_color_model(i);
        
        if(particles_color_model[i].empty()){
            continue;
        }
        
        //continue;
        if (m_particles[i].d->object_x_length_pixels == 0 ||  m_particles[i].d->object_y_length_pixels == 0){
            continue;
        }

        particles_ellipse_fitting[i] = ellipse_contour_test(
                    cv::Point(m_particles[i].d->x, m_particles[i].d->y),
                    m_particles[i].d->object_x_length_pixels * 0.5, m_particles[i].d->object_y_length_pixels * 0.5, ELLIPSE_FITTING_ANGLE_STEP, gradient_vectors, gradient_magnitude);
    }
#endif
    double sum_gradient_fitting = 0;
#ifdef USE_INTEL_TBB
    sum_gradient_fitting = tbb::parallel_reduce(
        tbb::blocked_range<vector<float>::const_iterator>(particles_ellipse_fitting.begin(), particles_ellipse_fitting.end(),
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
    if (particles_ellipse_fitting[i] == 0){
        continue;
    }

    min_fitting = std::min(min_fitting, particles_ellipse_fitting[i]);
    max_fitting = std::max(max_fitting, particles_ellipse_fitting[i]);
}

float range_fitting = max_fitting - min_fitting;
std::cout << "MAX " << max_fitting << " min " << min_fitting << std::endl;
#ifdef USE_INTEL_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, N / TBB_PARTITIONS),
        [this, &particles_ellipse_fitting, sum_gradient_fitting, min_fitting, range_fitting](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); i++) {
                if (particles_ellipse_fitting[i] == 0){
                    continue;
                }
                particles_ellipse_fitting[i] = (particles_ellipse_fitting[i] - min_fitting) / range_fitting;
            }
        }
    );
#else
    for (size_t i = 0; i < N; i++) {
        if (particles_ellipse_fitting[i] == 0){
            continue;
        }
        particles_ellipse_fitting[i] = (particles_ellipse_fitting[i] - min_fitting) / range_fitting;
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
    auto weight_particle = [this, &particles_color_model, &particles_ellipse_fitting] (size_t i){
        if (!particles_color_model[i].empty()) {
            const double distance_hist = cv::compareHist(color_model, particles_color_model[i], CV_COMP_BHATTACHARYYA);
            double score = 1;
            //score *= (1 - distance_hist) * particles_ellipse_fitting[i];
            score *= (1 - distance_hist);
            //score *= particles_ellipse_fitting[i];
            //std::cout << "SCORE: " << (1 - distance_hist) * particles_ellipse_fitting[i] << ' ' << (1 - distance_hist) << ' ' << particles_ellipse_fitting[i] << std::endl;
            m_particles[i].log_w += log(score);
        } else {
            m_particles[i].log_w += log(std::numeric_limits<double>::min());
            //m_particles[i].log_w += 0;
        }
    };

#ifdef USE_INTEL_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, N / TBB_PARTITIONS),
        [this, &particles_color_model, &weight_particle](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); i++) {
                weight_particle(i);
            }
        }
    );
#else
    for (size_t i = 0; i < N; i++) {
        weight_particle(i);
    }
#endif

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

    for (CParticleList::const_iterator it = m_particles_filtered.begin(); it != m_particles_filtered.end(); it++) {
        const double w = exp(it->log_w) / sumW;
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
        const double w = exp(it->log_w) / sumW;
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
