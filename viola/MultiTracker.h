#pragma once

#include <opencv2/opencv.hpp>
#include <boost/math/distributions/normal.hpp>
#include "StateEstimation.h"
#include "Tracker.h"

template <typename DEPTH_TYPE>
struct MultiTracker {
    const ImageRegistration *reg;
    const boost::math::normal_distribution<float> depth_distribution;

    std::vector<CImageParticleFilter<DEPTH_TYPE>> trackers;
    std::vector<StateEstimation> states;
    std::vector<StateEstimation> new_states;
    std::vector<Eigen::Vector2f> ellipse_normals;

    MultiTracker(const ImageRegistration *ir) :
        reg(ir),
        depth_distribution(0, DEPTH_SIGMA),
        ellipse_normals(calculate_ellipse_normals(MODEL_SEMIAXIS_X_METTERS, MODEL_SEMIAXIS_Y_METTERS,
                        ELLIPSE_FITTING_ANGLE_STEP))
    {
        ;
    };

    void insert_tracker(const cv::Point &center, const float center_depth,
                        const cv::Mat &hsv_frame, const cv::Mat &depth_frame, EllipseStash &ellipses)
    {
        auto already_tracked = [&](const float distance) {
            if (states.empty()) {
                return false;
            }

            const float squared_distance = distance * distance;
            for (size_t i = 0; i < states.size(); i++) {
                const StateEstimation &state = states[i];
                const float d_x = state.x - center.x;
                const float d_y = state.y - center.y;
                const float d_x_squared = d_x * d_x;
                const float d_y_squared = d_y * d_y;
                if (d_x_squared + d_y_squared < squared_distance) {
                    return true;
                }
            }
            return false;
        };

        if (states.size()){
            return;
        }

        if (already_tracked(100)) {
            return;
        }

        const cv::Size projection_size = ellipses.get_ellipse_size(BodyPart::HEAD, center_depth);

        const int radius_x = projection_size.width * 0.5;
        const int radius_y = projection_size.height * 0.5;
        const cv::Rect region = cv::Rect(center.x - radius_x, center.y - radius_y, projection_size.width, projection_size.height);
        const bool fits = rect_fits_in_frame(region, hsv_frame);

        if (!fits){
            return;
        }

        Eigen::Vector2i torso_center = translate_2D_vector_in_3D_space(center.x, center.y, center_depth, HEAD_TO_TORSE_CENTER_VECTOR,
                                                                reg->cameraMatrixColor, reg->lookupX, reg->lookupY);

        const cv::Size ellipse_axes = ellipses.get_ellipse_size(BodyPart::TORSO, center_depth);
        const cv::Rect torso_roi = cv::Rect(cvRound(torso_center[0] - ellipse_axes.width * 0.5f),
                                           cvRound(torso_center[1] - ellipse_axes.height * 0.5f),
                                               ellipse_axes.width, ellipse_axes.height);

        const bool torso_in_frame = rect_fits_in_frame(torso_roi, hsv_frame);

        if (!torso_in_frame){
            return;
        }

        trackers.push_back(CImageParticleFilter<DEPTH_TYPE>(&ellipses, reg, &depth_distribution));
        states.push_back(StateEstimation());
        new_states.push_back(StateEstimation());
        init_tracking(center, center_depth, hsv_frame, depth_frame, ellipse_normals,
                                  trackers.back(), states.back(), ellipses, *reg);
    };

    void tracking(const cv::Mat &hsv_frame, const cv::Mat &depth_frame,
                  const cv::Mat &gradient_vectors, const CSensoryFrame &observation,
                  CParticleFilter &PF, EllipseStash &ellipses, const ImageRegistration &reg)
    {
        const size_t N = trackers.size();
        std::cout << "TRACKERS " << N << std::endl;
        for (size_t i = 0; i < N; i++) {
            CImageParticleFilter<DEPTH_TYPE> &particles = trackers[i];
            StateEstimation &estimated_new_state = new_states[i];
            const StateEstimation &estimated_state = states[i];
            static CParticleFilter::TParticleFilterStats stats;
            do_tracking(PF, particles, observation, stats);
            //printf("RADIUS0 %d %d %f - %d %d %f\n", estimated_state.radius_x, estimated_state.radius_y, estimated_state.z, estimated_new_state.radius_x, estimated_new_state.radius_y, estimated_new_state.z);
            build_state_model(particles, estimated_state, estimated_new_state, hsv_frame,
                depth_frame, ellipses, reg);

            score_visual_model(estimated_state, estimated_new_state, gradient_vectors, ellipse_normals, depth_distribution);
            //printf("RADIUS1 %d %d %f - %d %d %f\n", estimated_state.radius_x, estimated_state.radius_y, estimated_state.z, estimated_new_state.radius_x, estimated_new_state.radius_y, estimated_new_state.z);
            particles.last_time = cv::getTickCount();
        }
    };

    void update(EllipseStash &ellipses)
    {
        const size_t N = trackers.size();
        for (size_t i = 0; i < N; i++) {
            CImageParticleFilter<DEPTH_TYPE> &particles = trackers[i];
            StateEstimation &estimated_new_state = new_states[i];
            StateEstimation &estimated_state = states[i];
            const float score = estimated_new_state.score_total;
            //printf("RADIUS2 %d %d %f - %d %d %f\n", estimated_state.radius_x, estimated_state.radius_y, estimated_state.z, estimated_new_state.radius_x, estimated_new_state.radius_y, estimated_new_state.z);
            if (score > LIKEHOOD_FOUND) {
                estimated_state.blend(estimated_new_state);
                particles.set_head_color_model(estimated_state.color_model);
                particles.set_torso_color_model(estimated_state.torso_color_model);
                particles.last_distance = estimated_state.z;
                particles.set_object_found();
            }

            if (score > LIKEHOOD_UPDATE) {

            }

            if (score < LIKEHOOD_FOUND) {
                particles.set_object_missing();
                //estimated_new_state = estimated_state;
                particles.init_particles(NUM_PARTICLES,
                                  make_pair(estimated_state.x, estimated_state.radius_x * 2),
                                  make_pair(estimated_state.y, estimated_state.radius_y * 2),
                                  make_pair(float(estimated_state.z), 100.f),
                                  make_pair(0, 0), make_pair(0, 0), make_pair(0, 0));
                //printf("INIT STD %f %d %d\n", particles.missing_uncertaincy_multipler, estimated_state.radius_x, estimated_state.radius_y);
            }
        }
    };

    std::vector<StateEstimation> delete_missing()
    {
        const size_t N = trackers.size();
        vector<size_t> trackers_to_delete;
        for (size_t i = 0; i < N; i++) {
            CImageParticleFilter<DEPTH_TYPE> &particles = trackers[i];
            if (particles.object_times_missing > MISSING_TIMES_THRESHOLD){
                trackers_to_delete.push_back(i);
            }
        }

        const size_t N_to_delete = trackers_to_delete.size();
        std::vector<StateEstimation> deleted_states;
        for (size_t i = 0; i < N_to_delete; i++) {
            trackers.erase(trackers.begin() + i);
            new_states.erase(new_states.begin() + i);

            deleted_states.push_back(states[i]);
            states.erase(states.begin() + i);
        }

        trackers.shrink_to_fit();
        new_states.shrink_to_fit();
        states.shrink_to_fit();
        return deleted_states;
    }

    void show(cv::Mat &color_display_frame, const cv::Mat &depth_frame) const
    {
        const size_t N = trackers.size();
        for (size_t i = 0; i < N; i++) {
            const CImageParticleFilter<DEPTH_TYPE> &particles = trackers[i];
            const StateEstimation &estimated_state = states[i];
            const StateEstimation &estimated_new_state = new_states[i];

            if (estimated_new_state.score_total >= LIKEHOOD_FOUND) {
                cv::ellipse(color_display_frame, estimated_state.center, cv::Size(3, 3), 0, 0, 360, cv::Scalar(0, 255, 0), -1, 8, 0);
                cv::ellipse(color_display_frame, estimated_state.center, cv::Size(estimated_state.radius_x, estimated_state.radius_y), 0, 0, 360, cv::Scalar(0, 0, 255), 3, 8, 0);
            }

            if (estimated_new_state.score_total > LIKEHOOD_UPDATE) {
                //cv::ellipse(color_display_frame, estimated_state.center, cv::Size(3, 3), 0, 0, 360, cv::Scalar(255, 255, 255), -1, 8, 0);
                //cv::ellipse(color_display_frame, estimated_state.center, cv::Size(estimated_state.radius_x, estimated_state.radius_y), 0, 0, 360, cv::Scalar(255, 255, 255), 3, 8, 0);
            }

            if (estimated_new_state.score_total < LIKEHOOD_FOUND) {
                cv::ellipse(color_display_frame, estimated_new_state.center, cv::Size(3, 3), 0, 0, 360, cv::Scalar(0, 255, 255), -1, 8, 0);
                cv::ellipse(color_display_frame, estimated_new_state.center, cv::Size(estimated_new_state.radius_x, estimated_new_state.radius_y), 0, 0, 360, cv::Scalar(0, 255, 255), 3, 8, 0);

                cv::ellipse(color_display_frame, estimated_state.center, cv::Size(3, 3), 0, 0, 360, cv::Scalar(0, 255, 0), -1, 8, 0);
                cv::ellipse(color_display_frame, estimated_state.center, cv::Size(estimated_state.radius_x, estimated_state.radius_y), 0, 0, 360, cv::Scalar(0, 0, 255), 3, 8, 0);
            }


            /*
            model_candidate.loadFromIplImage(new IplImage(color_frame(estimated_state.region)));
            //model_candidate.loadFromIplImage(new IplImage(w_mask_img));
            model_candidate_window.showImage(model_candidate);

            CImage model_candidate_histogram_image;
            model_candidate_histogram_image.loadFromIplImage(new IplImage(histogram_to_image(estimated_state.color_model, 10)));
            model_candidate_histogram_window.showImage(model_candidate_histogram_image);
            cv::line(color_display_frame, cv::Point(estimated_state.x, estimated_state.y), cv::Point(estimated_state.x + estimated_state.v_x, estimated_state.y + estimated_state.v_y), cv::Scalar(0, 255, 0), 5, 1, 0);

            {
                std::ostringstream oss;
                oss << "SCORE:" << estimated_state.score_total << " ";
                oss << "BHAT: " << estimated_state.score_color << " ";
                oss << "SHAPE: " << estimated_state.score_shape << " ";
                const cv::Point frame_center(gradient_magnitude.cols * 0.5 , gradient_magnitude.rows * 0.5);

                float fitting_magnitude = ellipse_contour_test(frame_center, estimated_state.radius_x, estimated_state.radius_y, ellipse_normals, gradient_vectors, gradient_magnitude, &color_display_frame);
                float fitting_01 = ellipse_contour_test(frame_center, estimated_state.radius_x, estimated_state.radius_y, ellipse_normals, gradient_vectors, cv::Mat(), &gradient_magnitude_scaled);
                ellipse_contour_test(estimated_state.center, estimated_state.radius_x, estimated_state.radius_y, ellipse_normals, gradient_vectors, cv::Mat(), &gradient_magnitude_scaled);

                oss << "SHAPE CENTER: " << fitting_01<< " (" << fitting_magnitude << ")";
                int fontFace =  cv::FONT_HERSHEY_PLAIN;
                double fontScale = 2;
                int thickness = 2;

                int baseline = 0;
                cv::Size textSize = cv::getTextSize(oss.str(), fontFace, fontScale, thickness, &baseline);
                cv::Point textOrg(0, textSize.height + 10);
                //cv::Point textOrg(textSize.width, textSize.height);
                putText(color_display_frame, oss.str(), textOrg, fontFace, fontScale, cv::Scalar(255, 255, 0), thickness, 8);
            }
            */

            cv::circle(color_display_frame, cv::Point(estimated_state.x, estimated_state.y), 20, cv::Scalar(255, 0, 0), 5, 1, 0);
            const size_t N_PARTICLES = particles.m_particles.size();

            {
                std::ostringstream oss;
                std::ostringstream oss2;

                oss << i << ' ' << estimated_state.score_total << ' ' << estimated_state.score_shape
                    << ' ' << estimated_state.score_color << ' ' << estimated_state.torso_color_score
                    << ' ' << estimated_state.score_z << ' ' << trackers[i].transition_model_std_xy;

                oss2 << i << ' ' << estimated_new_state.score_total << ' ' << estimated_new_state.score_shape
                     << ' ' << estimated_new_state.score_color << ' ' << estimated_new_state.torso_color_score
                     << ' ' << estimated_new_state.score_z;

                int fontFace =  cv::FONT_HERSHEY_PLAIN;
                double fontScale = 1;
                int thickness = 2;

                int baseline = 0;
                cv::Size textSize = cv::getTextSize(oss.str(), fontFace, fontScale, thickness, &baseline);
                cv::Point textOrg(estimated_state.x - textSize.width * 0.5f, estimated_state.y - textSize.height * 0.5f);
                cv::Point textOrg2 = textOrg + cv::Point(0, textSize.height * 1.2);
                //cv::Point textOrg(textSize.width, textSize.height);
                putText(color_display_frame, oss.str(), textOrg, fontFace, fontScale, cv::Scalar(255, 255, 0), thickness, 8);
                putText(color_display_frame, oss2.str(), textOrg2, fontFace, fontScale, cv::Scalar(255, 255, 0), thickness, 8);
            }

            double max_w = -100;
            for (size_t j = 0; j < N_PARTICLES; j++) {
                max_w = max(max_w, particles.m_particles[j].log_w);
            }

            max_w = exp(max_w);

            for (size_t j = 0; j < N_PARTICLES; j++) {
                int radius = cvRound(1 + 1.0f/20 * max_w/exp(particles.m_particles[j].log_w) );
                radius = std::min(radius, 255);
                radius = std::max(radius, 1);
                radius = 1;
                cv::circle(color_display_frame,
                           cv::Point(particles.m_particles[j].d->x, particles.m_particles[j].d->y), radius,
                           GlobalColorPalette[i], 1, 1, 0);
            }
        }

        cv::line(color_display_frame, cv::Point(color_display_frame.cols * 0.5, 0),
                 cv::Point(color_display_frame.cols * 0.5, color_display_frame.rows - 1), cv::Scalar(0, 0,
                         255));
        cv::line(color_display_frame, cv::Point(0, color_display_frame.rows * 0.5),
                 cv::Point(color_display_frame.cols - 1, color_display_frame.rows * 0.5), cv::Scalar(0, 255,
                         0));

        if(states.size()){
            std::ostringstream oss;
            oss << "SCORE " << new_states[0].score_total << " fit "  << new_states[0].score_shape << " color " << new_states[0].score_color;

            int fontFace =  cv::FONT_HERSHEY_PLAIN;
            double fontScale = 2;
            int thickness = 2;

            int baseline = 0;
            cv::Size textSize = cv::getTextSize(oss.str(), fontFace, fontScale, thickness, &baseline);
            cv::Point textOrg(0, textSize.height + 10);
            //cv::Point textOrg(textSize.width, textSize.height);
            putText(color_display_frame, oss.str(), textOrg, fontFace, fontScale, cv::Scalar(255, 255, 0),
                    thickness, 8);
        }
    };
};

