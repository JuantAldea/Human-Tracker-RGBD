#pragma once

#include <opencv2/opencv.hpp>
#include <boost/math/distributions/normal.hpp>
#include "StateEstimation.h"
#include "Tracker.h"

template <typename DEPTH_TYPE>
struct MultiTracker {
    ImageRegistration reg;
    boost::math::normal_distribution<float> depth_distribution;
    std::vector<CImageParticleFilter<DEPTH_TYPE>> trackers;
    std::vector<StateEstimation> states;
    std::vector<StateEstimation> new_states;
    std::vector<Eigen::Vector2f> ellipse_normals;

    MultiTracker(const ImageRegistration &ir) :
        reg(ir),
        depth_distribution(0, DEPTH_SIGMA),
        ellipse_normals(calculate_ellipse_normals(MODEL_SEMIAXIS_X_METTERS, MODEL_SEMIAXIS_Y_METTERS,
                        ELLIPSE_FITTING_ANGLE_STEP))
    {
        ;
    };

    void insert_tracker(const cv::Point &center, const float center_depth,
                        const cv::Mat &hsv_frame, EllipseStash &ellipses)
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

        trackers.push_back(CImageParticleFilter<DEPTH_TYPE>(&ellipses, &reg, &depth_distribution));
        states.push_back(StateEstimation());
        new_states.push_back(StateEstimation());
        init_tracking<DEPTH_TYPE>(center, center_depth, hsv_frame, ellipse_normals,
                                  trackers.back(), states.back(), ellipses);
    };

    void tracking(const cv::Mat &hsv_frame, const cv::Mat &depth_frame,
                  const cv::Mat &gradient_vectors, const CSensoryFrame &observation,
                  CParticleFilter &PF, EllipseStash &ellipses)
    {
        const size_t N = trackers.size();
        std::cout << "TRACKERS " << N << std::endl;
        for (size_t i = 0; i < N; i++) {
            CImageParticleFilter<DEPTH_TYPE> &particles = trackers[i];
            StateEstimation &estimated_new_state = new_states[i];
            const StateEstimation &estimated_state = states[i];
            static CParticleFilter::TParticleFilterStats stats;
            float mean_weight;
            do_tracking(estimated_new_state, PF, particles, observation, stats, mean_weight);
            build_visual_model<DEPTH_TYPE>(estimated_state, estimated_new_state, hsv_frame,
                                           depth_frame, ellipses);
            score_visual_model(estimated_new_state, particles, gradient_vectors, ellipse_normals);
            particles.last_time = cv::getTickCount();
        }
    };

    void update(EllipseStash &ellipses)
    {
        const size_t N = trackers.size();
        for (size_t i = 0; i < N; i++) {
            StateEstimation &estimated_new_state = new_states[i];
            CImageParticleFilter<DEPTH_TYPE> &particles = trackers[i];
            StateEstimation &estimated_state = states[i];
            const float score = estimated_new_state.score_total;

            if (score > LIKEHOOD_FOUND) {
                cv::Mat blended_color_model(estimated_state.color_model.rows, estimated_state.color_model.cols, estimated_state.color_model.type());
                cv::addWeighted(estimated_state.color_model, 1 - score, estimated_new_state.color_model, score, 0, blended_color_model);
                //estimated_new_state.color_model = blended_color_model;
                estimated_state = estimated_new_state;
                estimated_state.color_model = blended_color_model;
                particles.set_color_model(estimated_state.color_model);
                //blended_color_model = estimated_new_state.color_model;
                particles.last_distance = estimated_state.z;
            }

            if (score > LIKEHOOD_UPDATE) {
            }

            if (score < LIKEHOOD_FOUND) {
                /*
                particles.init_particles(NUM_PARTICLES,
                                  make_pair(estimated_state.x, estimated_state.radius_x), make_pair(estimated_state.y, estimated_state.radius_y),
                                  make_pair(float(estimated_state.z), 100.f),
                                  make_pair(0, 0), make_pair(0, 0), make_pair(0, 0),
                                  make_pair(MODEL_AXIS_X_METTERS, MODEL_AXIS_Y_METTERS),
                                  reg, &ellipses
                                 );
                estimated_state.factor *= 1.2;
                */

            }
        }
    };

    void show(cv::Mat &color_display_frame, const cv::Mat &depth_frame)
    {
        const size_t N = trackers.size();
        for (size_t i = 0; i < N; i++) {
            const CImageParticleFilter<DEPTH_TYPE> &particles = trackers[i];
            //const StateEstimation &estimated_state = new_states[i];
            const StateEstimation &estimated_state = states[i];
            const StateEstimation &estimated_new_state = new_states[i];

            if (estimated_state.score_total > LIKEHOOD_FOUND) {
                cv::ellipse(color_display_frame, estimated_state.center, cv::Size(3, 3), 0, 0, 360, cv::Scalar(0, 255, 0), -1, 8, 0);
                cv::ellipse(color_display_frame, estimated_state.center, cv::Size(estimated_state.radius_x, estimated_state.radius_y), 0, 0, 360, cv::Scalar(0, 0, 255), 3, 8, 0);
            }

            if (estimated_state.score_total > LIKEHOOD_UPDATE) {
                cv::ellipse(color_display_frame, estimated_state.center, cv::Size(3, 3), 0, 0, 360, cv::Scalar(255, 255, 255), -1, 8, 0);
                cv::ellipse(color_display_frame, estimated_state.center, cv::Size(estimated_state.radius_x, estimated_state.radius_y), 0, 0, 360, cv::Scalar(255, 255, 255), 3, 8, 0);
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

            cv::circle(color_display_frame, cv::Point(estimated_state.x, estimated_state.y), 20,
                       cv::Scalar(255, 0, 0), 5, 1, 0);
            cv::circle(color_display_frame, estimated_state.center, 150, cv::Scalar(0, 0, 255), 3, 8, 0);
            const size_t N_PARTICLES = particles.m_particles.size();

            {
                std::ostringstream oss;
                std::ostringstream oss2;
                oss << i << ' ' << estimated_state.score_total << ' ' << estimated_state.score_shape << ' ' << estimated_state.score_color;
                oss2 << i << ' ' << estimated_new_state.score_total << ' ' << estimated_new_state.score_shape << ' ' << estimated_new_state.score_color;
                int fontFace =  cv::FONT_HERSHEY_PLAIN;
                double fontScale = 2;
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

        /*
        {
            const cv::Point frame_center(color_display_frame.cols * 0.5 , color_display_frame.rows * 0.5);
            const Eigen::Vector2f center(frame_center.x, frame_center.y);
            float depth = depth_frame.at<DEPTH_TYPE>(frame_center.y, frame_center.x);
            Eigen::Vector2i top_corner, bottom_corner;
            std::tie(top_corner, bottom_corner) = project_model(
                    center,
                    depth,
                    //Eigen::Vector2f(MODEL_SEMIAXIS_X_METTERS, MODEL_SEMIAXIS_Y_METTERS),
                    //Eigen::Vector2f(PERSON_TORSO_X_SEMIAXIS_METTERS, PERSON_TORSO_Y_SEMIAXIS_METTERS),
                    Eigen::Vector2f(PERSON_HEAD_X_SEMIAXIS_METTERS, PERSON_HEAD_Y_SEMIAXIS_METTERS),
                    reg.cameraMatrixColor, reg.lookupX, reg.lookupY);

            const Eigen::Vector2i diagonal_vector = bottom_corner - top_corner;
            int radius_x = cvRound(diagonal_vector[0] * 0.5);
            int radius_y = cvRound(diagonal_vector[1] * 0.5);
            //cv::circle(color_display_frame, frame_center, radius_x, cv::Scalar(0, 0, 255), 3, 8, 0);
            cv::ellipse(color_display_frame, frame_center, cv::Size(radius_x, radius_y), 0, 0, 360,
                        cv::Scalar(0, 0, 255), 3, 8, 0);
            const Eigen::Vector2i torso_center = project_vector(center, depth, Eigen::Vector3f(0, 0.45, 0),
                                                 reg.cameraMatrixColor, reg.lookupX, reg.lookupY);
            cv::ellipse(color_display_frame, cv::Point(torso_center[0], torso_center[1]),
                        cv::Size(radius_x * 2, radius_y * 2), 0, 0, 360, cv::Scalar(0, 0, 255), 3, 8, 0);
            {
                std::ostringstream oss;
                oss << "DEPTH:" << depth << " ";
                oss << "AXIS X: " << radius_x << " ";
                oss << "AXIS Y: " << radius_y << " ";

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
        }
        */

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

