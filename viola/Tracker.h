#pragma once

#include "StateEstimation.h"
#include "EllipseStash.h"

template<typename DEPTH_TYPE>
bool init_tracking(const cv::Point &center, DEPTH_TYPE center_depth, const cv::Mat &hsv_frame,
                   const vector<Eigen::Vector2f> &shape_model,
                   const ImageRegistration &reg, CImageParticleFilter<DEPTH_TYPE> &particles,
                   StateEstimation &state, EllipseStash &ellipses)
{
    state.x = center.x;
    state.y = center.y;
    state.z = center_depth;
    state.v_x = 0;
    state.v_y = 0;
    state.v_z = 0;
    state.score_total = 1;
    state.score_color = 1;
    state.score_shape = 1;
    state.center = center;
    const cv::Mat mask = ellipses.get_ellipse_mask_1D(BodyPart::HEAD, center_depth);
    const cv::Mat mask_weights = ellipses.get_ellipse_mask_weights(BodyPart::HEAD, center_depth);

    state.radius_x = mask.cols * 0.5;
    state.radius_y = mask.rows * 0.5;
    state.region = cv::Rect(center.x - state.radius_x, center.y - state.radius_y, mask.cols, mask.rows);

    //const cv::Mat mask = fast_create_ellipse_mask(state.region, 1, n_pixels);
    const cv::Mat hsv_roi = hsv_frame(state.region);
    //state.color_model = compute_color_model(hsv_roi, mask);
    state.color_model = compute_color_model2(hsv_roi, mask_weights);

    particles.set_color_model(state.color_model);
    particles.set_shape_model(shape_model);
    particles.last_distance = state.z;

    particles.initializeParticles(NUM_PARTICLES,
                                  make_pair(state.x, state.radius_x), make_pair(state.y, state.radius_y),
                                  make_pair(float(state.z), 100.f),
                                  make_pair(0, 0), make_pair(0, 0), make_pair(0, 0),
                                  make_pair(MODEL_AXIS_X_METTERS, MODEL_AXIS_Y_METTERS),
                                  reg, &ellipses
                                 );
    std::cout << "MEAN detected circle: " << state.x << ' ' << state.y << ' ' << state.z << std::endl;

    /*
    {
        //const cv::Mat model2 = compute_color_model(hsv_roi, mask);
        //CImage model_histogram_image2;
        //model_histogram_image2.loadFromIplImage(new IplImage(histogram_to_image(model2, 10)));
        //model_histogram_window2.showImage(model_histogram_image2);
        cout << "SON IGUALES? " << model2.size() << ' ' << model.size() << std::endl;
        cout << "SON IGUALES? " << type2str(model2.type()) << ' ' << type2str(model.type()) << std::endl;
        cout << "SON IGUALES? " << cv::norm(model2, model) << std::endl;
        assert(cv::norm(model2, model) == 0);
    }
    */
    /*
    {
        const cv::Mat model_frame = cv::Mat(color_frame(model_roi).size(), color_frame.type());
        const cv::Mat mask = cv::Mat::ones(color_frame(model_roi).size(), color_frame(model_roi).type());
        bitwise_and(color_frame(model_roi), ones, model_frame, mask);
        CImage model_frame_image;
        model_frame_image.loadFromIplImage(new IplImage(color_frame(model_roi)));
        model_image_window.showImage(model_frame_image);
        CImage model_histogram_image;
        model_histogram_image.loadFromIplImage(new IplImage(histogram_to_image(particles.color_model, 10)));
        model_histogram_window.showImage(model_histogram_image);
    }
    */
    return true;
};



template <typename DEPTH_TYPE>
void do_tracking(StateEstimation &state, CParticleFilter &PF,
                 CImageParticleFilter<DEPTH_TYPE> &particles,
                 const CSensoryFrame &observation, CParticleFilter::TParticleFilterStats &stats,
                 float &mean_weight)
{
    PF.executeOn(particles, NULL, &observation, &stats);
    mean_weight = particles.get_mean(state.x, state.y, state.z, state.v_x, state.v_y, state.v_z);
    //cout << "ESS_beforeResample " << stats.ESS_beforeResample << " weightsVariance_beforeResample " << stats.weightsVariance_beforeResample << std::endl;
    //cout << "Particle filter ESS: " << particles.ESS() << endl;
}

template <typename DEPTH_TYPE>
void build_visual_model(const StateEstimation &old_state, StateEstimation &new_state,
                        const cv::Mat &hsv_frame, const cv::Mat &depth_frame,
                        EllipseStash &ellipses)
{
    Eigen::Vector2i top_corner, bottom_corner;
    const double center_measured_depth = depth_frame.at<DEPTH_TYPE>(cvRound(new_state.y),
                                         cvRound(new_state.x));
    //TODO USE MEAN DEPTH OF THE ELLIPSE
    const double z = center_measured_depth > 0 ? center_measured_depth : old_state.z;
    /*
    {
        std::tie(top_corner, bottom_corner) = project_model(Eigen::Vector2f(new_state.x, new_state.y),
                                              z,
                                              Eigen::Vector2f(MODEL_SEMIAXIS_X_METTERS, MODEL_SEMIAXIS_Y_METTERS), reg.cameraMatrixColor,
                                              reg.lookupX, reg.lookupY);

        new_state.radius_x = (bottom_corner[0] - top_corner[0]) * 0.5;
        new_state.radius_y = (bottom_corner[1] - top_corner[1]) * 0.5;
        new_state.center = cv::Point(new_state.x, new_state.y);
        new_state.region = cv::Rect(top_corner[0], top_corner[1], bottom_corner[0] - top_corner[0],
                                    bottom_corner[1] - top_corner[1]);
    }
    */

    {
        const cv::Size projection_size = ellipses.get_ellipse_size(BodyPart::HEAD, z);
        new_state.radius_x = projection_size.width * 0.5;
        new_state.radius_y = projection_size.height * 0.5;
        new_state.center = cv::Point(new_state.x, new_state.y);
        new_state.region = cv::Rect(new_state.x - new_state.radius_x, new_state.y - new_state.radius_y,
            projection_size.width, projection_size.height);
    }


    if (!rect_fits_in_frame(new_state.region, hsv_frame)){
        return;
    }

    //const cv::Mat mask = fast_create_ellipse_mask(new_state.region, 1, n_pixels);
    const cv::Mat mask = ellipses.get_ellipse_mask_1D(BodyPart::HEAD, z);
    const cv::Mat mask_weights = ellipses.get_ellipse_mask_weights(BodyPart::HEAD, z);
    cv::Mat hsv_roi = hsv_frame(new_state.region);
    //new_state.color_model = compute_color_model(hsv_roi, mask);
    new_state.color_model = compute_color_model2(hsv_roi, mask_weights);
}

template <typename DEPTH_TYPE>
void score_visual_model(StateEstimation &state,
                        const CImageParticleFilter<DEPTH_TYPE> &particles, const cv::Mat &gradient_vectors,
                        const std::vector<Eigen::Vector2f> &shape_model)
{
    if (state.color_model.empty()) {
        state.score_total = -1;
        state.score_color = -1;
        state.score_shape = -1;
        return;
    }

    state.score_color = 1 - cv::compareHist(state.color_model, particles.color_model,
                                            CV_COMP_BHATTACHARYYA);
    state.score_shape = ellipse_contour_test(state.center, state.radius_x, state.radius_y,
                        shape_model, gradient_vectors, cv::Mat(), nullptr);
    state.score_total = state.score_color * state.score_shape;
}
