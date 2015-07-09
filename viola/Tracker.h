#pragma once

#include "StateEstimation.h"
#include "EllipseStash.h"
CDisplayWindow image2("image2");

template<typename DEPTH_TYPE>
bool init_tracking(const cv::Point &center, float center_depth, const cv::Mat &hsv_frame, const cv::Mat &depth_frame,
                   const vector<Eigen::Vector2f> &shape_model, CImageParticleFilter<DEPTH_TYPE> &particles,
                   StateEstimation &state, EllipseStash &ellipses, const ImageRegistration &reg)
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

    const cv::Mat hsv_roi = hsv_frame(state.region);
    const cv::Mat depth_roi = depth_frame(state.region);
    cv::Mat depth_roi_masked;
    depth_roi.copyTo(depth_roi_masked, mask);

    float non_zero = cv::countNonZero(depth_roi_masked);
    cv::Scalar sum = cv::sum(depth_roi_masked);
    state.average_z = sum[0] / non_zero;
    /*
    {
        //TODO WTF?
        //double minVal, maxVal;
        //cv::minMaxLoc(depth_roi_masked, &minVal, &maxVal);
        //cv::Mat image_depth;
        //depth_roi_masked.convertTo(image_depth, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
        const int n_pixels = ellipses.get_ellipse_pixels(BodyPart::HEAD, center_depth);
        const int n_pixels2 = cv::countNonZero(mask);
        std::cout <<"Z " << state.average_z << " " << center_depth << " " << non_zero << ' ' << n_pixels<< ' ' << n_pixels2 << ' ' << n_pixels2 - n_pixels<<std::endl;
        std::cout << state.region << std::endl;
        CImage model_frame_image;
        model_frame_image.loadFromIplImage(new IplImage(image_depth));
        image2.showImage(model_frame_image);
    }
    */

    state.color_model = compute_color_model2(hsv_roi, mask_weights);

    particles.set_head_color_model(state.color_model);
    particles.set_shape_model(shape_model);
    //particles.last_distance = state.average_z;
    particles.last_distance = state.z;

    //CHEST
    const cv::Mat torso_mask_weights = ellipses.get_ellipse_mask_weights(BodyPart::TORSO, center_depth);
    Eigen::Vector2i torso_center = translate_2D_vector_in_3D_space(center.x, center.y, center_depth, HEAD_TO_TORSE_CENTER_VECTOR,
                                                                reg.cameraMatrixColor, reg.lookupX, reg.lookupY);

    const cv::Rect torso_rect = cv::Rect(cvRound(torso_center[0] - torso_mask_weights.cols * 0.5f),
                                         cvRound(torso_center[1] - torso_mask_weights.rows * 0.5f),
                                         torso_mask_weights.cols, torso_mask_weights.rows);
    
    const cv::Mat torso_roi = hsv_frame(torso_rect);
    state.torso_color_model = compute_color_model2(torso_roi, torso_mask_weights);
    particles.set_torso_color_model(state.torso_color_model);
    
    particles.init_particles(NUM_PARTICLES, make_pair(state.x, state.radius_x), make_pair(state.y, state.radius_y),
                            make_pair(float(state.z), 100.f),
                            make_pair(0, 0), make_pair(0, 0), make_pair(0, 0));
    return true;
};



template <typename DEPTH_TYPE>
void do_tracking(CParticleFilter &PF, CImageParticleFilter<DEPTH_TYPE> &particles,
                 const CSensoryFrame &observation, CParticleFilter::TParticleFilterStats &stats)
{
    PF.executeOn(particles, NULL, &observation, &stats);
}

template <typename DEPTH_TYPE>
void build_state_model(const CImageParticleFilter<DEPTH_TYPE> &particles,
                        const StateEstimation &old_state, StateEstimation &new_state,
                        const cv::Mat &hsv_frame, const cv::Mat &depth_frame,
                        EllipseStash &ellipses)
{
    particles.get_mean(new_state.x, new_state.y, new_state.z, new_state.v_x, new_state.v_y, new_state.v_z);
    Eigen::Vector2i top_corner, bottom_corner;
    const double center_measured_depth = depth_frame.at<DEPTH_TYPE>(cvRound(new_state.y),
                                         cvRound(new_state.x));
    //TODO USE MEAN DEPTH OF THE ELLIPSE
    const double z = center_measured_depth > 0 ? center_measured_depth : old_state.z;


    const cv::Size projection_size = ellipses.get_ellipse_size(BodyPart::HEAD, z);
    new_state.radius_x = projection_size.width * 0.5;
    new_state.radius_y = projection_size.height * 0.5;
    new_state.center = cv::Point(new_state.x, new_state.y);
    new_state.region = cv::Rect(new_state.x - new_state.radius_x, new_state.y - new_state.radius_y,
        projection_size.width, projection_size.height);


    if (!rect_fits_in_frame(new_state.region, hsv_frame)){
        return;
    }

    const cv::Mat mask_weights = ellipses.get_ellipse_mask_weights(BodyPart::HEAD, z);
    cv::Mat hsv_roi = hsv_frame(new_state.region);
    new_state.color_model = compute_color_model2(hsv_roi, mask_weights);


}

void score_visual_model(const StateEstimation &state, StateEstimation &new_state, const cv::Mat &gradient_vectors,
                        const std::vector<Eigen::Vector2f> &shape_model)
{
    if (new_state.color_model.empty()) {
        new_state.score_total = -1;
        new_state.score_color = -1;
        new_state.score_shape = -1;
        return;
    }

    new_state.score_color = 1 - cv::compareHist(new_state.color_model, state.color_model,
                                            CV_COMP_BHATTACHARYYA);
    new_state.score_shape = ellipse_contour_test(new_state.center, new_state.radius_x, new_state.radius_y,
                        shape_model, gradient_vectors, cv::Mat(), nullptr);
    /*
    new_state.torso_color_score = 1 - cv::compareHist(new_state.torso_color_model, state.torso_color_model,
                                            CV_COMP_BHATTACHARYYA);
    */
    
    //TODO UPDATE WITH DEPTH AND TORSO?
    new_state.score_total = new_state.score_color * new_state.score_shape;
}
