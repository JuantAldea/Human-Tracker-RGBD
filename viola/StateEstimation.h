#pragma once

struct StateEstimation
{


    float x;
    float y;
    float z;
    float v_x;
    float v_y;
    float v_z;

    float average_z;

    int radius_x;
    int radius_y;
    cv::Point center;
    cv::Rect region;
    cv::Mat color_model;
    float factor = 1;
    float score_color;
    float score_shape;
    float score_total;
    float score_z;

    cv::Mat torso_color_model;
    float torso_color_score;

    StateEstimation():
        x(-1), y(-1), z(-1), v_x(-1), v_y(-1), v_z(-1), average_z(-1),
        radius_x(-1), radius_y(-1),
        score_color(-1), score_shape(-1), score_total(-1), torso_color_score(-1)
    {
        ;
    };

    void print(void) const
    {
        std::printf("Estate: (%f %f %f) -> (%d %d)\n", x, y, z, radius_x, radius_y);
        std::cout << center << std::endl;
        std::cout << region << std::endl;
    }

    void blend(const StateEstimation &o)
    {
        cv::Mat blended_color_model(color_model.rows, color_model.cols, color_model.type());
        cv::Mat blended_torso_color_model(torso_color_model.rows, torso_color_model.cols, torso_color_model.type());
        
        cv::addWeighted(color_model, 1 - o.score_total, o.color_model, o.score_total, 0, blended_color_model);

        if(!o.torso_color_model.empty()){
            cv::addWeighted(torso_color_model, 1 - o.score_total, o.torso_color_model, o.score_total, 0, blended_torso_color_model);
        } else {
            blended_torso_color_model = torso_color_model;
        }

        *this = o;
        color_model = blended_color_model;
        torso_color_model = blended_torso_color_model;
    }
};
