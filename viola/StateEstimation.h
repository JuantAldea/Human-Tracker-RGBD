#pragma once

struct StateEstimation
{
    float x_head;
    float y_head;
    float z_head;
    float v_x_head;
    float v_y_head;
    float v_z_head;
    int radius_x_head;
    int radius_y_head;
    cv::Point center_head;
    float score_shape_head;
    float score_total_head;
    cv::Rect region_head;
    cv::Mat color_model_head;

    float x;
    float y;
    float z;
    float v_x;
    float v_y;
    float v_z;

    int radius_x;
    int radius_y;
    cv::Point center;
    cv::Rect region;
    cv::Mat color_model;
    float factor = 1;
    float score_color;
    float score_shape;
    float score_total;

    StateEstimation():
        x(-1), y(-1), z(-1), v_x(-1), v_y(-1), v_z(-1), radius_x(-1), radius_y(-1), center(), region(), color_model(), score_color(-1), score_shape(-1), score_total(-1)
    {
        ;
    };

    void print(void)
    {
        std::printf("Estate: (%f %f %f) -> (%d %d)\n", x, y, z, radius_x, radius_y);
        std::cout << center << std::endl;
        std::cout << region << std::endl;
    }

};
