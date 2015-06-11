#pragma once

#include "project_config.h"

IGNORE_WARNINGS_PUSH

#include <mrpt/otherlibs/do_opencv_includes.h>

IGNORE_WARNINGS_POP

cv::Mat compute_color_model(const cv::Mat &hsv, const cv::Mat &mask);
cv::Mat histogram_to_image(const cv::Mat &histogram, const int scale);
std::tuple<cv::Mat, cv::Mat, cv::Mat> sobel_operator(const cv::Mat &image);

cv::Mat compute_color_model(const cv::Mat &hsv, const cv::Mat &mask)
{
    // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    cv::Mat histogram;
    const int hbins = 31;
    const int sbins = 32;

    {
        // hue varies from 0 to 179, it's scaled down by a half
        const int histSize[] = {hbins, sbins};
        // so that it fits in a byte.
        const float hranges[] = {0, 180};
        // saturation varies from 0 (black-gray-white) to
        // 255 (pure spectrum color)
        const float sranges[] = {0, 256};
        const float* ranges[] = {hranges, sranges};
        const int channels[] = {0, 1};
        cv::calcHist(&hsv, 1, channels, mask, histogram, 2, histSize, ranges, true, false);
    }

    cv::Mat histogram_v;
    {
        const int channels[] = {2};
        const int histSize[] = {sbins};
        const float range[] = {0, 256} ;
        const float* histRange = {range};
        cv::calcHist(&hsv, 1, channels, mask, histogram_v, 1, histSize, &histRange, true, false);
    }

    histogram_v = histogram_v.t();
    histogram.push_back(histogram_v);

    double sum = 0;
    for (int h = 0; h < histogram.rows; h++) {
        for (int s = 0; s < histogram.cols; s++) {
            sum += histogram.at<float>(h, s);
        }
    }

    for (int h = 0; h < histogram.rows; h++) {
        for (int s = 0; s < histogram.cols; s++) {
            histogram.at<float>(h, s) /= sum;
        }
    }

    return histogram;
}

cv::Mat histogram_to_image(const cv::Mat &histogram, const int scale)
{
    cv::Mat histImg = cv::Mat::zeros(histogram.rows * scale, histogram.cols * scale, CV_8UC1);
    double maxVal = 0;
    cv::minMaxLoc(histogram, 0, &maxVal, 0, 0);
    for (int row = 0; row < histogram.rows; row++) {
        for (int col = 0; col < histogram.cols; col++) {
            const float binVal = histogram.at<float>(row, col);
            const int intensity = cvRound(255 * (binVal / maxVal));
            cv::rectangle(histImg, cv::Point(row * scale, col * scale),
                          cv::Point((row + 1) * scale - 1, (col + 1) * scale - 1),
                          cv::Scalar::all(intensity), CV_FILLED);
        }
    }
    return histImg;
}


std::tuple<cv::Mat, cv::Mat, cv::Mat> sobel_operator(const cv::Mat &image)
{

    cv::Mat orig = image.clone();
    cv::GaussianBlur(orig, orig, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT);
    cv::Mat image_gray;
    if (image.channels() > 1){
        cvtColor(orig, image_gray, CV_RGB2GRAY);
    }else{
        image_gray = orig.clone();
    }

    cv::Mat grad_x, grad_y;
    cv::Mat squared_grad_x, squared_grad_y;
    cv::Mat gradient_modulus;
    cv::Mat image_gray_float;
    
    image_gray.convertTo(image_gray_float, CV_32F);
    cv::Sobel(image_gray_float, grad_x, CV_16S, 1, 0, 7, 1, 0, cv::BORDER_DEFAULT);
    cv::Sobel(image_gray_float, grad_y, CV_16S, 0, 1, 7, 1, 0, cv::BORDER_DEFAULT);

    cv::Mat grad_x_float, grad_y_float;
    grad_x.convertTo(grad_x_float, CV_32F);
    grad_y.convertTo(grad_y_float, CV_32F);
    cv::pow(grad_x_float, 2.f, squared_grad_x);
    cv::pow(grad_y_float, 2.f, squared_grad_y);
    cv::sqrt(squared_grad_x + squared_grad_y, gradient_modulus);

    grad_x_float /= gradient_modulus;
    grad_y_float /= gradient_modulus;
    //std::vector<cv::Mat> gradients = {grad_x_float, grad_y_float};

    cv::Mat gradient_vectors;
    cv::merge(std::vector<cv::Mat> {grad_x_float, grad_y_float}, gradient_vectors);

    cv::Mat gradient_modulus_scaled;
    double min, max;
    cv::minMaxLoc(gradient_modulus, &min, &max);
    cv::convertScaleAbs(gradient_modulus, gradient_modulus_scaled, 255/max);

    //cv::ellipse2Poly(cv::Mat(), cv::Point(), cv::Size(), 360, 0, 0, cv::Scalar(255, 0, 0), 1, 0);
    return std::make_tuple(gradient_vectors, gradient_modulus, gradient_modulus_scaled);
}
/*
std::vector<cv::Point> ellipse2Poly()
{
    int XY_SHIFT = 16;
    int XY_ONE = 1 << XY_SHIFT;

    axes.width = std::abs(axes.width), axes.height = std::abs(axes.height);
    int delta = (std::max(axes.width, axes.height) + (XY_ONE >> 1)) >> XY_SHIFT;
    delta = delta < 3 ? 90 : delta < 10 ? 30 : delta < 15 ? 18 : 5;
    std::vector<cv::Point> v;
    ellipse2Poly( center, axes, angle, arc_start, arc_end, delta, v );
    return v;
}
*/

float ellipse_shape_gradient_test(const cv::Point &center, const float radius_x, const float radius_y, const int angle_step, const cv::Mat &gradient_vectors, const cv::Mat &gradient_magnitude, cv::Mat *output = nullptr)
{
    float dot_sum = 0;
    
    const float total_steps = 360.0 / angle_step;
    
    for (int i = 0; i < 90; i += angle_step){
        Eigen::Vector2f v = calculate_ellipse_normal(radius_x, radius_y, M_PI * i / total_steps);
        v.normalize();

        const cv::Vec2f v_1(v[0], v[1]);
        const cv::Vec2f v_2(-v[0], -v[1]);
        const cv::Vec2f v_3(-v[1], v[0]);
        const cv::Vec2f v_4(v[1], -v[0]);

        const cv::Point pixel_coordinates_1 = center + cv::Point(cvRound(v_1[0] * radius_x), cvRound(v_1[1] * radius_y));
        const cv::Point pixel_coordinates_2 = center + cv::Point(cvRound(v_2[0] * radius_x), cvRound(v_2[1] * radius_y));
        const cv::Point pixel_coordinates_3 = center + cv::Point(cvRound(v_3[0] * radius_x), cvRound(v_3[1] * radius_y));
        const cv::Point pixel_coordinates_4 = center + cv::Point(cvRound(v_4[0] * radius_x), cvRound(v_4[1] * radius_y));

        const cv::Vec2f &gradient_v1 = gradient_vectors.at<cv::Vec2f>(pixel_coordinates_1.y, pixel_coordinates_1.x);
        const cv::Vec2f &gradient_v2 = gradient_vectors.at<cv::Vec2f>(pixel_coordinates_2.y, pixel_coordinates_2.x);
        const cv::Vec2f &gradient_v3 = gradient_vectors.at<cv::Vec2f>(pixel_coordinates_3.y, pixel_coordinates_3.x);
        const cv::Vec2f &gradient_v4 = gradient_vectors.at<cv::Vec2f>(pixel_coordinates_4.y, pixel_coordinates_4.x);
        
        const float magnitude_v1 = gradient_magnitude.at<float>(pixel_coordinates_1.y, pixel_coordinates_1.x);
        const float magnitude_v2 = gradient_magnitude.at<float>(pixel_coordinates_2.y, pixel_coordinates_2.x);
        const float magnitude_v3 = gradient_magnitude.at<float>(pixel_coordinates_3.y, pixel_coordinates_3.x);
        const float magnitude_v4 = gradient_magnitude.at<float>(pixel_coordinates_4.y, pixel_coordinates_4.x);
        
        
        const float dot_1 = magnitude_v1 * std::abs(gradient_v1[0] * v_1[0] + gradient_v1[1] * v_1[1]);
        const float dot_2 = magnitude_v2 * std::abs(gradient_v2[0] * v_2[0] + gradient_v2[1] * v_2[1]);
        const float dot_3 = magnitude_v3 * std::abs(gradient_v3[0] * v_3[0] + gradient_v3[1] * v_3[1]);
        const float dot_4 = magnitude_v4 * std::abs(gradient_v4[0] * v_4[0] + gradient_v4[1] * v_4[1]);
/*
        const float dot_1 = std::abs(gradient_v1[0] * v_1[0] + gradient_v1[1] * v_1[1]);
        const float dot_2 = std::abs(gradient_v2[0] * v_2[0] + gradient_v2[1] * v_2[1]);
        const float dot_3 = std::abs(gradient_v3[0] * v_3[0] + gradient_v3[1] * v_3[1]);
        const float dot_4 = std::abs(gradient_v4[0] * v_4[0] + gradient_v4[1] * v_4[1]);
*/    
        dot_sum += dot_1 + dot_2 + dot_3 + dot_4;
        
        
        if (output != nullptr){
            const cv::Point end_model_v1(pixel_coordinates_1 + cv::Point(cvRound(v_1[0] * 10), cvRound(v_1[1] * 10)));
            const cv::Point end_model_v2(pixel_coordinates_2 + cv::Point(cvRound(v_2[0] * 10), cvRound(v_2[1] * 10)));
            const cv::Point end_model_v3(pixel_coordinates_3 + cv::Point(cvRound(v_3[0] * 10), cvRound(v_3[1] * 10)));
            const cv::Point end_model_v4(pixel_coordinates_4 + cv::Point(cvRound(v_4[0] * 10), cvRound(v_4[1] * 10)));

            cv::arrowedLine(*output, pixel_coordinates_1, end_model_v1, cv::Scalar(0, 0, 255), 1, 8, 0);
            cv::arrowedLine(*output, pixel_coordinates_2, end_model_v2, cv::Scalar(0, 0, 255), 1, 8, 0);
            cv::arrowedLine(*output, pixel_coordinates_3, end_model_v3, cv::Scalar(0, 0, 255), 1, 8, 0);
            cv::arrowedLine(*output, pixel_coordinates_4, end_model_v4, cv::Scalar(0, 0, 255), 1, 8, 0);

            const cv::Point end_1(pixel_coordinates_1 + cv::Point(cvRound(gradient_v1[0] * 10), cvRound(gradient_v1[1] * 10)));
            const cv::Point end_2(pixel_coordinates_2 + cv::Point(cvRound(gradient_v2[0] * 10), cvRound(gradient_v2[1] * 10)));
            const cv::Point end_3(pixel_coordinates_3 + cv::Point(cvRound(gradient_v3[0] * 10), cvRound(gradient_v3[1] * 10)));
            const cv::Point end_4(pixel_coordinates_4 + cv::Point(cvRound(gradient_v4[0] * 10), cvRound(gradient_v4[1] * 10)));
            
            cv::arrowedLine(*output, pixel_coordinates_1, end_1, cv::Scalar(180, 180, 180), 1, 8, 0);
            cv::arrowedLine(*output, pixel_coordinates_2, end_2, cv::Scalar(180, 180, 180), 1, 8, 0);
            cv::arrowedLine(*output, pixel_coordinates_3, end_3, cv::Scalar(180, 180, 180), 1, 8, 0);
            cv::arrowedLine(*output, pixel_coordinates_4, end_4, cv::Scalar(180, 180, 180), 1, 8, 0);
        }
    }

    return dot_sum / total_steps;
}
