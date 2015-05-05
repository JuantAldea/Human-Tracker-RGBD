#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Werror"
#pragma GCC diagnostic ignored "-Wlong-long"
#pragma GCC diagnostic ignored "-pedantic"
#pragma GCC diagnostic ignored "-pedantic-errors"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <mrpt/otherlibs/do_opencv_includes.h>

#pragma GCC diagnostic pop

cv::Mat compute_color_model(const cv::Mat &hsv, const cv::Mat &mask);
cv::Mat sobel_operator(const cv::Mat &image);

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

cv::Mat sobel_operator(const cv::Mat &image)
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
    std::vector<cv::Mat> gradients = {grad_x_float, grad_y_float};

    cv::Mat gradient_vectors;
    cv::merge(gradients, gradient_vectors);

    cv::Mat gradient_modulus_scaled;
    double min, max;
    cv::minMaxLoc(gradient_modulus, &min, &max);
    cv::convertScaleAbs(gradient_modulus, gradient_modulus_scaled, 255/max);

    //cv::ellipse2Poly(cv::Mat(), cv::Point(), cv::Size(), 360, 0, 0, cv::Scalar(255, 0, 0), 1, 0);
    return gradient_modulus_scaled;
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