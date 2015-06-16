#pragma once

#include "project_config.h"
#include "ellipse_functions.h"

IGNORE_WARNINGS_PUSH

#include <mrpt/otherlibs/do_opencv_includes.h>

IGNORE_WARNINGS_POP

cv::Mat compute_color_model(const cv::Mat &hsv, const cv::Mat &mask);
cv::Mat histogram_to_image(const cv::Mat &histogram, const int scale);
std::tuple<cv::Mat, cv::Mat, cv::Mat> sobel_operator(const cv::Mat &image);

cv::Mat calc_hist2D(const cv::Mat &image, const int channels[], const cv::Mat &mask, const cv::Mat &weights, const int ndims, const int hist_size[], const float * const ranges[])
{
    cv::Mat hist = cv::Mat::zeros(hist_size[0], hist_size[1], CV_32FC1);
    
    float bin_widths[ndims];
    for (int c = 0; c < ndims; c++){
        bin_widths[c] = (ranges[c][1] - ranges[c][0]) / hist_size[c];
    }
    //const float bin_width_1 = (ranges[0][1] - ranges[0][0]) / hist_size[0];
    //const float bin_width_2 = (ranges[1][1] - ranges[1][0]) / hist_size[1];

    const int img_channels = image.channels();
    for (int i = 0; i < image.rows; i++){
        for (int j = 0; j < image.cols; j++){
            if (mask.at<uchar>(i, j)){
                uint bins[ndims];
                for (int c = 0; c < ndims; c++){
                    const uchar value = image.at<uchar>(i, j * img_channels + channels[c]);
                    bins[c] = value / bin_widths[c];
                    //const uint bin_h = hsv[0] / bin_width_1;
                    //const uint bin_s = hsv[1] / bin_width_2;
                }
                //const cv::Vec3b hsv = image.at<cv::Vec3b>(i, j);
                //const uint bin_h = hsv[0] / bin_width_1;
                //const uint bin_s = hsv[1] / bin_width_2;
                //hist.at<float>(bin_h, bin_s) += 1;
                hist.at<float>(bins[0], bins[1]) += 1;
                float *p = hist.ptr<float>(0);
                /*
                for (int c = 0; c < ndims - 1; c++){
                    p += hist_size[c] * bins[c];
                }
                */
                p += hist_size[0] * bins[0];
                p += hist_size[1] * bins[1];
                *p += 1;
                //make
                p[bins[ndims - 1]] += 1;
                

            }
        }
    }
    return hist;
}

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
    /*
    cv::Mat histogram_v;
    {
        const int channels[] = {2};
        const int histSize[] = {sbins};
        const float range[] = {0, 256} ;
        const float *ranges[] = {range};
        cv::calcHist(&hsv, 1, channels, mask, histogram_v, 1, histSize, ranges, true, false);
    }

    histogram_v = histogram_v.t();
    histogram.push_back(histogram_v);
    
    */
    cv::Scalar sum = cv::sum(histogram);
    histogram /= sum[0];
    
    return histogram;
}

cv::Mat compute_color_model2(const cv::Mat &hsv, const cv::Mat &mask)
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
        //cv::calcHist(&hsv, 1, channels, mask, histogram, 2, histSize, ranges, true, false);
        histogram = calc_hist2D(hsv, channels, mask, cv::Mat(), 2, histSize, ranges);
    }
    /*
    cv::Mat histogram_v;
    {
        const int channels[] = {2};
        const int histSize[] = {sbins};
        const float range[] = {0, 256} ;
        const float *ranges[] = {range};
        //cv::calcHist(&hsv, 1, channels, mask, histogram_v, 1, histSize, &histRange, true, false);
        histogram = calc_hist2D(hsv, channels, mask, cv::Mat(), 1, histSize, ranges);
    }

    histogram_v = histogram_v.t();
    histogram.push_back(histogram_v);
    */
    cv::Scalar sum = cv::sum(histogram);
    histogram /= sum[0];
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
    //cv::GaussianBlur(orig, orig, cv::Size(25, 25), 0, 0, cv::BORDER_DEFAULT);
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
    
    //cv::Mat image_gray2;
    //bilateralFilter ( image_gray, image_gray2, 15, 80, 80 );
    //image_gray = image_gray2;
    
    image_gray.convertTo(image_gray_float, CV_32F);
    cv::Sobel(image_gray_float, grad_x, CV_32F, 1, 0, 7, 1, 0, cv::BORDER_DEFAULT);
    cv::Sobel(image_gray_float, grad_y, CV_32F, 0, 1, 7, 1, 0, cv::BORDER_DEFAULT);
    
    //cv::medianBlur ( grad_x, grad_x, 5 );
    //cv::medianBlur ( grad_y, grad_y, 5 );



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

