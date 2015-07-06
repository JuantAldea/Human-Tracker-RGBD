#pragma once

#include "project_config.h"
#include "ellipse_functions.h"

IGNORE_WARNINGS_PUSH

#include <mrpt/otherlibs/do_opencv_includes.h>
#include <opencv2/ocl/ocl.hpp>
IGNORE_WARNINGS_POP
#include <cassert>
cv::Mat compute_color_model(const cv::Mat &hsv, const cv::Mat &mask);
cv::Mat histogram_to_image(const cv::Mat &histogram, const int scale);
std::tuple<cv::Mat, cv::Mat, cv::Mat> sobel_operator(const cv::Mat &image);

//cv::Mat calc_hist2D(const cv::Mat &image, const int channels[], const cv::Mat &mask, const cv::Mat &weights, const int ndims, const int hist_size[], const float * const ranges[])
cv::Mat calc_hist2D(const cv::Mat &image, const int channels[], const cv::Mat &weights, const int ndims, const int hist_size[], const float * const ranges[])
{
    int hist_rows = hist_size[0];
    int hist_cols = 1;
    if (ndims > 1){
        hist_cols = hist_size[1];
    }
    cv::Mat hist = cv::Mat::zeros(hist_rows, hist_cols, CV_32FC1);
    float *hist_data = hist.ptr<float>(0);
    float bin_widths[ndims];
    for (int c = 0; c < ndims; c++){
        bin_widths[c] = (ranges[c][1] - ranges[c][0]) / hist_size[c];
    }

    cv::Mat_<uchar> image2 = image;
    const int img_channels = image.channels();
    for (int i = 0; i < image.rows; i++){
        const uchar *p_row = image.ptr<uchar>(i);
        //const uchar *mask_row = mask.ptr<uchar>(i);
        const float *mask_row = weights.ptr<float>(i);
        for (int j = 0; j < image.cols; j++){
            if (mask_row[j]){
                uint bins[ndims];
                for (int c = 0; c < ndims; c++){
                    const uchar value = p_row[j * img_channels + channels[c]];
                    bins[c] = value / bin_widths[c];
                }

                //const cv::Vec3b hsv = image.at<cv::Vec3b>(i, j);
                //const uint bin_h = hsv[0] / bin_width_1;
                //const uint bin_s = hsv[1] / bin_width_2;
                //hist.at<float>(bin_h, bin_s) += 1;
                //std::cout << i << ' ' << j << ' ' << sqrt(pow(i - image.rows, 2 ) + pow(j - image.cols, 2)) << ' ' << weights.at<float>(i, j) << std::endl;

                //general case would be
                // array_index = dot( exclusive_prefix_product(dimx, dimy, ..., dimn), (x, y, z) ). e.g: dot((1, dimx, dimx*dimy), (x, y, z))
                switch (ndims){
                    case 1:
                        hist_data[bins[0]] += weights.at<float>(i, j);
                    break;
                    default:
                        hist_data[bins[0]*hist_cols + bins[1]] += weights.at<float>(i, j);
                        //hist.at<float>(bins[0], bins[1]) += 1;//mask.at<float>(i, j);
                    break;
                }

                //float *p = hist.ptr<float>(0);
                /*
                for (int c = 0; c < ndims - 1; c++){
                    p += hist_size[c] * bins[c];
                }
                p += hist_size[0] * bins[0];
                p += bins[1];
                float *a = &hist.at<float>(bins[0], bins[1]);
                std::cout <<"p " << p << std::endl;
                std::cout << "a "  << a << std::endl;
                assert(p == a);
                *p += 1;
                //make
                //p[bins[ndims - 1]] += 1;
                */


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

    cv::Scalar sum = cv::sum(histogram);
    histogram /= sum[0];
    return histogram;
}

cv::Mat compute_color_model2(const cv::Mat &hsv, const cv::Mat &weights)
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
        histogram = calc_hist2D(hsv, channels, weights, 2, histSize, ranges);
    }

    cv::Mat histogram_v;
    {
        const int channels[] = {2};
        const int histSize[] = {sbins};
        const float range[] = {0, 256} ;
        const float *ranges[] = {range};
        //cv::calcHist(&hsv, 1, channels, mask, histogram_v, 1, histSize, &histRange, true, false);
        histogram_v = calc_hist2D(hsv, channels, weights, 1, histSize, ranges);
    }

    histogram_v = histogram_v.t();
    histogram.push_back(histogram_v);

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
#define OPENCL_OCL
#ifdef OPENCL_OCL
    using TYPE_MAT = cv::ocl::oclMat;
    namespace TYPE_OP = cv::ocl;
#else
    using TYPE_MAT = cv::Mat;
    namespace TYPE_OP = cv;
#endif
    
    int64_t t0 = cv::getTickCount();

    TYPE_MAT orig(image);
    TYPE_MAT image_gray;
    
    int64_t t1 = cv::getTickCount();
    
    if (image.channels() > 1){
        TYPE_OP::cvtColor(orig, image_gray, CV_RGB2GRAY);
    }else{
        image_gray = orig;
    }
    int64_t t2 = cv::getTickCount();

//#define USE_BILINEAR
#ifdef USE_BILINEAR
    TYPE_MAT image_gray_aux;
    TYPE_OP::bilateralFilter(image_gray, image_gray_aux, 15, 80, 80);
    image_gray = image_gray_aux;
#endif

    TYPE_MAT image_gray_float;
    image_gray.convertTo(image_gray_float, CV_32F);
    
    int64_t t3 = cv::getTickCount();
    
    TYPE_MAT grad_x, grad_y;
    TYPE_OP::Sobel(image_gray_float, grad_x, CV_32F, 1, 0, 7, 1, 0, cv::BORDER_DEFAULT);
    TYPE_OP::Sobel(image_gray_float, grad_y, CV_32F, 0, 1, 7, 1, 0, cv::BORDER_DEFAULT);
    
    int64_t t4 = cv::getTickCount();
    
    TYPE_MAT gradient_modulus;
    TYPE_OP::magnitude(grad_x, grad_y, gradient_modulus);
    
    int64_t t5 = cv::getTickCount();
    
    grad_x /= gradient_modulus;
    grad_y /= gradient_modulus;

    int64_t t6 = cv::getTickCount();

    TYPE_MAT gradient_modulus_copy = gradient_modulus.clone();
    TYPE_OP::threshold(gradient_modulus_copy, gradient_modulus, 1000, 0, cv::THRESH_TOZERO);
    
    int64_t t7 = cv::getTickCount();
    
    double min, max;
    TYPE_OP::minMaxLoc(gradient_modulus, &min, &max);
    TYPE_MAT gradient_modulus_scaled;
    gradient_modulus.convertTo(gradient_modulus_scaled, CV_8UC1, 255 / max);

    int64_t t8 = cv::getTickCount();

    TYPE_MAT gradient_vectors;
    TYPE_OP::merge(std::vector<TYPE_MAT> {grad_x, grad_y}, gradient_vectors);
    
    int64_t t9 = cv::getTickCount();


    //going back to CPU.
    cv::Mat cpu_gradient_vectors = gradient_vectors;
    cv::Mat cpu_gradient_modulus = gradient_modulus;
    cv::Mat cpu_gradient_modulus_scaled = gradient_modulus_scaled;
    
    int64_t t10 = cv::getTickCount();
    
    const float inv_freq = 1.0/cv::getTickFrequency();
    const double t1_0 = (t1 - t0) * inv_freq;
    const double t2_1 = (t2 - t1) * inv_freq;
    const double t3_2 = (t3 - t2) * inv_freq;
    const double t4_3 = (t4 - t3) * inv_freq;
    const double t5_4 = (t5 - t4) * inv_freq;
    const double t6_5 = (t6 - t5) * inv_freq;
    const double t7_6 = (t7 - t6) * inv_freq;
    const double t8_7 = (t8 - t7) * inv_freq;
    const double t9_8 = (t9 - t8) * inv_freq;
    const double t10_9 = (t10 - t9) * inv_freq;

    const double t10_0 = (t10 - t0) * inv_freq;
    
    std::cout << "TIMES_SOBEL " << t10_0 << ',' << t1_0 << ',' << t2_1 << ',' << t3_2 << ',' << t4_3 << ',' << t5_4 << ',' << t6_5 << ',' << t7_6 << ',' << t8_7 << ',' << t9_8 << ',' << t10_9 << std::endl;

    
    return std::make_tuple(cpu_gradient_vectors, cpu_gradient_modulus, cpu_gradient_modulus_scaled);
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

