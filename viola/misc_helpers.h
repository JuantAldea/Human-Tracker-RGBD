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
#include <string>

std::string type2str(int type)
{
    std::string r;

    const uchar depth = type & CV_MAT_DEPTH_MASK;
    const uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

// the magical Quake fast inverse sqrt, A.K.A 0x5f3759df
float Q_rsqrt(const float number)
{
    const float threehalfs = 1.5F;
    const float x2  = number * 0.5F;
    float y;
    long i;

    y = number;
    i  = *(long *)&y;
    i = 0x5f3759df - (i >> 1);
    y = *(float *)&i;
    y = y * (threehalfs - (x2 * y * y));
    y = y * (threehalfs - (x2 * y * y));

    return y;
}

#pragma GCC diagnostic pop


static const cv::Scalar GlobalColorPalette[] = {
    cv::Scalar(  0,   0, 255),
    cv::Scalar(255,   0,   0),
    cv::Scalar(  0, 255,   0),
    cv::Scalar(255, 255,   0),
    cv::Scalar(  0, 255, 255),
    cv::Scalar(255,   0, 255),
    cv::Scalar(  0,   0, 128),
    cv::Scalar(128,   0,   0),
    cv::Scalar(  0, 128,   0),
    cv::Scalar(128, 128,   0),
    cv::Scalar(  0, 128, 128),
    cv::Scalar(128,   0, 128),
    cv::Scalar(  0,   0,   0),
    cv::Scalar(255, 255, 255),
    cv::Scalar(128, 128, 128),
    cv::Scalar(160, 164, 160),
    cv::Scalar(192, 192, 192)
};
/*

enum class GlobalColorNames  {
    RED = 0,
    GREEN,
    BLUE,
    CYAN,
    MAGENTA,
    YELLOW,
    DARK_RED,
    DARK_GREEN,
    DARK_BLUE,
    DARK_CYAN,
    DARK_MAGENTA,
    DARK_YELLOW,
    BLACK,
    WHITE,
    MEDIUM_GRAY,
    LIGHT_GRAY_1,
    LIGHT_GRAY_2,
};
*/
