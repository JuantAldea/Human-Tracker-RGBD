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

inline bool rect_fits_in_rect(const cv::Rect &smaller, const cv::Rect &bigger)
{

    if (smaller.x < 0 || smaller.y < 0){
        return false;
    }

    if (smaller.width <= 0 || smaller.height <= 0){
        return false;
    }

    if ((smaller.x + smaller.width) > (bigger.x + bigger.width)){
        return false;
    }

    if ((smaller.y + smaller.height) > (bigger.y + bigger.height)){
        return false;
    }

    return true;

    //const cv::Rect region_intersection = bigger & smaller;
    //const cv::Rect region_intersection2 = smaller & bigger;

    //return smaller.area() == region_intersection.area();
}

inline bool rect_fits_in_frame(const cv::Rect &r, const cv::Mat &f)
{
    const cv::Rect frame_region = cv::Rect(0, 0, f.cols, f.rows);
    return rect_fits_in_rect(r, frame_region);
}

inline cv::Rect clamp_rect_to_frame(const cv::Rect &r, const cv::Mat &f)
{
    const int x_0 = r.x;
    const int y_0 = r.y;

    const int x_1 = r.x + r.width - 1;
    const int y_1 = r.y + r.height - 1;

    const int clamped_x0 = std::max(0, std::min(f.cols - 1, x_0));
    const int clamped_y0 = std::max(0, std::min(f.rows - 1, y_0));

    const int clamped_x1 = std::max(0, std::min(f.cols - 1, x_1));
    const int clamped_y1 = std::max(0, std::min(f.rows - 1, y_1));
    return cv::Rect(clamped_x0, clamped_y0, clamped_x1 - clamped_x0, clamped_y1 - clamped_y0);
}

inline bool point_in_mat(const int x, const int y, const cv::Mat& mat)
{
    return (x >= 0) && (x < mat.cols) && (y >= 0) && (y < mat.rows);
}


int handle_OpenCV_error( int status, const char* func_name, const char* err_msg, const char* file_name, int line, void* userdata )
{
    std::cerr << func_name << ": " << err_msg << std::endl;
    throw;
    return 0;
}

cv::Mat histogram_to_image(const std::vector<double> &x, const std::vector<double> &hits)
{
    cv::Mat image = cv::Mat::zeros(1000, x.size(), CV_8U);
    for (size_t i = 0; i < x.size(); i++){
        std:: cout << i << ' ' <<  x[i] << ' ' << hits[i] / x.size()  << std::endl;
        int height = cvRound((hits[i] / x.size()) * 1000);
        const cv::Point base = cv::Point(i, 0);
        const cv::Point top = cv::Point(i, height);
        cv::line(image, base, top, cv::Scalar::all(255), 1);
    }
    return image;
}
