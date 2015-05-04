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


cv::Mat histogram_to_image(const cv::Mat &histogram, const int scale);

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
