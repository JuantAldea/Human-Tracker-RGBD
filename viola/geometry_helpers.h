#pragma once

#include <mrpt/otherlibs/do_opencv_includes.h>
#include <vector>

cv::Mat create_ellipse_mask(const cv::Point &center, const int axis_x, const int axis_y, const int ndims);
cv::Mat create_ellipse_mask(const cv::Rect &rectangle, const int ndims);
inline bool point_within_ellipse(const cv::Point &point, const cv::Point &center, const int radi_x, const int radi_y);

inline bool point_within_ellipse(const cv::Point &point, const cv::Point &center, const int radi_x, const int radi_y)
{
    return (((point.x - center.x) * (point.x - center.x)) / float((radi_x * radi_x)) + ((point.y - center.y) * (point.y - center.y)) / float((radi_y * radi_y))) <= 1;
}

cv::Mat create_ellipse_mask(const cv::Rect &rectangle, const int ndims)
{
    return create_ellipse_mask(cv::Point(rectangle.width / 2, rectangle.height / 2), rectangle.width, rectangle.height, ndims);
}

cv::Mat create_ellipse_mask(const cv::Point &center, const int axis_x, const int axis_y, const int ndims)
{
    cv::Mat mask;
    mask.create(axis_y, axis_x, CV_8UC1);
    const int channels = mask.channels();
    const int nRows = mask.rows;
    const int nCols = mask.cols * channels;

    const int radi_x = cvRound(axis_x / 2.0);
    const int radi_y = cvRound(axis_y / 2.0);

    for (int i = 0; i < nRows; i++) {
        uchar* mask_row = mask.ptr<uchar>(i);
        for (int j = 0; j < nCols; j++) {
            mask_row[j] = point_within_ellipse(cv::Point(j, i), center, radi_x, radi_y) ? 0xff : 0x0;
        }
    }

    std::vector<cv::Mat> mask_channels(ndims);
    for (int i = 0; i < ndims; i++) {
        mask_channels[i] = mask;
    }

    cv::Mat mask_ndims;
    cv::merge(mask_channels, mask_ndims);
    return mask_ndims;
}
