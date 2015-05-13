#pragma once

#include <vector>

#include "project_config.h"

IGNORE_WARNINGS_PUSH

#include <mrpt/otherlibs/do_opencv_includes.h>

IGNORE_WARNINGS_POP


cv::Mat create_ellipse_mask(const cv::Point &center, const int axis_x, const int axis_y, const int ndims);

cv::Mat create_ellipse_mask(const cv::Rect &rectangle, const int ndims);

inline bool point_within_ellipse(const cv::Point &point, const cv::Point &center, const int radi_x, const int radi_y);

inline Eigen::Vector3f point_3D_reprojection(const float x, const float y, const float Z, const float inv_fx, const float inv_fy, const float cx, const float cy);

inline Eigen::Vector3f  point_3D_reprojection(const Eigen::Vector2f &v, const float Z, const float inv_fx, const float inv_fy, const float cx, const float cy);

std::vector<Eigen::Vector3f> points_3D_reprojection(const std::vector<Eigen::Vector2f> &points, const cv::Mat &depth_data, const float inv_fx, const float inv_fy, const float cx, const float cy);

cv::Mat depth_3D_reprojection(const cv::Mat &depth_data, const float inv_fx, const float inv_fy, const float cx, const float cy);

inline bool point_within_ellipse(const cv::Point &point, const cv::Point &center, const int radi_x, const int radi_y)
{
    return (((point.x - center.x) * (point.x - center.x)) / float((radi_x * radi_x)) + ((point.y - center.y) * (point.y - center.y)) / float((radi_y * radi_y))) <= 1;
}

cv::Mat create_ellipse_mask(const cv::Rect &rectangle, const int ndims)
{
    return create_ellipse_mask(cv::Point(rectangle.width / 2.f, rectangle.height / 2.f), rectangle.width, rectangle.height, ndims);
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
    
    //TODO USE_INTEL_TBB?
    for (int i = 0; i < nRows; i++) {
        uchar* mask_row = mask.ptr<uchar>(i);
        for (int j = 0; j < nCols; j++) {
            mask_row[j] = point_within_ellipse(cv::Point(j, i), center, radi_x, radi_y) ? 0xff : 0x0;
        }
    }

    std::vector<cv::Mat> mask_channels(ndims, mask);
    cv::Mat mask_ndims;
    cv::merge(mask_channels, mask_ndims);

    return mask_ndims;
}

inline Eigen::Vector3f point_3D_reprojection(const float x, const float y, const float Z, const float inv_fx, const float inv_fy, const float cx, const float cy)
{
    return Eigen::Vector3f((x - cx) * Z * inv_fx, (y - cy) * Z * inv_fy, Z);
}

inline Eigen::Vector3f point_3D_reprojection(const Eigen::Vector2f &v, const float Z, const float inv_fx, const float inv_fy, const float cx, const float cy)
{
    return Eigen::Vector3f((v[0] - cx) * Z * inv_fx, (v[1] - cy) * Z * inv_fy, Z);
}

std::vector<Eigen::Vector3f> points_3D_reprojection(const std::vector<Eigen::Vector2f> &points, const cv::Mat &depth_data, const float inv_fx, const float inv_fy, const float cx, const float cy)
{
    const size_t N = points.size();
    std::vector<Eigen::Vector3f> reprojected_points(N);
#ifdef USE_INTEL_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, N / TBB_PARTITIONS),
        [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); i++) {
                //TODO depth_data.at<float>(points[i][1], points[i][0]) or the other way around?
                reprojected_points[i] = point_3D_reprojection(points[i], depth_data.at<float>(points[i][1], points[i][0]), inv_fx, inv_fy, cx, cy);
            }
        }
    );
#else
    for (size_t i = 0; i < N; i++){
        reprojected_points[i] = point_3D_reprojection(points[i], Z,  inv_fx, inv_fy, cx, cy);
    }
#endif
    return reprojected_points;
}


cv::Mat depth_3D_reprojection(const cv::Mat &depth_data, const float inv_fx, const float inv_fy, const float cx, const float cy)
{
    cv::Mat reprojection = cv::Mat(depth_data.size(), CV_32FC3);
#ifdef USE_INTEL_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, depth_data.rows, depth_data.rows / TBB_PARTITIONS),
        [&](const tbb::blocked_range<int> &r){
            for (int x = r.begin(); x < r.end(); x++) {
                const float *p_depth = depth_data.ptr<float>(x);
                cv::Vec3f *p_reprojection = reprojection.ptr<cv::Vec3f>(x);
                for (int y = 0; y < depth_data.cols; y++) {
                    cv::Vec3f &v = p_reprojection[y];
                    v[2] = p_depth[y] / 1000.f;
                    v[1] = ((y - cy) * v[2]) * inv_fy;
                    v[0] = ((x - cx) * v[2]) * inv_fx;
                }
            }
        }
    );
#else
    for (int x = 0; x < depth_data.rows; x++) {
        const float *p_depth = depth_data.ptr<float>(x);
        cv::Vec3f *p_reprojection = reprojection.ptr<cv::Vec3f>(x);
        for (int y = 0; y < depth_data.cols; y++) {
            cv::Vec3f &v = p_reprojection[y];
            v[2] = p_depth[y] / 1000.f;
            v[1] = ((y - params.cy) * v[2]) * inv_fy;
            v[0] = ((x - params.cx) * v[2]) * inv_fx;
        }
    }
#endif
    return reprojection;
}

