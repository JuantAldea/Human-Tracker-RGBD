#pragma once

#include <vector>

#include "project_config.h"

IGNORE_WARNINGS_PUSH

#include <mrpt/otherlibs/do_opencv_includes.h>

IGNORE_WARNINGS_POP


cv::Mat create_ellipse_mask(const cv::Point &center, const int axis_x, const int axis_y, const int ndims);

cv::Mat create_ellipse_mask(const cv::Rect &rectangle, const int ndims);

inline bool point_within_ellipse(const cv::Point &point, const cv::Point &center, const int radi_x, const int radi_y);

////////////////////

inline Eigen::Vector3f point_3D_reprojection(const float x, const float y, const float depth, const float inv_fx, const float inv_fy, const float cx, const float cy);
inline Eigen::Vector3f point_3D_reprojection(const Eigen::Vector2f &v, const float depth, const float inv_fx, const float inv_fy, const float cx, const float cy);

template<typename DEPTH_DATA_TYPE>
std::vector<Eigen::Vector3f> points_3D_reprojection(const std::vector<Eigen::Vector2f> &points, const cv::Mat &depth, const float inv_fx, const float inv_fy, const float cx, const float cy);

template<typename DEPTH_DATA_TYPE>
cv::Mat depth_3D_reprojection(const cv::Mat &depth, const float inv_fx, const float inv_fy, const float cx, const float cy);

template<typename DEPTH_DATA_TYPE>
inline std::tuple<Eigen::Vector2i, Eigen::Vector2i> project_model(const Eigen::Vector2f &model_center, const cv::Mat &depth, const Eigen::Vector2f &model_semi_axis_lengths,
    const cv::Mat &cameraMatrix, const cv::Mat &lookupX, const cv::Mat &lookupY);

std::tuple<Eigen::Vector2i, Eigen::Vector2i> project_model(const Eigen::Vector2f &model_center, const float depth, const Eigen::Vector2f &model_semi_axis_lengths,
    const cv::Mat &cameraMatrix, const cv::Mat &lookupX, const cv::Mat &lookupY);

/////////////////////////////////
/////////////////////////////////
/////////////////////////////////

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

inline Eigen::Vector3f point_3D_reprojection(const float x, const float y, const float depth, const float inv_fx, const float inv_fy, const float cx, const float cy)
{
    return Eigen::Vector3f((x - cx) * depth * inv_fx, (y - cy) * depth * inv_fy, depth);
}

inline Eigen::Vector3f point_3D_reprojection(const Eigen::Vector2f &v, const float depth, const float inv_fx, const float inv_fy, const float cx, const float cy)
{
    return point_3D_reprojection(v[0], v[1], depth, inv_fx, inv_fy, cx, cy);
}

template<typename DEPTH_TYPE>
std::vector<Eigen::Vector3f> points_3D_reprojection(const std::vector<Eigen::Vector2f> &points, const cv::Mat &depth, const float inv_fx, const float inv_fy, const float cx, const float cy)
{
    const size_t N = points.size();
    std::vector<Eigen::Vector3f> reprojected_points(N);
#ifdef USE_INTEL_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, N / TBB_PARTITIONS),
        [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); i++) {
                reprojected_points[i] = point_3D_reprojection(points[i], depth.at<DEPTH_TYPE>(points[i][1], points[i][0]), inv_fx, inv_fy, cx, cy);
            }
        }
    );
#else
    for (size_t i = 0; i < N; i++){
        reprojected_points[i] = point_3D_reprojection(points[i], depth.at<DEPTH_TYPE>(points[i][1], points[i][0]), inv_fx, inv_fy, cx, cy);
    }
#endif
    return reprojected_points;
}

template<typename DEPTH_TYPE>
cv::Mat depth_3D_reprojection(const cv::Mat &depth, const float inv_fx, const float inv_fy, const float cx, const float cy)
{
    cv::Mat reprojection = cv::Mat(depth.size(), depth.type());
#ifdef USE_INTEL_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, depth.rows, depth.rows / TBB_PARTITIONS),
        [&](const tbb::blocked_range<int> &r){
            for (int x = r.begin(); x < r.end(); x++) {
                const DEPTH_TYPE *p_depth = depth.ptr<DEPTH_TYPE>(x);
                cv::Vec3f *p_reprojection = reprojection.ptr<cv::Vec3f>(x);
                for (int y = 0; y < depth.cols; y++) {
                    cv::Vec3f &v = p_reprojection[y];
                    v[2] = p_depth[y] / 1000.f;
                    v[1] = ((y - cy) * v[2]) * inv_fy;
                    v[0] = ((x - cx) * v[2]) * inv_fx;
                }
            }
        }
    );
#else
    for (int x = 0; x < depth.rows; x++) {
        const DEPTH_TYPE *p_depth = depth.ptr<DEPTH_TYPE>(x);
        cv::Vec3f *p_reprojection = reprojection.ptr<cv::Vec3f>(x);
        for (int y = 0; y < depth.cols; y++) {
            cv::Vec3f &v = p_reprojection[y];
            v[2] = p_depth[y] / 1000.f;
            v[1] = ((y - params.cy) * v[2]) * inv_fy;
            v[0] = ((x - params.cx) * v[2]) * inv_fx;
        }
    }
#endif
    return reprojection;
}


  ////////////////////////////////
 // Reprojection with mappings //
////////////////////////////////

inline Eigen::Vector3f point_3D_reprojection (const Eigen::Vector2f &v, const float depth, const cv::Mat &lookupX, const cv::Mat &lookupY)
{
    static const float badPoint = std::numeric_limits<float>::quiet_NaN();
    const float x = lookupX.at<float>(0, v[0]);
    const float y = lookupY.at<float>(0, v[1]);
    register const float depthValue = depth / 1000.0f;

    // Check for invalid measurements
    if (isnan(depthValue) || depthValue <= 0.001) {
        return Eigen::Vector3f(badPoint, badPoint, badPoint);
    }
    
    const float z_coord = depthValue;
    const float x_coord = x * depthValue;
    const float y_coord = y * depthValue;

    return Eigen::Vector3f(x_coord, y_coord, z_coord);
}

template<typename DEPTH_TYPE>
inline Eigen::Vector3f point_3D_reprojection(const Eigen::Vector2f &v, const cv::Mat &depth, const cv::Mat &lookupX, const cv::Mat &lookupY)
{
    return point_3D_reprojection(v, depth.at<DEPTH_TYPE>(v[1], v[0]), lookupX, lookupY);
}

template<typename DEPTH_TYPE>
std::vector<Eigen::Vector3f> points_3D_reprojection(const std::vector<Eigen::Vector2f> &vectors, const cv::Mat &depth, const cv::Mat &lookupX, const cv::Mat &lookupY)
{
    const size_t N = vectors.size();
    std::vector<Eigen::Vector3f> reprojected_points(N);
#ifdef USE_INTEL_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, N, N / TBB_PARTITIONS),
        [&](const tbb::blocked_range<int> &r){
            for (int i = r.begin(); i < r.end(); i++) {
                reprojected_points[i] = point_3D_reprojection<DEPTH_TYPE>(vectors[i], depth, lookupX, lookupY);
            }
        }
    );
#else
     for (int i = 0; i < N; i++) {
        reprojected_points[i] = point_3D_reprojection<DEPTH_TYPE>(vectors[i], depth, lookupX, lookupY);
    }
#endif    
    return reprojected_points;
}

inline Eigen::Vector2i point_3D_projection(const Eigen::Vector3f &v, const float fx, const float fy, const float cx, const float cy)
{
    const float inv_z = 1.0 / v[2];
    return Eigen::Vector2i(cvRound(fx * v[0] * inv_z + cx), cvRound(fy * v[1] * inv_z + cy));
}

template<typename DEPTH_TYPE>
inline std::tuple<Eigen::Vector2i, Eigen::Vector2i> project_model(const Eigen::Vector2f &model_center, const cv::Mat &depth, const Eigen::Vector2f &model_semi_axis_lengths,
    const cv::Mat &cameraMatrix, const cv::Mat &lookupX, const cv::Mat &lookupY)
{
    return project_model(model_center, depth.at<DEPTH_TYPE>(model_center[1], model_center[0]), model_semi_axis_lengths, cameraMatrix, lookupX, lookupY);
}

std::tuple<Eigen::Vector2i, Eigen::Vector2i> project_model(const Eigen::Vector2f &model_center, const float depth, const Eigen::Vector2f &model_semi_axis_lengths,
    const cv::Mat &cameraMatrix, const cv::Mat &lookupX, const cv::Mat &lookupY)
{
    const Eigen::Vector3f circle_center_3d = point_3D_reprojection(Eigen::Vector2f(model_center[0], model_center[1]), depth, lookupX, lookupY);
    const Eigen::Vector3f circle_left_top_corner_3d = circle_center_3d + Eigen::Vector3f(-model_semi_axis_lengths[0], -model_semi_axis_lengths[1], 0);
    const Eigen::Vector3f circle_left_bottom_corner_3d = circle_center_3d + Eigen::Vector3f(model_semi_axis_lengths[0], model_semi_axis_lengths[1], 0);

    const float fx = cameraMatrix.at<double>(0, 0);
    const float fy = cameraMatrix.at<double>(1, 1);
    const float cx = cameraMatrix.at<double>(0, 2);
    const float cy = cameraMatrix.at<double>(1, 2);
    
    const Eigen::Vector2i top_corner = point_3D_projection(circle_left_top_corner_3d, fx, fy, cx, cy);
    const Eigen::Vector2i bottom_corner = point_3D_projection(circle_left_bottom_corner_3d, fx, fy, cx, cy);
    
    //cout << "circle " << center_x << ' ' << center_y << " radius2d " << radius_2d << " radius 3d " <<  (x_radius + y_radius) * 0.5 << " radius x "<< x_radius << " radius y " << y_radius << std::endl;
    return std::make_tuple(top_corner, bottom_corner);
}
