#pragma once

#include <vector>

#include "project_config.h"

IGNORE_WARNINGS_PUSH

#include <mrpt/otherlibs/do_opencv_includes.h>

IGNORE_WARNINGS_POP



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


////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

inline Eigen::Vector3f pixel_depth_to_3D_coordiantes(const float x, const float y, const float depth, const double inv_fx, const double inv_fy, const float cx, const float cy)
{
    static const float badPoint = std::numeric_limits<float>::quiet_NaN();
    if (std::abs(depth) < 0.001){
        return Eigen::Vector3f(badPoint, badPoint, badPoint);
    }
    return Eigen::Vector3f((x - cx) * depth * inv_fx, (y - cy) * depth * inv_fy, depth);
}

inline Eigen::Vector3f pixel_depth_to_3D_coordiantes(const Eigen::Vector2f &v, const float depth, const float inv_fx, const float inv_fy, const float cx, const float cy)
{
    static const float badPoint = std::numeric_limits<float>::quiet_NaN();
    if (std::abs(depth) < 0.001){
        return Eigen::Vector3f(badPoint, badPoint, badPoint);
    }

    return pixel_depth_to_3D_coordiantes(v[0], v[1], depth, inv_fx, inv_fy, cx, cy);
}

inline Eigen::Vector3f pixel_depth_to_3D_coordiantes(const float x, const float y, const float z, const cv::Mat &cameraMatrix)
{
    const float inv_fx = 1.0f / cameraMatrix.at<double>(0, 0);
    const float inv_fy = 1.0f / cameraMatrix.at<double>(1, 1);
    const float cx = cameraMatrix.at<double>(0, 2);
    const float cy = cameraMatrix.at<double>(1, 2);
    return pixel_depth_to_3D_coordiantes(x, y, z, inv_fx, inv_fy, cx, cy);
}

inline Eigen::Vector3f pixel_depth_to_3D_coordiantes(const Eigen::Vector3f &v, const cv::Mat &cameraMatrix)
{
    const float inv_fx = 1.0f / cameraMatrix.at<double>(0, 0);
    const float inv_fy = 1.0f / cameraMatrix.at<double>(1, 1);
    const float cx = cameraMatrix.at<double>(0, 2);
    const float cy = cameraMatrix.at<double>(1, 2);
    return pixel_depth_to_3D_coordiantes(v[0], v[1], v[2], inv_fx, inv_fy, cx, cy);
}

std::vector<Eigen::Vector3f> pixel_depth_to_3D_coordiantes(const std::vector<Eigen::Vector3f> &xyd_vectors, const double inv_fx, const double inv_fy, const double cx, const double cy)
{
    const size_t N = xyd_vectors.size();
    std::vector<Eigen::Vector3f> coordinates_3D(N);
    //TODO TBB
    for (size_t i = 0; i < N; i++){
        const Eigen::Vector3f &v = xyd_vectors[i];
        coordinates_3D[i] = pixel_depth_to_3D_coordiantes(v[0], v[1], v[2], inv_fx, inv_fy, cx, cy);
    }

    return coordinates_3D;
}

inline std::vector<Eigen::Vector3f> pixel_depth_to_3D_coordiantes(const std::vector<Eigen::Vector3f> &xyd_vectors, const cv::Mat &cameraMatrix)
{
    const float inv_fx = 1.0f / cameraMatrix.at<double>(0, 0);
    const float inv_fy = 1.0f / cameraMatrix.at<double>(1, 1);
    const float cx = cameraMatrix.at<double>(0, 2);
    const float cy = cameraMatrix.at<double>(1, 2);
    return pixel_depth_to_3D_coordiantes(xyd_vectors, inv_fx, inv_fy, cx, cy);
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
                reprojected_points[i] = pixel_depth_to_3D_coordiantes(points[i], depth.at<DEPTH_TYPE>(points[i][1], points[i][0]), inv_fx, inv_fy, cx, cy);
            }
        }
    );
#else
    for (size_t i = 0; i < N; i++){
        reprojected_points[i] = pixel_depth_to_3D_coordiantes(points[i], depth.at<DEPTH_TYPE>(points[i][1], points[i][0]), inv_fx, inv_fy, cx, cy);
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
                    v[2] = p_depth[y];
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
            v[2] = p_depth[y];
            v[1] = ((y - cy) * v[2]) * inv_fy;
            v[0] = ((x - cx) * v[2]) * inv_fx;
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
    static const int badPoint = std::numeric_limits<int>::quiet_NaN();
    if (std::abs(v[2]) < 0.001){
        return Eigen::Vector2i(badPoint, badPoint);
    }
 
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
    static const int badPoint = std::numeric_limits<int>::quiet_NaN();
    if (std::abs(depth) < 0.001){
        auto nan_v = Eigen::Vector2i(badPoint, badPoint);
        return std::make_tuple(nan_v, nan_v);
    }

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
    
    if (top_corner[0] < 0 || top_corner[1] < 0 || bottom_corner[0] < 0 || bottom_corner[1] < 0){
        auto nan_v = Eigen::Vector2i(badPoint, badPoint);
        return std::make_tuple(nan_v, nan_v);
    }

    if (top_corner[0] > bottom_corner[0] || top_corner[1] > bottom_corner[1]){
        auto nan_v = Eigen::Vector2i(badPoint, badPoint);
        return std::make_tuple(nan_v, nan_v);
    }
    
    return std::make_tuple(top_corner, bottom_corner);
}

