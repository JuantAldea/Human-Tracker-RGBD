#pragma once

#include <vector>

IGNORE_WARNINGS_PUSH

#include <mrpt/otherlibs/do_opencv_includes.h>

IGNORE_WARNINGS_POP

#include "project_config.h"

using namespace Eigen;

constexpr float bad_value_float = std::numeric_limits<float>::quiet_NaN();
constexpr int bad_value_int = std::numeric_limits<int>::quiet_NaN();


const Vector3f bad_point_3D_float = Vector3f(bad_value_float, bad_value_float, bad_value_float);
const Vector2i bad_point_2D_int = Vector2i(bad_value_int, bad_value_int);
const std::tuple<Vector2i, Vector2i> bad_point_pair_2D_int = std::make_tuple(bad_point_2D_int, bad_point_2D_int);

inline Vector3f point_3D_reprojection(const float x, const float y, const float depth,
    const float inv_fx, const float inv_fy, const float cx, const float cy);

inline Vector3f point_3D_reprojection(const Vector2f &v, const float depth,
    const float inv_fx, const float inv_fy, const float cx, const float cy);

template<typename DEPTH_DATA_TYPE>
std::vector<Vector3f> points_3D_reprojection(const std::vector<Vector2f> &points, const cv::Mat &depth,
    const float inv_fx, const float inv_fy, const float cx, const float cy);

template<typename DEPTH_DATA_TYPE>
cv::Mat depth_3D_reprojection(const cv::Mat &depth, const float inv_fx, const float inv_fy, const float cx, const float cy);

template<typename DEPTH_DATA_TYPE>
inline std::tuple<Vector2i, Vector2i> project_model(const Vector2f &model_center, const cv::Mat &depth, const Vector2f &model_semi_axes,
    const cv::Mat &cameraMatrix, const cv::Mat &lookupX, const cv::Mat &lookupY);

std::tuple<Vector2i, Vector2i> project_model(const Vector2f &model_center, const float depth, const Vector2f &model_semi_axes,
    const cv::Mat &cameraMatrix, const cv::Mat &lookupX, const cv::Mat &lookupY);

inline Vector3f pixel_depth_to_3D_coordiantes(const float x, const float y, const float depth,
    const double inv_fx, const double inv_fy, const float cx, const float cy)
{
    if (unlikely(std::abs(depth) < 0.001)){
        return bad_point_3D_float;
    }

    return Vector3f((x - cx) * depth * inv_fx, (y - cy) * depth * inv_fy, depth);
}

inline Vector3f pixel_depth_to_3D_coordiantes(const Vector2f &v, const float depth,
    const float inv_fx, const float inv_fy, const float cx, const float cy)
{
    if (unlikely(std::abs(depth) < 0.001)){
        return bad_point_3D_float;
    }

    return pixel_depth_to_3D_coordiantes(v[0], v[1], depth, inv_fx, inv_fy, cx, cy);
}

inline Vector3f pixel_depth_to_3D_coordiantes(const float x, const float y, const float z, const cv::Mat &cameraMatrix)
{
    const float inv_fx = 1.0f / cameraMatrix.at<double>(0, 0);
    const float inv_fy = 1.0f / cameraMatrix.at<double>(1, 1);
    const float cx = cameraMatrix.at<double>(0, 2);
    const float cy = cameraMatrix.at<double>(1, 2);
    return pixel_depth_to_3D_coordiantes(x, y, z, inv_fx, inv_fy, cx, cy);
}

inline Vector3f pixel_depth_to_3D_coordiantes(const Vector3f &v, const cv::Mat &cameraMatrix)
{
    const float inv_fx = 1.0f / cameraMatrix.at<double>(0, 0);
    const float inv_fy = 1.0f / cameraMatrix.at<double>(1, 1);
    const float cx = cameraMatrix.at<double>(0, 2);
    const float cy = cameraMatrix.at<double>(1, 2);
    return pixel_depth_to_3D_coordiantes(v[0], v[1], v[2], inv_fx, inv_fy, cx, cy);
}

inline std::vector<Vector3f> pixel_depth_to_3D_coordiantes(const std::vector<Vector3f> &xyd_vectors,
    const double inv_fx, const double inv_fy, const double cx, const double cy)
{
    const size_t N = xyd_vectors.size();
    std::vector<Vector3f> coordinates_3D(N);

    auto loop_body_lambda = [&](const int i){
        const Vector3f &v = xyd_vectors[i];
        coordinates_3D[i] = pixel_depth_to_3D_coordiantes(v[0], v[1], v[2], inv_fx, inv_fy, cx, cy);
    };

#ifdef USE_INTEL_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, N / TBB_PARTITIONS),
        [&](const tbb::blocked_range<size_t> &r){
            for (size_t i = r.begin(); i < r.end(); i++) {
                loop_body_lambda(i);
            }
        }
    );
#else
    for (size_t i = 0; i < N; i++) {
        loop_body_lambda(i);
    }
#endif
    return coordinates_3D;
}

inline std::vector<Vector3f> pixel_depth_to_3D_coordiantes(const std::vector<Vector3f> &xyd_vectors, const cv::Mat &cameraMatrix)
{
    const float inv_fx = 1.0f / cameraMatrix.at<double>(0, 0);
    const float inv_fy = 1.0f / cameraMatrix.at<double>(1, 1);
    const float cx = cameraMatrix.at<double>(0, 2);
    const float cy = cameraMatrix.at<double>(1, 2);
    return pixel_depth_to_3D_coordiantes(xyd_vectors, inv_fx, inv_fy, cx, cy);
}

template<typename DEPTH_TYPE>
cv::Mat depth_3D_reprojection(const cv::Mat &depth, const float inv_fx, const float inv_fy, const float cx, const float cy)
{
    cv::Mat reprojection = cv::Mat(depth.size(), depth.type());

    auto depth_3D_reprojection_lambda = [&](const int x){
        const DEPTH_TYPE *p_depth = depth.ptr<DEPTH_TYPE>(x);
        cv::Vec3f *p_reprojection = reprojection.ptr<cv::Vec3f>(x);
        for (int y = 0; y < depth.cols; y++) {
            cv::Vec3f &v = p_reprojection[y];
            v[2] = p_depth[y];
            v[1] = ((y - cy) * v[2]) * inv_fy;
            v[0] = ((x - cx) * v[2]) * inv_fx;
        }
    };

#ifdef USE_INTEL_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, depth.rows, depth.rows / TBB_PARTITIONS),
        [&](const tbb::blocked_range<int> &r){
            for (int x = r.begin(); x < r.end(); x++) {
                depth_3D_reprojection_lambda(x);
            }
        }
    );
#else
    for (int x = 0; x < depth.rows; x++) {
        depth_3D_reprojection_lambda(x);
    }
#endif
    return reprojection;
}


  ////////////////////////////////
 // Reprojection with mappings //
////////////////////////////////

//TODO float depth templated?
inline Vector3f point_3D_reprojection (const float p_x, const float p_y, const float depth, const cv::Mat &lookupX, const cv::Mat &lookupY)
{
    const float x = lookupX.at<float>(0, cvRound(p_x));
    const float y = lookupY.at<float>(0, cvRound(p_y));
    register const float depth_metters = depth / 1000.0f;

    // Check for invalid measurements
    if (unlikely(isnan(depth_metters) || depth_metters <= 0.001)) {
        return bad_point_3D_float;
    }

    const float z_coord = depth_metters;
    const float x_coord = x * depth_metters;
    const float y_coord = y * depth_metters;

    return Vector3f(x_coord, y_coord, z_coord);
}

inline Vector3f point_3D_reprojection (const Vector2f &v, const float depth, const cv::Mat &lookupX, const cv::Mat &lookupY)
{
    return point_3D_reprojection(v[0], v[1], depth, lookupX, lookupY);
}


template<typename DEPTH_TYPE>
inline Vector3f point_3D_reprojection(const Vector2f &v, const cv::Mat &depth, const cv::Mat &lookupX, const cv::Mat &lookupY)
{
    return point_3D_reprojection(v, depth.at<DEPTH_TYPE>(v[1], v[0]), lookupX, lookupY);
}

template<typename DEPTH_TYPE>
std::vector<Vector3f> points_3D_reprojection(const std::vector<Vector2f> &points, const cv::Mat &depth,
    const float inv_fx, const float inv_fy, const float cx, const float cy)
{
    const size_t N = points.size();
    std::vector<Vector3f> reprojected_points(N);
#ifdef USE_INTEL_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, N / TBB_PARTITIONS),
        [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); i++) {
                reprojected_points[i] = pixel_depth_to_3D_coordiantes(points[i],
                    depth.at<DEPTH_TYPE>(points[i][1], points[i][0]), inv_fx, inv_fy, cx, cy);
            }
        }
    );
#else
    for (size_t i = 0; i < N; i++){
        reprojected_points[i] = pixel_depth_to_3D_coordiantes(points[i],
            depth.at<DEPTH_TYPE>(points[i][1], points[i][0]), inv_fx, inv_fy, cx, cy);
    }
#endif
    return reprojected_points;
}


template<typename DEPTH_TYPE>
std::vector<Vector3f> points_3D_reprojection(const std::vector<Vector2f> &vectors, const cv::Mat &depth,
    const cv::Mat &lookupX, const cv::Mat &lookupY)
{
    const size_t N = vectors.size();
    std::vector<Vector3f> reprojected_points(N);
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


inline Vector2i point_3D_projection(const Vector3f &v, const float fx, const float fy, const float cx, const float cy)
{
    if (unlikely(std::abs(v[2]) < 0.001)){
        return bad_point_2D_int;
    }

    const float inv_z = 1.0 / v[2];
    return Vector2i(cvRound(fx * v[0] * inv_z + cx), cvRound(fy * v[1] * inv_z + cy));
}


template<typename DEPTH_TYPE>
inline std::tuple<Vector2i, Vector2i> project_model(const Vector2f &model_center, const cv::Mat &depth,
    const Vector2f &model_semi_axes, const cv::Mat &cameraMatrix, const cv::Mat &lookupX, const cv::Mat &lookupY)
{
    return project_model(model_center, depth.at<DEPTH_TYPE>(model_center[1], model_center[0]),
        model_semi_axes, cameraMatrix, lookupX, lookupY);
}


std::tuple<Vector2i, Vector2i>
project_model(const Vector2f &model_center, const float depth, const Vector2f &model_semi_axes,
    const cv::Mat &cameraMatrix, const cv::Mat &lookupX, const cv::Mat &lookupY)
{
    if (unlikely(std::abs(depth) < 0.001)){
        return bad_point_pair_2D_int;
    }

    const Vector3f model_center_3d = point_3D_reprojection(model_center, depth, lookupX, lookupY);
    const Vector3f model_left_top_corner_3d = model_center_3d+ Vector3f(-model_semi_axes[0], -model_semi_axes[1], 0);
    const Vector3f model_left_bottom_corner_3d = model_center_3d + Vector3f(model_semi_axes[0], model_semi_axes[1], 0);

    const float fx = cameraMatrix.at<double>(0, 0);
    const float fy = cameraMatrix.at<double>(1, 1);
    const float cx = cameraMatrix.at<double>(0, 2);
    const float cy = cameraMatrix.at<double>(1, 2);

    const Vector2i top_corner = point_3D_projection(model_left_top_corner_3d, fx, fy, cx, cy);
    const Vector2i bottom_corner = point_3D_projection(model_left_bottom_corner_3d, fx, fy, cx, cy);

    //std::cout << "model " << model_center_3d[0] << ' ' << model_center_3d[1] << std::endl;

    if (unlikely(top_corner[0] < 0 || top_corner[1] < 0 || bottom_corner[0] < 0 || bottom_corner[1] < 0)){
        return bad_point_pair_2D_int;
    }

    if (unlikely(top_corner[0] > bottom_corner[0] || top_corner[1] > bottom_corner[1])){
        return bad_point_pair_2D_int;
    }

    return std::make_tuple(top_corner, bottom_corner);
}

Vector2i project_vector(const Vector2f &origin, const float depth, const Vector3f &vector,
    const cv::Mat &cameraMatrix, const cv::Mat &lookupX, const cv::Mat &lookupY)
{
    if (unlikely(std::abs(depth) < 0.001)){
        return bad_point_2D_int;
    }

    const Vector3f center_3d = point_3D_reprojection(origin, depth, lookupX, lookupY);
    const Vector3f point_3d = center_3d + vector;

    const float fx = cameraMatrix.at<double>(0, 0);
    const float fy = cameraMatrix.at<double>(1, 1);
    const float cx = cameraMatrix.at<double>(0, 2);
    const float cy = cameraMatrix.at<double>(1, 2);

    const Vector2i projected_point = point_3D_projection(point_3d, fx, fy, cx, cy);

    return projected_point;
}

Vector2i translate_2D_vector_in_3D_space(const int x, const int y, const float depth, const Vector3f &translation,
    const float fx, const float fy, const float cx, const float cy, const cv::Mat &lookupX, const cv::Mat &lookupY)
{    
    const Eigen::Vector3f point_3D = point_3D_reprojection(x, y, depth, lookupX, lookupY);
    const Eigen::Vector3f point_3D_translated = point_3D + translation;
    const Eigen::Vector2i point_2D_translated = point_3D_projection(point_3D_translated, fx, fy, cx, cy);
    return point_2D_translated;
}

Vector2i translate_2D_vector_in_3D_space(const int x, const int y, const float depth, const Vector3f &translation,
    const cv::Mat &cameraMatrix, const cv::Mat &lookupX, const cv::Mat &lookupY)
{
    const float fx = cameraMatrix.at<double>(0, 0);
    const float fy = cameraMatrix.at<double>(1, 1);
    const float cx = cameraMatrix.at<double>(0, 2);
    const float cy = cameraMatrix.at<double>(1, 2);

    return translate_2D_vector_in_3D_space(x, y, depth, translation, fx, fy, cx, cy, lookupX, lookupY);
}
