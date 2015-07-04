#pragma once

#include "project_config.h"

IGNORE_WARNINGS_PUSH

#include <mrpt/otherlibs/do_opencv_includes.h>

IGNORE_WARNINGS_POP

#include "misc_helpers.h"
using namespace Eigen;

cv::Mat create_ellipse_mask(const cv::Point &center, const int axis_x, const int axis_y, const int n_dims);

cv::Mat create_ellipse_weight_mask(const cv::Mat &ellipse_mask,
                                   std::function<float(int, int)> w_function);

inline cv::Mat create_ellipse_mask(const cv::Rect &rectangle, const int n_dims);

inline bool point_within_ellipse(const cv::Point &point, const cv::Point &center,
                                 const float squared_radi_x_inv, const float squared_radi_y_inv);

inline Vector2f calculate_ellipse_normal(const int axis_x, const int axis_y, const float angle);

inline Vector2f calculate_ellipse_orthonormal(const int axis_x, const int axis_y, const float angle);

inline bool point_within_ellipse(const cv::Point &point, const cv::Point &center,
                                 const float squared_radi_x_inv, const float squared_radi_y_inv)
{
    const int center_point_vector_x = point.x - center.x;
    const int center_point_vector_y = point.y - center.y;
    return ((center_point_vector_x * center_point_vector_x) * squared_radi_x_inv +
            (center_point_vector_y * center_point_vector_y) * squared_radi_y_inv) <= 1.0;
}

inline cv::Mat create_ellipse_mask(const cv::Rect &rectangle, const int n_dims)
{
    return create_ellipse_mask(cv::Point(rectangle.width / 2.f, rectangle.height / 2.f),
                               rectangle.width, rectangle.height, n_dims);
}

cv::Mat create_ellipse_mask(const cv::Point &center, const int axis_x, const int axis_y,
                            const int n_dims)
{
    cv::Mat mask = cv::Mat::zeros(axis_y, axis_x, CV_8UC1);
    const int rows = mask.rows;
    const int cols = mask.cols;
    const int n_rows_half = rows >> 1;
    const int n_cols_half = cols >> 1;
    const float radi_x = axis_x / 2.0;
    const float radi_y = axis_y / 2.0;
    const float squared_radi_x_inv = 1.0f / (radi_x * radi_x);
    const float squared_radi_y_inv = 1.0f / (radi_y * radi_y);
    const int cols_minus_1 = cols - 1;
    const int rows_minus_1 = rows - 1;
    //TODO USE_INTEL_TBB?
    for (int i = 0; i < n_rows_half; i++) {
        //uchar* mask_row_upper = mask.ptr<uchar>(i);
        //uchar* mask_row_lower = mask.ptr<uchar>(rows_minus_1 - i);
        for (int j = 0; j < n_cols_half; j++) {
            const uchar point_within = point_within_ellipse(cv::Point(j, i), center,
                                       squared_radi_x_inv, squared_radi_y_inv) ? 1 : 0;
            const int right_side_j = cols_minus_1 - j;
            const int lower_half_i = rows_minus_1 - i;
            if (point_within) {
                cv::line(mask, cv::Point(j, i), cv::Point(right_side_j, i), 1);
                cv::line(mask, cv::Point(j, lower_half_i), cv::Point(right_side_j, lower_half_i), 1);
            }
        }
    }

    if (n_dims == 1) {
        return mask;
    }

    std::vector<cv::Mat> mask_channels(n_dims, mask);
    cv::Mat mask_ndims;
    cv::merge(mask_channels, mask_ndims);
    return mask_ndims;
}

cv::Mat fast_create_ellipse_mask(int xc, int yc, const int aa, const int bb, const int n_dims,
                                 int &n_pixels)
{
    //e(x,y) = b^2*x^2 + a^2*y^2 - a^2*b^2

    /*
    from: http://sydney.edu.au/engineering/it/research/tr/tr531.pdf
    from: http://enchantia.com/graphapp/doc/tech/ellipses.html
    @book{patrick2001drawing,
      title={Drawing Ellipses Using Filled Rectangles},
      author={Patrick, Lachlan J},
      year={2001},
      publisher={Basser Department of Computer Science, University of Sydney}
    }
    */
    xc--;
    yc--;
#define incx() x++, dxt += d2xt, t += dxt
#define incy() y--, dyt += d2yt, t += dyt
    cv::Mat mask = cv::Mat::zeros(bb, aa, CV_8UC1);
    const int a = (aa / 2.0f);
    const int b = (bb / 2.0f);
    int x = 0, y = b;
    unsigned int width = 1;
    long a2 = (long)a * a, b2 = (long)b * b;
    long crit1 = -(a2 / 4.0f + a % 2 + b2);
    long crit2 = -(b2 / 4.0f + b % 2 + a2);
    long crit3 = -(b2 / 4.0f + b % 2);
    // e(x+1/2,y-1/2) - (a^2+b^2)/4
    long t = -a2 * y;
    long dxt = 2 * b2 * x, dyt = -2 * a2 * y;
    long d2xt = 2 * b2, d2yt = 2 * a2;
    const auto value = cv::Scalar::all(1);
    n_pixels = 0;
    while (y >= 0 && x <= a) {
        if (t + b2 * x <= crit1 || t + a2 * y <= crit3) {
            //e(x+1,y-1/2) <= 0
            //e(x+1/2,y) <= 0
            incx();
            width += 2;
        } else if (t - a2 * y > crit2) {
            //e(x+1/2,y-1) > 0
            //row(xc-x, yc-y, width);
            cv::line(mask, cv::Point(xc - x, yc - y), cv::Point(xc - x + width, yc - y), value);
            n_pixels += width + 1;
            if (y != 0) {
                //row(xc-x, yc+y, width);
                cv::line(mask, cv::Point(xc - x, yc + y), cv::Point(xc - x + width, yc + y), value);
                n_pixels += width + 1;
            }
            incy();
        } else {
            //row(xc-x, yc-y, width);
            cv::line(mask, cv::Point(xc - x, yc - y), cv::Point(xc - x + width, yc - y), value);
            n_pixels += width + 1;
            if (y != 0) {
                //row(xc-x, yc+y, width);
                cv::line(mask, cv::Point(xc - x, yc + y), cv::Point(xc - x + width, yc + y), value);
                n_pixels += width + 1;
            }
            incx();
            incy();
            width += 2;
        }
    }
    if (b == 0) {
        //row(xc-a, yc, 2*a+1);
        cv::line(mask, cv::Point(xc - a, yc), cv::Point(xc - a + 2 * a + 1, yc), value);
        n_pixels += 2 * a + 1 + 1;
    }

    if (n_dims == 1) {
        return mask;
    }

    std::vector<cv::Mat> mask_channels(n_dims, mask);
    cv::Mat mask_ndims;
    cv::merge(mask_channels, mask_ndims);
    return mask_ndims;
}

inline cv::Mat fast_create_ellipse_mask(const cv::Rect &rectangle, const int n_dims,
                                        int &n_pixels)
{
    return fast_create_ellipse_mask(cvRound(rectangle.width * 0.5f),
                                    cvRound(rectangle.height * 0.5f),
                                    rectangle.width, rectangle.height, n_dims, n_pixels);
}

cv::Mat create_ellipse_weight_mask(const cv::Mat &ellipse_mask)
{
    const int rows = ellipse_mask.rows;
    const int cols = ellipse_mask.cols;
    cv::Mat weight_mask;
    weight_mask.create(rows, cols, CV_32FC1);
    const int radi_x = cols >> 1;
    const int radi_y = rows >> 1;
    const int cols_minus_1 = cols - 1;
    //const float inv_max_distance_squared = 1.0f / std::max(radi_x * radi_x, radi_y * radi_y);
    const float inv_max_distance = Q_rsqrt(std::max(radi_x * radi_x, radi_y * radi_y));
    float sum_r = 0;
    for (int i = 0; i < radi_y; i++) {
        float *weight_mask_row_upper = weight_mask.ptr<float>(i);
        float *weight_mask_row_lower = weight_mask.ptr<float>(rows - 1 - i);
        const int d_y = std::abs(radi_y - i);
        for (int j = 0; j < radi_x; j++) {
            const int d_x = std::abs(radi_x - j);
            const uchar point_within = ellipse_mask.at<uchar>(i, j);
            const float r = std::sqrt(d_x * d_x + d_y * d_y) * inv_max_distance;
            const float w = r < 1.0 ? (1 - r * r) : 0;
            sum_r += w;
            const float distance_weight = point_within * w;
            weight_mask_row_upper[j] = distance_weight;
            weight_mask_row_lower[j] = distance_weight;
            const int right_side_j = cols_minus_1 - j;
            weight_mask_row_upper[right_side_j] = distance_weight;
            weight_mask_row_lower[right_side_j] = distance_weight;
        }
    }

    return weight_mask;// / sum_r;
}

inline Vector2f calculate_ellipse_point(const cv::Point &center, const float axis_x,
                                        const float axis_y, const float angle)
{
    return Vector2f(axis_x * cos(angle) + center.x, axis_y * sin(angle) + center.y);
}

inline Vector2f calculate_ellipse_normal(const float axis_x, const float axis_y,
        const float angle)
{
    return Vector2f(axis_x * cos(angle), axis_y * sin(angle));
}

inline Vector2f calculate_ellipse_orthonormal(const float axis_x, const float axis_y,
        const float angle)
{
    Vector2f normal = calculate_ellipse_normal(axis_x, axis_y, angle);
    normal.normalize();
    return normal;
}

std::vector<Vector2f> calculate_ellipse_normals(const float radius_x,
        const float radius_y, const int angle_step)
{
    const int total_steps = 360 / angle_step;
    std::vector<Vector2f> normal_vectors(total_steps / 4);
    for (int i = 0, j = 0; i < 90; i += angle_step, j++) {
        normal_vectors[j] = calculate_ellipse_normal(radius_x, radius_y, M_PI * i / 180.0);
        const float inv_modulus = Q_rsqrt(normal_vectors[j][0] * normal_vectors[j][0] +
                                          normal_vectors[j][1] * normal_vectors[j][1]);
        normal_vectors[j] *= inv_modulus;
        //std::cout << normal_vectors[j][0] << ' ' << normal_vectors[j][1] << std::endl;
        //normal_vectors[j].normalize();
    }
    return normal_vectors;
}

// This function can perform both, gradient magnitude aware and  purely gradient direction evaluation; the second
// is made by passing an empty matrix as gradient_magnitude parameter.
float ellipse_contour_test(const cv::Point &center, const float radius_x, const float radius_y,
                           const std::vector<Vector2f> &normal_vectors,
                           const cv::Mat &gradient_vectors, const cv::Mat &gradient_magnitude,
                           cv::Mat * const output = nullptr)
{
    const int total_vectors = normal_vectors.size();
    float dot_sum = 0;
    for (int i = 0; i < total_vectors; i++) {
        const Vector2f &v = normal_vectors[i];
        const float v_x = v[0];
        const float v_y = v[1];
        const cv::Vec2f v_1(v_x, v_y);
        const cv::Vec2f v_2(-v_x, -v_y);
        const cv::Vec2f v_3(-v_y, v_x);
        const cv::Vec2f v_4(v_y, -v_x);

        const int pixel_coordinates_1_x = center.x + cvRound(v_1[0] * radius_x);
        const int pixel_coordinates_1_y = center.y + cvRound(v_1[1] * radius_y);
        const int pixel_coordinates_2_x = center.x + cvRound(v_2[0] * radius_x);
        const int pixel_coordinates_2_y = center.y + cvRound(v_2[1] * radius_y);
        const int pixel_coordinates_3_x = center.x + cvRound(v_3[0] * radius_x);
        const int pixel_coordinates_3_y = center.y + cvRound(v_3[1] * radius_y);
        const int pixel_coordinates_4_x = center.x + cvRound(v_4[0] * radius_x);
        const int pixel_coordinates_4_y = center.y + cvRound(v_4[1] * radius_y);
        /*
        std::cout << "CENTER " << center.x << ' ' << center.y << ' ' << radius_x << ' ' << radius_y << std::endl;
        std::cout << pixel_coordinates_1_x << ' ' << pixel_coordinates_1_y << std::endl;
        std::cout << pixel_coordinates_2_x << ' ' << pixel_coordinates_2_y << std::endl;
        std::cout << pixel_coordinates_3_x << ' ' << pixel_coordinates_3_y << std::endl;
        std::cout << pixel_coordinates_4_x << ' ' << pixel_coordinates_4_y << std::endl;
        */
        const cv::Vec2f &gradient_v1 = gradient_vectors.at<cv::Vec2f>(pixel_coordinates_1_y,
                                       pixel_coordinates_1_x);
        const cv::Vec2f &gradient_v2 = gradient_vectors.at<cv::Vec2f>(pixel_coordinates_2_y,
                                       pixel_coordinates_2_x);
        const cv::Vec2f &gradient_v3 = gradient_vectors.at<cv::Vec2f>(pixel_coordinates_3_y,
                                       pixel_coordinates_3_x);
        const cv::Vec2f &gradient_v4 = gradient_vectors.at<cv::Vec2f>(pixel_coordinates_4_y,
                                       pixel_coordinates_4_x);

        float magnitude_v1;
        float magnitude_v2;
        float magnitude_v3;
        float magnitude_v4;

        if (!gradient_magnitude.empty()) {
            magnitude_v1 = gradient_magnitude.at<float>(pixel_coordinates_1_y, pixel_coordinates_1_x);
            magnitude_v2 = gradient_magnitude.at<float>(pixel_coordinates_2_y, pixel_coordinates_2_x);
            magnitude_v3 = gradient_magnitude.at<float>(pixel_coordinates_3_y, pixel_coordinates_3_x);
            magnitude_v4 = gradient_magnitude.at<float>(pixel_coordinates_4_y, pixel_coordinates_4_x);
        } else {
            magnitude_v1 = 1;
            magnitude_v2 = 1;
            magnitude_v3 = 1;
            magnitude_v4 = 1;
        }

        const float dot_1 = magnitude_v1 * std::abs(gradient_v1[0] * v_1[0] + gradient_v1[1] * v_1[1]);
        const float dot_2 = magnitude_v2 * std::abs(gradient_v2[0] * v_2[0] + gradient_v2[1] * v_2[1]);
        const float dot_3 = magnitude_v3 * std::abs(gradient_v3[0] * v_3[0] + gradient_v3[1] * v_3[1]);
        const float dot_4 = magnitude_v4 * std::abs(gradient_v4[0] * v_4[0] + gradient_v4[1] * v_4[1]);
        /*
                const float dot_1 = std::abs(gradient_v1[0] * v_1[0] + gradient_v1[1] * v_1[1]);
                const float dot_2 = std::abs(gradient_v2[0] * v_2[0] + gradient_v2[1] * v_2[1]);
                const float dot_3 = std::abs(gradient_v3[0] * v_3[0] + gradient_v3[1] * v_3[1]);
                const float dot_4 = std::abs(gradient_v4[0] * v_4[0] + gradient_v4[1] * v_4[1]);
        */
        dot_sum += dot_1 + dot_2 + dot_3 + dot_4;

        if (unlikely(output != nullptr)) {
            const cv::Point pixel_coordinates_1 = center + cv::Point(cvRound(v_1[0] * radius_x),
                                                  cvRound(v_1[1] * radius_y));
            const cv::Point pixel_coordinates_2 = center + cv::Point(cvRound(v_2[0] * radius_x),
                                                  cvRound(v_2[1] * radius_y));
            const cv::Point pixel_coordinates_3 = center + cv::Point(cvRound(v_3[0] * radius_x),
                                                  cvRound(v_3[1] * radius_y));
            const cv::Point pixel_coordinates_4 = center + cv::Point(cvRound(v_4[0] * radius_x),
                                                  cvRound(v_4[1] * radius_y));

            const cv::Point end_model_v1(pixel_coordinates_1 + cv::Point(cvRound(v_1[0] * 10),
                                         cvRound(v_1[1] * 10)));
            const cv::Point end_model_v2(pixel_coordinates_2 + cv::Point(cvRound(v_2[0] * 10),
                                         cvRound(v_2[1] * 10)));
            const cv::Point end_model_v3(pixel_coordinates_3 + cv::Point(cvRound(v_3[0] * 10),
                                         cvRound(v_3[1] * 10)));
            const cv::Point end_model_v4(pixel_coordinates_4 + cv::Point(cvRound(v_4[0] * 10),
                                         cvRound(v_4[1] * 10)));

            cv::arrowedLine(*output, pixel_coordinates_1, end_model_v1, cv::Scalar(255, 0, 0), 1, 8, 0);
            cv::arrowedLine(*output, pixel_coordinates_2, end_model_v2, cv::Scalar(255, 0, 0), 1, 8, 0);
            cv::arrowedLine(*output, pixel_coordinates_3, end_model_v3, cv::Scalar(255, 0, 0), 1, 8, 0);
            cv::arrowedLine(*output, pixel_coordinates_4, end_model_v4, cv::Scalar(255, 0, 0), 1, 8, 0);

            const cv::Point end_1(pixel_coordinates_1 + cv::Point(cvRound(gradient_v1[0] * 10),
                                  cvRound(gradient_v1[1] * 10)));
            const cv::Point end_2(pixel_coordinates_2 + cv::Point(cvRound(gradient_v2[0] * 10),
                                  cvRound(gradient_v2[1] * 10)));
            const cv::Point end_3(pixel_coordinates_3 + cv::Point(cvRound(gradient_v3[0] * 10),
                                  cvRound(gradient_v3[1] * 10)));
            const cv::Point end_4(pixel_coordinates_4 + cv::Point(cvRound(gradient_v4[0] * 10),
                                  cvRound(gradient_v4[1] * 10)));

            cv::arrowedLine(*output, pixel_coordinates_1, end_1, cv::Scalar(180, 180, 180), 1, 8, 0);
            cv::arrowedLine(*output, pixel_coordinates_2, end_2, cv::Scalar(180, 180, 180), 1, 8, 0);
            cv::arrowedLine(*output, pixel_coordinates_3, end_3, cv::Scalar(180, 180, 180), 1, 8, 0);
            cv::arrowedLine(*output, pixel_coordinates_4, end_4, cv::Scalar(180, 180, 180), 1, 8, 0);
        }
    }

    return dot_sum / (total_vectors * 4);
}

