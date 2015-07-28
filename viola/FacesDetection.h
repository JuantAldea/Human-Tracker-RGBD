#pragma once

#include <vector>

#include "project_config.h"

IGNORE_WARNINGS_PUSH
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <mrpt/otherlibs/do_opencv_includes.h>
#include <opencv2/ocl/ocl.hpp>

IGNORE_WARNINGS_POP

namespace viola_faces
{

using namespace cv;

typedef cv::Rect face_shape;
typedef cv::Rect eye;
typedef std::vector<eye> eyes;
typedef std::pair<face_shape, eyes> face;
typedef std::vector<face> faces;


faces detect_faces(const Mat &frame, CascadeClassifier &face_cascade, CascadeClassifier &eyes_cascade, const float scale);

faces detect_faces(const cv::ocl::oclMat &ocl_frame, cv::ocl::OclCascadeClassifier &face_cascade, cv::ocl::OclCascadeClassifier &eyes_cascade, const float scale);

void print_faces(const faces &detected_faces, cv::Mat &frame, const float scale_width, const float scale_height);

std::vector<cv::Vec3f> detect_circles(const cv::Mat &image);

std::vector<cv::Rect> detect_faces_dual(const cv::ocl::oclMat &ocl_frame, cv::ocl::OclCascadeClassifier &face_cascade, cv::ocl::OclCascadeClassifier &eyes_cascade,
    const float scale, dlib::frontal_face_detector &dlib_detector, const cv::Mat &color_frame, cv::Mat &color_display_frame);

}
