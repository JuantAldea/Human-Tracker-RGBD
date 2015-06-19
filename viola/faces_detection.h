#pragma once

#include <vector>

#include "project_config.h"

IGNORE_WARNINGS_PUSH

#include <mrpt/otherlibs/do_opencv_includes.h>

IGNORE_WARNINGS_POP

namespace viola_faces
{

using namespace cv;

typedef cv::Rect face_shape;
typedef cv::Rect eye;
typedef std::vector<eye> eyes;
typedef std::pair<face_shape, eyes> face;
typedef std::vector<face> faces;


faces detect_faces(const Mat &frame, CascadeClassifier &face_cascade, CascadeClassifier &eyes_cascade);

void print_faces(const faces &detected_faces, cv::Mat &frame, const float scale_width, const float scale_height);

std::vector<cv::Vec3f> detect_circles(const cv::Mat &image);

}
