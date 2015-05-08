#pragma once

#include <vector>

#include "project_config.h"

IGNORE_WARNINGS_PUSH
#include <mrpt/otherlibs/do_opencv_includes.h>
IGNORE_WARNINGS_POP

using namespace cv;

namespace viola_faces
{

typedef Rect face_shape;
typedef Rect eye;
typedef std::vector<eye> eyes;
typedef std::pair<face_shape, eyes> face;
typedef std::vector<face> faces;

String face_cascade_name = "cascades/lbpcascade_frontalface.xml";
String eyes_cascade_name = "cascades/haarcascade_eye_tree_eyeglasses.xml";
//String upperbodycascade_name = "cascades/haarcascade_upperbody.xml";

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
//CascadeClassifier upperbody_cascade;
faces detect_faces(const Mat &frame);
void print_faces(const faces &detected_faces, Mat &frame, const float scale_width, const float scale_height);

std::vector<cv::Vec3f> detect_circles(const cv::Mat &image);

}
