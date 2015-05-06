#pragma once

#include <vector>

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
