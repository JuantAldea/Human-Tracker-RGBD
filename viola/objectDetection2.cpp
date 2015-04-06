/**
 * @file objectDetection2.cpp
 * @author A. Huaman ( based in the classic facedetect.cpp in samples/c )
 * @brief A simplified version of facedetect.cpp, show how to load a cascade classifier and how to find objects (Face + eyes) in a video stream - Using LBP here
 */
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


typedef Rect face_shape;
typedef Rect eye;
typedef vector<eye> eyes;
typedef pair<face_shape, eyes> face;
typedef vector<face> faces;

void print_faces(const faces &detected_faces, Mat &frame, float scale);
faces detect_faces(const Mat &frame);
vector<Rect> detect_upper_bodies(const Mat &frame);
void CannyThreshold(int, void* io);


String face_cascade_name = "lbpcascade_frontalface.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
String upperbodycascade_name = "haarcascade_upperbody.xml";

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier upperbody_cascade;

int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;

RNG rng(12345);


int main(void)
{
    if (!face_cascade.load(face_cascade_name)) {
        printf("--(!)Error loading\n");
        return -1;
    };

    if (!eyes_cascade.load(eyes_cascade_name)) {
        printf("--(!)Error loading\n");
        return -1;
    };

    if (!upperbody_cascade.load(upperbodycascade_name)) {
        printf("--(!)Error loading\n");
        return -1;
    };

    namedWindow("Color image", CV_WINDOW_AUTOSIZE);
    //namedWindow("Grey canny", CV_WINDOW_AUTOSIZE);
    //namedWindow("Disparity map", CV_WINDOW_AUTOSIZE);
    namedWindow("Disparity map eq", CV_WINDOW_AUTOSIZE);
    //namedWindow("Disparity map canny", CV_WINDOW_AUTOSIZE);
    namedWindow("Disparity map canny eq", CV_WINDOW_AUTOSIZE);
    //namedWindow("Depth image", CV_WINDOW_AUTOSIZE);
    //namedWindow("Point cloud map", CV_WINDOW_AUTOSIZE);
    //namedWindow("Grey image", CV_WINDOW_AUTOSIZE);

    VideoCapture capture(CV_CAP_OPENNI);
    capture.set(CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_SXGA_15HZ);

    Mat color_frame;
    Mat grey_frame;
    Mat depth_frame;
    Mat grey_canny;
    Mat disparity_map;
    Mat disparity_map_canny;
    Mat disparity_map_eq;
    Mat disparity_map_eq_canny;

    pair<Mat*, Mat*> io(&grey_frame, &grey_canny);
    pair<Mat*, Mat*> io2(&disparity_map, &disparity_map_canny);
    pair<Mat*, Mat*> io3(&disparity_map_eq, &disparity_map_eq_canny);

    createTrackbar("Min Threshold:", "Disparity map canny", &lowThreshold, max_lowThreshold, CannyThreshold, &io2);
    createTrackbar("Min Threshold:", "Disparity map canny", &lowThreshold, max_lowThreshold, CannyThreshold, &io2);
    createTrackbar("Min Threshold:", "Disparity map canny eq", &lowThreshold, max_lowThreshold, CannyThreshold, &io3);

    if (!capture.isOpened()) {
        return -1;
    }

    while (true) {
        capture.grab();
        capture.retrieve(color_frame, CV_CAP_OPENNI_BGR_IMAGE);
        capture.retrieve(grey_frame, CV_CAP_OPENNI_GRAY_IMAGE);
        capture.retrieve(disparity_map, CV_CAP_OPENNI_DISPARITY_MAP);
        //capture.retrieve(depth_frame, CV_CAP_OPENNI_DEPTH_MAP);
        //capture.retrieve(point_cloud_map, CV_CAP_OPENNI_POINT_CLOUD_MAP);

        auto detected_faces = detect_faces(color_frame);
        print_faces(detected_faces, color_frame, 1);

        //CannyThreshold(0, &io);
        //CannyThreshold(0, &io2);

        equalizeHist(disparity_map, disparity_map_eq);
        CannyThreshold(0, &io3);
        print_faces(detected_faces, disparity_map_eq, 0.5);
        print_faces(detected_faces, disparity_map_eq_canny, 0.5);


        //vector<Vec3f> circles;
        //blur(grey_frame, grey_frame, Size(3, 3));
        //HoughCircles(grey_frame, circles, CV_HOUGH_GRADIENT, 1, 1, 200, 100, 0, 0);
        /*
        auto bodies = detect_upper_bodies(color_frame);

        for (auto body : bodies) {
            rectangle(color_frame, body, Scalar(0, 0, 255));
        }
        for (size_t i = 0; i < circles.size(); i++) {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            // circle center
            circle(color_frame, center, 3, Scalar(0, 255, 0), -1, 8, 0);
            // circle outline
            circle(color_frame, center, radius, Scalar(0, 0, 255), 3, 8, 0);
        }

        */
        //imshow("Disparity map", disparity_map);
        imshow("Disparity map eq", disparity_map_eq);
        //imshow("Disparity map canny", disparity_map_canny);
        imshow("Disparity map canny eq", disparity_map_eq_canny);
        imshow("Color image", color_frame);
        //imshow("Grey canny", grey_canny);

        //imshow("Depth map", depth_frame);
        //imshow("Point cloud map", point_cloud_map);
        //imshow("Grey image", grey_frame);

        int c = waitKey(10);
        if ((char)c == 'c') {
            break;
        }

    }
    return 0;
}

void print_faces(const faces &detected_faces, Mat &frame, float scale)
{
    for (auto detected_face : detected_faces) {
        face_shape &f = detected_face.first;
        eyes &e = detected_face.second;
        Point center((f.x + f.width / 2) * scale, (f.y + f.height / 2) * scale);
        ellipse(frame, center, Size((f.width / 2) * scale, (f.height / 2) * scale), 0, 0, 360, Scalar(255, 0, 0), 2, 8, 0);
        for (size_t j = 0; j < e.size(); j++) {
            Point eye_center((f.x + e[j].x + e[j].width / 2) * scale, (f.y + e[j].y + e[j].height / 2) * scale);
            int radius = cvRound((e[j].width + e[j].height) * 0.25 * scale);
            circle(frame, eye_center, radius, Scalar(255, 0, 255), 3, 8, 0);
        }
    }
}


faces detect_faces(const Mat &frame)
{
    Mat frame_gray;

    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    faces detected_faces;

    std::vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0, Size(80, 80));
    for (size_t i = 0; i < faces.size(); i++) {
        Mat faceROI = frame_gray(faces[i]);
        std::vector<Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
        if (eyes.size() == 2) {
            detected_faces.push_back(face(faces[i], eyes));
        }
    }

    return detected_faces;
}

vector<Rect> detect_upper_bodies(const Mat &frame)
{
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    std::vector<Rect> upperbodies;
    upperbody_cascade.detectMultiScale(frame, upperbodies, 1.1, 2, 0, Size(110, 110));
    //for (size_t i = 0; i < upperbodies.size(); i++) {
        //Point center(upperbodies[i].x + upperbodies[i].width / 2, upperbodies[i].y + upperbodies[i].height / 2);
        //ellipse(frame, center, Size(upperbodies[i].width / 2, upperbodies[i].height / 2), 0, 0, 360, Scalar(500, 0, 0), 2, 8, 0);
    //}
    return upperbodies;

}

void CannyThreshold(int, void *io)
{
    Mat input(*((std::pair<Mat*, Mat*>*)io)->first);
    Mat *output = ((std::pair<Mat*, Mat*>*)io)->second;
    blur(input, *output, Size(5, 5));
    Canny(*output, *output, lowThreshold, lowThreshold * ratio, kernel_size);
}
