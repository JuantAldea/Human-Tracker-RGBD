#include "faces_detection.h"

using namespace cv;

namespace viola_faces
{

void print_faces(const faces &detected_faces, Mat &frame, float scale_width, float scale_height)
{
    for (auto detected_face : detected_faces) {
        face_shape &f = detected_face.first;
        eyes &e = detected_face.second;
        Point center((f.x + f.width / 2 + f.width * 0.50 * 0.05) * scale_width, (f.y + f.height / 2) * scale_height);
        ellipse(frame, center, Size((f.width / 2) * scale_width * 0.8, (f.height / 2) * scale_height), 0, 0, 360, Scalar(255, 0, 0), 2, 8, 0);
        ellipse(frame, center, Size(5, 5), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
        for (size_t j = 0; j < e.size(); j++) {
            Point eye_center((f.x + e[j].x + e[j].width / 2) * scale_width, (f.y + e[j].y + e[j].height / 2) * scale_height);
            int radius = cvRound((e[j].width + e[j].height) * 0.25 * scale_width);
            circle(frame, eye_center, radius, Scalar(255, 0, 255), 3, 8, 0);
        }
    }
}

faces detect_faces(const Mat &frame, CascadeClassifier &face_cascade, CascadeClassifier &eyes_cascade)
{
    Mat frame_gray;

    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    faces detected_faces;

    std::vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0, Size(80, 80));
    for (auto detected_face : faces) {
        Mat faceROI = frame_gray(detected_face);
        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
        if (eyes.size() == 2) {
            detected_faces.push_back(face(detected_face, eyes));
        }
    }

    return detected_faces;
}

std::vector<cv::Vec3f> detect_circles(const cv::Mat &image)
{
    using namespace cv;
    vector<Vec3f> circles;
    Mat src_gray;
    cvtColor(image, src_gray, CV_BGR2GRAY);
    GaussianBlur(src_gray, src_gray, Size(3, 3), 2, 2);
    HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows / 8, 200, 100, 50, 0);
    return circles;
}

}
