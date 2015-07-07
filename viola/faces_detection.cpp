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

faces detect_faces(const Mat &frame, CascadeClassifier &face_cascade, CascadeClassifier &eyes_cascade, const float scale)
{
    const float inv_scale = 1.f / scale;
    cv::Mat frame_gray_aux;
    if(frame.channels() > 1){
        std::cout << "COLOR CONVERT" << std::endl;
        cv::cvtColor(frame, frame_gray_aux, CV_BGR2GRAY);
    } else {
        frame_gray_aux = frame;
    }
    
    equalizeHist(frame_gray_aux, frame_gray_aux);

    cv::Mat frame_gray;
    if (scale != 1){
        frame_gray = cv::Mat(cvRound(frame_gray_aux.rows * inv_scale), cvRound(frame_gray_aux.cols * inv_scale), frame_gray_aux.type());
        cv::resize(frame_gray_aux, frame_gray, frame_gray.size(), 0, 0, INTER_LINEAR);
    } else {
         frame_gray = frame_gray_aux;
    }
    
    faces detected_faces;
    std::vector<Rect> faces;
    //face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(80, 80));
    face_cascade.detectMultiScale(frame_gray, faces, 1.2, 5, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30), Size(190, 190));
    for (auto detected_face : faces) {
        std::cout << "EYES FACES DETECTED: " << faces.size() << std::endl;
        Mat faceROI = frame_gray(detected_face);
        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE);
        if (eyes.size() == 2) {
            std::cout << "FOUND " << detected_face << std::endl;
            cv::Rect face_rescaled = cv::Rect(
                detected_face.x * scale,
                detected_face.y * scale,
                detected_face.width * scale,
                detected_face.height * scale);
            detected_faces.push_back(face(face_rescaled, eyes));
            //detected_faces.push_back(face(detected_face, eyes));
        }
        detected_faces.push_back(face(detected_face, eyes));
    }

    return detected_faces;
}


faces detect_faces(const cv::ocl::oclMat &ocl_frame, cv::ocl::OclCascadeClassifier &face_cascade, cv::ocl::OclCascadeClassifier &eyes_cascade, const float scale)
{
    const float inv_scale = 1.f / scale;
    cv::ocl::oclMat ocl_frame_gray_aux;
    if(ocl_frame.channels() > 1){
        cv::ocl::cvtColor(ocl_frame, ocl_frame_gray_aux, CV_BGR2GRAY);
    } else {
        ocl_frame_gray_aux = ocl_frame;
    }

    cv::ocl::equalizeHist(ocl_frame_gray_aux, ocl_frame_gray_aux);
    
    //cv::ocl::oclMat ocl_frame_gray;
    cv::ocl::oclMat ocl_frame_gray(cvRound(ocl_frame_gray_aux.rows * inv_scale), cvRound(ocl_frame_gray_aux.cols * inv_scale), ocl_frame_gray_aux.type());
    cv::ocl::resize(ocl_frame_gray_aux, ocl_frame_gray, ocl_frame_gray.size(), 0, 0, INTER_LINEAR);

    faces detected_faces;
    
    std::vector<Rect> faces;
    //face_cascade.detectMultiScale(ocl_frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(80, 80));
    face_cascade.detectMultiScale(ocl_frame_gray, faces, 1.2, 5, 0 | CV_HAAR_SCALE_IMAGE, Size(28, 48), Size(354, 590));
    std::cout << "EYES FACES DETECTED: " << faces.size() << std::endl;
    for (auto detected_face : faces) {
        cv::ocl::oclMat faceROI = ocl_frame_gray(detected_face);
        std::vector<Rect> eyes;
        //eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
        if (eyes.size() == 2) {
            cv::Rect face_rescaled = cv::Rect(
                detected_face.x * scale,
                detected_face.y * scale,
                detected_face.width * scale,
                detected_face.height * scale);
            detected_faces.push_back(face(face_rescaled, eyes));
        }
        detected_faces.push_back(face(detected_face, eyes));
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
    HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows / 8, 170, 100, 80, 300);
    return circles;
}

}
