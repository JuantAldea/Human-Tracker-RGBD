#include <thread>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include "FacesDetection.h"
#include "Kinect2VideoReader.h"
#include "fuzzyHRI3/smileestimator.h"

#include "ImageRegistration.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    SmileEstimator SmileEstimatorForFace;
    string face_cascade_name = "../cascades/haarcascade_frontalface_alt2.xml";
    string eyes_cascade_name = "../cascades/haarcascade_eye_tree_eyeglasses.xml";
    cv::ocl::OclCascadeClassifier ocl_face_cascade;
    cv::ocl::OclCascadeClassifier ocl_eyes_cascade;

    if (!ocl_face_cascade.load(face_cascade_name)) {
        printf("--(!)Error loading %s\n", face_cascade_name.c_str());
        return -1;
    }

    if (!ocl_eyes_cascade.load(eyes_cascade_name)) {
        printf("--(!)Error loading %s\n", eyes_cascade_name.c_str());
        return -1;
    }

    dlib::frontal_face_detector face_detector = dlib::get_frontal_face_detector();
    string name = std::string("/run/media/juen/1cf91ca4-036c-44f3-a9b8-35deb7ced99c/videos/video0/video0");
    cerr << name << std::endl;
    name[65] = argv[1][0];
    cerr << name << std::endl;
    Kinect2VideoReader video_feed(std::string("013572345247"),
        //std::string("/run/media/juen/1cf91ca4-036c-44f3-a9b8-35deb7ced99c/videos/video5/video0"),
        name,
        //std::string("/run/media/juen/1cf91ca4-036c-44f3-a9b8-35deb7ced99c/videos/video0/video0"),
        //std::string("/home/juen/videos/video3/video0"),
        std::string("avi"));
    //std::thread kinect_frame_update(&Kinect2VideoReader::update, &video_feed);
    char *calib_dir = getenv("HOME");

    const std::string calib_path = std::string(calib_dir) + "/kinect2_calib/";

    //Registration initialization
    ImageRegistration reg;
    reg.init(calib_path, video_feed.get_device_serial_number());

    cv::Mat color_mat, depth_mat;
    cv::Mat color_display_frame;
    cv::waitKey(40);

    cv::BackgroundSubtractorMOG2 background_subtractor;
    background_subtractor.set("nmixtures", 3);
    cv::Mat background_mask;

    while (true) {
        video_feed.grab_next(color_mat, depth_mat);
        cv::Mat registered_color, registered_depth;
        reg.register_images(color_mat, depth_mat, registered_color, registered_depth);

        color_display_frame = registered_color.clone();

        background_subtractor(registered_color, background_mask);

        cv::ocl::oclMat ocl_color_frame(registered_color);
        cv::ocl::oclMat ocl_gray_frame;
        cv::ocl::cvtColor(ocl_color_frame, ocl_gray_frame, cv::COLOR_BGR2GRAY);
        vector<Rect> faces_roi = viola_faces::detect_faces_dual(ocl_gray_frame, ocl_face_cascade, ocl_eyes_cascade, 1, face_detector, registered_color, color_display_frame);

        if (faces_roi.size()) {
            faces_roi[0].x = faces_roi[0].x + faces_roi[0].width * 0.25;
            faces_roi[0].width *= 0.50;
            //faces_roi[0].height *= 1.2;

            Mat gray_frame = ocl_gray_frame;
            Mat gray_face = gray_frame(faces_roi[0]);
            Mat depth_face = registered_depth(faces_roi[0]);

            int face_x = depth_face.cols * 0.5;
            int face_y = depth_face.rows * 0.5;

            uint16_t z = depth_face.at<uint16_t>(face_y, face_x);

            uint16_t Z_TOLERANCE = 250;
            //Mat face_mask;
            //cv::inRange(depth_face, std::max(z - Z_TOLERANCE, 600), z + Z_TOLERANCE, face_mask);


            imshow("FACE", gray_face);
            imshow("DEPTH FACE", depth_face);
            //imshow("DEPTH MASK", face_mask);

            Mat small_gray_face(40, 48, 8, 3);
            Mat small_depth_face(40, 48, 1, 1);

            resize(gray_face, small_gray_face, small_gray_face.size());
            resize(depth_face, small_depth_face, small_depth_face.size());

            imshow("FACE SMALL", small_gray_face);
            Mat face_mask;
            cv::inRange(small_depth_face, std::max(z - Z_TOLERANCE, 600), z + Z_TOLERANCE, face_mask);
            imshow("DEPTH SMALL", face_mask);
            //cout << face_mask.rows << ' ' << face_mask.cols << std::endl;
            //cout << face_mask.rows << ' ' << face_mask.cols << std::endl;
            Mat small_gray_face_masked;//(small_gray_face.size(), small_gray_face.type());
            //face_mask = Mat::ones(face_mask.size(), face_mask.type());
            small_gray_face.copyTo(small_gray_face_masked, face_mask);
            imshow("FACE SMALL MASKED", small_gray_face_masked);

            IplImage * ipl_small_gray_face = new IplImage(small_gray_face_masked);
            int smileClass = SmileEstimatorForFace.estimate(ipl_small_gray_face);
            cout << "STREAM:0:" << smileClass - 1 << endl;
            cout << "STREAM:1:" << z << endl;
            cv::Scalar color = (smileClass == 1) ? Scalar(0, 0, 0) : Scalar(255, 255, 255);


            rectangle(color_display_frame, faces_roi[0], color, 1);
            delete ipl_small_gray_face;
        }

        imshow("VIDEO", color_display_frame);
        //imshow("DEPTH", registered_depth);
        cv::waitKey(1);
    }
}
