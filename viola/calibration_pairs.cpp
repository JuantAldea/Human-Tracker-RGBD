#include <iostream>

#include "project_config.h"

IGNORE_WARNINGS_PUSH

#include <opencv2/opencv.hpp>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/threading.h>
#include <libfreenect2/registration.h>


IGNORE_WARNINGS_POP

using namespace std;
using namespace cv;

void detect_and_draw_chessboard(const cv::Mat &view, cv::Mat &output)
{
    cv::Size boardSize(9, 7);
    cv::Mat viewGray;
    vector<Point2f> pointbuf;
    cvtColor(view, viewGray, COLOR_BGR2GRAY);
    output = view.clone();
    cv::Mat filtered = viewGray.clone();
    //GaussianBlur(viewGray,filtered, Size(0,0), 1);
    bool found = findChessboardCorners(filtered, boardSize, pointbuf, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
    if(found){
        cornerSubPix(viewGray, pointbuf, Size(11,11), Size(-1,-1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 0.01));
        drawChessboardCorners(output, boardSize, Mat(pointbuf), found);    
    }
}

int main (void)
{
    libfreenect2::Freenect2Device *dev;
    libfreenect2::Freenect2 freenect2;
    libfreenect2::SyncMultiFrameListener *listener;
    libfreenect2::FrameMap frames_kinect2;

    dev = freenect2.openDefaultDevice();

    if (dev == nullptr) {
        std::cout << "no device connected or failure opening the default one!" << std::endl;
        exit(-1);
    }

    listener = new libfreenect2::SyncMultiFrameListener(libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);
    dev->setColorFrameListener(listener);
    dev->setIrAndDepthFrameListener(listener);
    dev->start();
    int i = 0;
    while(true){
        listener->release(frames_kinect2);
        listener->waitForNewFrame(frames_kinect2);
        
        libfreenect2::Frame *rgb = frames_kinect2[libfreenect2::Frame::Color];
        //libfreenect2::Frame *depth = frames_kinect2[libfreenect2::Frame::Depth];
        libfreenect2::Frame *ir = frames_kinect2[libfreenect2::Frame::Ir];

        cv::Mat color_mat = cv::Mat(rgb->height, rgb->width, CV_8UC3, rgb->data);
        cv::Mat ir_mat = cv::Mat(ir->height, ir->width, CV_32FC1, ir->data);
        //cv::Mat depth_mat = cv::Mat(depth->height, depth->width, CV_32FC1, depth->data);

        double min, max;
        cv::minMaxLoc(ir_mat, &min, &max);

        cv::Mat ir_mat_3c;
        std::vector<cv::Mat> channels = { ir_mat, ir_mat, ir_mat };
        cv::merge(channels, ir_mat_3c);
        ir_mat_3c = ir_mat_3c / max;
        cv::Mat ir_mat_3c_8u;
        ir_mat_3c.convertTo(ir_mat_3c_8u, CV_8U, 255.0);
        
        cv::Mat view_ir, view_color;
        detect_and_draw_chessboard(ir_mat_3c_8u, view_ir);
        //detect_and_draw_chessboard(color_mat, view_color);

        imshow("COLOR", color_mat);
        //imshow("ir_mat", ir_mat_3c_8u);
        //imshow("COLOR", view_color);
        imshow("ir_mat", view_ir);

        int c = waitKey(100);
        if ((char)c == 'c') {
            break;
        } else if (char(c) == 's') {
            vector<int> compression_params;
            
            compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(9);
            
            char str_color[100];
            char str_ir[100];
            
            sprintf(str_color, "./imgs/right_%d.png", i);
            sprintf(str_ir, "./imgs/left_%d.png", i);
        
            imwrite(str_color, color_mat, compression_params);
            imwrite(str_ir, ir_mat_3c_8u, compression_params);

            cout << "Frame saved to " << str_color << " & " << str_ir << endl;
            i++;
        }
    }

    dev->stop();
    dev->close();
    listener->release(frames_kinect2);
}

    


