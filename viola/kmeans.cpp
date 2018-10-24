#include <mrpt/otherlibs/do_opencv_includes.h>
#include "Kinect2VideoReader.h"
#include <thread>
#include <iostream>
#include "ImageRegistration.h"
using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {

    Kinect2VideoReader video_feed(std::string("013572345247"),
        std::string("/run/media/juen/1cf91ca4-036c-44f3-a9b8-35deb7ced99c/videos/video4/video0"),
        //std::string("/home/juen/videos/video3/video0"),
        std::string("avi"));
    std::thread kinect_frame_update(&Kinect2VideoReader::update, &video_feed);

    char *calib_dir = getenv("HOME");
    const std::string calib_path = std::string(calib_dir) + "/kinect2_calib/";

    //Registration initialization
    ImageRegistration reg;
    reg.init(calib_path, std::string("013572345247"));


    cv::Mat color_mat, depth_mat, src;
    while(true){
        //src = imread(argv[1]);
        //color_mat = imread(argv[1]);
        video_feed.grab(color_mat, depth_mat);
        cv::Mat registered_depth, registered_color;
        std::cout <<"REGISTER\n";
        reg.register_images(color_mat, depth_mat, registered_color, registered_depth);
        std::cout <<"REGISTER2\n";
        color_mat = registered_color;


        blur(src, src, Size(15,15));

        Mat bestLabels, centers, clustered;

        /*
        src = color_mat;
        Mat p = Mat::zeros(src.cols*src.rows, 5, CV_32F);
        vector<Mat> bgr;
        cv::split(src, bgr);
        for(int i=0; i<src.cols*src.rows; i++) {
            p.at<float>(i,0) = (i/src.cols) / src.rows;
            p.at<float>(i,1) = (i%src.cols) / src.cols;
            p.at<float>(i,2) = bgr[0].data[i] / 255.0;
            p.at<float>(i,3) = bgr[1].data[i] / 255.0;
            p.at<float>(i,4) = bgr[2].data[i] / 255.0;
        }
        */


        src = depth_mat;
        Mat p = Mat::zeros(src.cols*src.rows, 3, CV_32F);
        for(int i=0; i<src.cols*src.rows; i++) {
            p.at<float>(i,0) = (i/src.cols) / src.rows;
            p.at<float>(i,1) = (i%src.cols) / src.cols;
            p.at<float>(i,2) = src.at<uint32_t>(i);
            //p.at<float>(i,3) = src.data[i] /4500.0;
            //p.at<float>(i,4) = src.data[i] /4500.0;

        }


        int K = 8;
        std::cout << "KMEANS\n";
        cv::kmeans(p, K, bestLabels,
                TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
                3, KMEANS_PP_CENTERS, centers);

        int colors[K];
        for(int i=0; i<K; i++) {
            colors[i] = 255/(i+1);
        }
        // i think there is a better way to do this mayebe some Mat::reshape?
        clustered = Mat(src.rows, src.cols, CV_32F);
        for(int i=0; i<src.cols*src.rows; i++) {
            clustered.at<float>(i/src.cols, i%src.cols) = (float)(colors[bestLabels.at<int>(0,i)]);
    //      cout << bestLabels.at<int>(0,i) << " " <<
    //              colors[bestLabels.at<int>(0,i)] << " " <<
    //              clustered.at<float>(i/src.cols, i%src.cols) << " " <<
    //              endl;
        }

        clustered.convertTo(clustered, CV_8U);
        imshow("clustered", clustered);
        waitKey(3);
    }
    kinect_frame_update.join();

    waitKey();
    return 0;
}
