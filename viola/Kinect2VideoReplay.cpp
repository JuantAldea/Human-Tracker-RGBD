#include "Kinect2VideoReader.h"

#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    Kinect2VideoReader video_feed = Kinect2VideoReader(std::string("013572345247"), std::string("/media/juant/VIDEOS/video/video0"), std::string("avi"));
    VideoCapture color ("/media/juant/VIDEOS/video/video0_color.avi");
    double t0 = cv::getTickCount();
    while(true){
        Mat frame;
        cout << color.get(CV_CAP_PROP_POS_MSEC) << endl;
        cout << color.get(CV_CAP_PROP_POS_FRAMES) << endl;
        cv::Mat rgb, depth;
        video_feed.grab(rgb, depth);
        //color >> frame;
        //imshow("frame", frame);
        double t1 = cv::getTickCount();
        double dt =  (t1 - t0) /cv::getTickFrequency();
        cv::Mat rgb2;
        cv::flip(rgb, rgb2, 1);

        {
            std::ostringstream oss;
            oss << "TIME " << dt;

            int fontFace =  cv::FONT_HERSHEY_PLAIN;
            double fontScale = 2;
            int thickness = 2;

            int baseline = 0;
            cv::Size textSize = cv::getTextSize(oss.str(), fontFace, fontScale, thickness, &baseline);
            cv::Point textOrg(rgb.cols/2 + 50, rgb.rows/2 + 100);
            putText(rgb2, oss.str(), textOrg, fontFace, fontScale, cv::Scalar(255, 255, 0), thickness, 8);
        }
        imshow("rgb", rgb2);
        imshow("depth", depth/ 4500.0f);
        waitKey(1);
    }
}
