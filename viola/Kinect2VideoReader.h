#pragma once

#include "Kinect2Feed.h"

#include <string>
using namespace std;


class Kinect2VideoReader : public Kinect2Feed
{
public:
    Kinect2VideoReader(const string &serial_number, const string &file_base_name, const string &file_extension);
    void skip_n_frames(const size_t n);
    void skip_n_seconds(const double n);
    
    void close();

    void grab(cv::Mat &color, cv::Mat &depth);
    void grab_copy(cv::Mat &color, cv::Mat &depth);

protected:
    double t0, t1;
    cv::VideoCapture rgb;
    cv::VideoCapture depth_1_3;
    cv::VideoCapture depth_4;
    float inv_framerate;
    cv::Mat current_rgb, current_depth;
    void update_frames();
    
    void get_depth_frame(cv::Mat &frame);
    void get_rgb_frame(cv::Mat &frame);
};

Kinect2VideoReader::Kinect2VideoReader(const string &serial_number, const string &file_base_name, const string &file_extension) :
    Kinect2Feed(serial_number),
    t0(0),
    t1(0),
    rgb(file_base_name + std::string("_color.") + file_extension),
    depth_1_3(file_base_name + std::string("_depth_1_3.") + file_extension),
    depth_4(file_base_name + std::string("_depth_4.") + file_extension),
    inv_framerate(1.0/rgb.get(CV_CAP_PROP_FPS))
{
    if (!rgb.isOpened() || !depth_1_3.isOpened() || !depth_4.isOpened()){
        std::cout << "One of the videos couldn't be opened." << std::endl;
        std::cout << file_base_name + std::string("_color.") + file_extension << std::endl;
        std::cout << file_base_name + std::string("_depth_1_3.") + file_extension << std::endl;
        std::cout << file_base_name + std::string("_depth_4.") + file_extension << std::endl;
        exit(-1);
    }
}


void Kinect2VideoReader::close()
{
    t0 = t1;
}

void Kinect2VideoReader::skip_n_frames(const size_t n)
{
    const double next_frame = rgb.get(CV_CAP_PROP_POS_FRAMES);
    rgb.set(CV_CAP_PROP_POS_FRAMES, next_frame + n);
}

void Kinect2VideoReader::skip_n_seconds(const double n)
{

    const double current_ms_time = rgb.get(CV_CAP_PROP_POS_MSEC);
    const double next_ms_time = current_ms_time + n * 1000;
    
    const double next_frame11 = rgb.get(CV_CAP_PROP_POS_FRAMES);
    const double next_frame22 = depth_1_3.get(CV_CAP_PROP_POS_FRAMES);
    const double next_frame33 = depth_4.get(CV_CAP_PROP_POS_FRAMES);

    rgb.set(CV_CAP_PROP_POS_MSEC, next_ms_time);
    depth_1_3.set(CV_CAP_PROP_POS_MSEC, next_ms_time);
    depth_4.set(CV_CAP_PROP_POS_MSEC, next_ms_time);
    
    const double next_frame1 = rgb.get(CV_CAP_PROP_POS_FRAMES);
    const double next_frame2 = depth_1_3.get(CV_CAP_PROP_POS_FRAMES);
    const double next_frame3 = depth_4.get(CV_CAP_PROP_POS_FRAMES);
}

void Kinect2VideoReader::update_frames()
{
    get_rgb_frame(current_rgb);
    get_depth_frame(current_depth);
}

void Kinect2VideoReader::get_rgb_frame(cv::Mat &frame)
{
    rgb >> frame;
}

void Kinect2VideoReader::get_depth_frame(cv::Mat &depthf)
{   
    cv::Mat depth3;
    cv::Mat depth4;
    
    depth_1_3 >> depth3;
    depth_4 >> depth4;

    
    depthf = cv::Mat::zeros(depth3.rows, depth3.cols, CV_32FC1);

    size_t n_bytes = depth3.rows * depth3.cols;

    uchar *depthf_ptr = depthf.data;
    uchar *depth3_ptr = depth3.data;
    uchar *depth4_ptr = depth4.data;
    for (size_t i = 0; i < n_bytes; i++){
      *(depthf_ptr + 0) = *(depth3_ptr + 0);
      *(depthf_ptr + 1) = *(depth3_ptr + 1);
      *(depthf_ptr + 2) = *(depth3_ptr + 2);
      *(depthf_ptr + 3) = *(depth4_ptr + 0);
      depth3_ptr += 3;
      depth4_ptr += 3;
      depthf_ptr += 4;
    }
}


void Kinect2VideoReader::grab_copy(cv::Mat &color, cv::Mat &depth)
{
    grab(color, depth);
}

void Kinect2VideoReader::grab(cv::Mat &color, cv::Mat &depth)
{   
    t1 = cv::getTickCount();
    
    if (!opened){
        t0 = t1;
        opened = true;
        update_frames();
    }
    
    const double dt = (t1 - t0) / cv::getTickFrequency();
    
    if (dt > 1/17.0){
        update_frames();
        t0 = t1;
    }
    
    color = current_rgb;
    depth = current_depth;
}