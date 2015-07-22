#pragma once

#include <opencv2/opencv.hpp>

#include <string>
using namespace std;

class Kinect2Feed 
{
public:
    Kinect2Feed();
    Kinect2Feed(const string &serial);
    
    virtual void close();

    virtual void grab(cv::Mat &color, cv::Mat &depth) = 0;
    virtual void grab_copy(cv::Mat &color, cv::Mat &depth) = 0;

    std::string get_device_serial_number() const;

protected:
    std::string device_serial_number;
    bool opened;
};

std::string Kinect2Feed::get_device_serial_number() const
{
    return device_serial_number;
}

Kinect2Feed::Kinect2Feed() :
    opened(false)
{
    ;
}

Kinect2Feed::Kinect2Feed(const string &serial) :
    device_serial_number(serial),
    opened(false)
{
    ;
}

void Kinect2Feed::close()
{
    ;    
}