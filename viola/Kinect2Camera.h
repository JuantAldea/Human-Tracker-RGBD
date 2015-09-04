#pragma once
#include <string>
using namespace std;

#include "Kinect2Feed.h"

IGNORE_WARNINGS_PUSH

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/threading.h>
#include <libfreenect2/registration.h>

IGNORE_WARNINGS_POP

class Kinect2Camera : public Kinect2Feed
{
public:
    Kinect2Camera();
    Kinect2Camera(const string &serial);

    ~Kinect2Camera();

    void open();
    void open(const string &serial);

    void close();

    void grab(cv::Mat &color, cv::Mat &depth) override;
    void grab_copy(cv::Mat &color, cv::Mat &depth) override;

protected:
    libfreenect2::Freenect2Device *dev;
    libfreenect2::Freenect2 freenect2;
    libfreenect2::SyncMultiFrameListener *listener;
    libfreenect2::FrameMap frames_kinect2;
    libfreenect2::PacketPipeline *pipeline;
};

Kinect2Camera::Kinect2Camera(const string &serial) :
    Kinect2Feed(serial)
{
    open(serial);
}

Kinect2Camera::Kinect2Camera() :
    Kinect2Feed()
{
    open();
}

Kinect2Camera::~Kinect2Camera()
{
    close();
}

void Kinect2Camera::open()
{
    if (freenect2.enumerateDevices() == 0) {
        std::cout << "No Kinect V2 connected." << std::endl;
        exit(-1);
    }

    device_serial_number = freenect2.getDefaultDeviceSerialNumber();
    open(device_serial_number);
}


void Kinect2Camera::open(const string &serial_number)
{
    if (freenect2.enumerateDevices() == 0) {
        std::cout << "No Kinect V2 connected." << std::endl;
        exit(-1);
    }

    device_serial_number = serial_number;

    pipeline = new libfreenect2::OpenCLPacketPipeline();

    const float minDepth = 0;
    const float maxDepth = 12.0;
    const bool bilateral_filter = true;
    const bool edge_aware_filter = true;

    libfreenect2::DepthPacketProcessor::Config config;
    config.EnableBilateralFilter = bilateral_filter;
    config.EnableEdgeAwareFilter = edge_aware_filter;
    config.MinDepth = minDepth;
    config.MaxDepth = maxDepth;

    pipeline->getDepthPacketProcessor()->setConfiguration(config);
    dev = freenect2.openDevice(device_serial_number, pipeline);

    if (dev == nullptr) {
        std::cout << "Device #" << device_serial_number << " not conected." << std::endl;
        exit(-1);
    }

    //listener = new libfreenect2::SyncMultiFrameListener(libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);
    listener = new libfreenect2::SyncMultiFrameListener(libfreenect2::Frame::Color | libfreenect2::Frame::Depth);
    dev->setColorFrameListener(listener);
    dev->setIrAndDepthFrameListener(listener);
    dev->start();
    opened = true;
}

void Kinect2Camera::close()
{
    if(opened){
        listener->release(frames_kinect2);
        dev->stop();
        dev->close();
    }
}

void Kinect2Camera::grab(cv::Mat &color_mat, cv::Mat &depth_mat)
{
    listener->release(frames_kinect2);
    listener->waitForNewFrame(frames_kinect2);

    libfreenect2::Frame *rgb = frames_kinect2[libfreenect2::Frame::Color];
    color_mat = cv::Mat(rgb->height, rgb->width, CV_8UC3, rgb->data);

    libfreenect2::Frame *depth = frames_kinect2[libfreenect2::Frame::Depth];
    depth_mat = cv::Mat(depth->height, depth->width, CV_32FC1, depth->data);

    //libfreenect2::Frame *ir = frames_kinect2[libfreenect2::Frame::Ir];
    //ir = cv::Mat(depth->height, depth->width, CV_32FC1, ir->data);
}

void Kinect2Camera::grab_copy(cv::Mat &color_mat, cv::Mat &depth_mat)
{
    cv::Mat color_mat_shared, depth_mat_shared;
    grab(color_mat_shared, depth_mat_shared);
    color_mat = color_mat_shared.clone();
    depth_mat = depth_mat_shared.clone();
}
