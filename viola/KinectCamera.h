#pragma once

#include <map>

#include "project_config.h"

IGNORE_WARNINGS_PUSH

#include <mrpt/otherlibs/do_opencv_includes.h>

#ifdef USE_KINECT_2
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#endif

IGNORE_WARNINGS_POP

class KinectCamera
{
public:
    enum class FrameType
    {
        COLOR, DEPTH,
        #ifdef USE_KINECT_2
        IR,
        #else
        GRAY_IMAGE, DISPARITY_MAP, VALID_DEPTH_MASK,
        #endif
    };

    enum class CameraType
    {
        COLOR,
        IR
    };

#ifdef USE_KINECT_2
    using IRCameraParams = libfreenect2::Freenect2Device::IrCameraParams;
#else
    struct IRCameraParams
    {
        float cx;
        float cy;
        float fx;
        float fy;
        float k1;
        float k2;
        float k3;
        float p1;
        float p2;
        IRCameraParams():
            cx(0), cy(0), fx(0), fy(0), 
            k1(0), k2(0), k3(0), 
            p1(0), p2(0)
        {
            ;
        }
    };
#endif

    typedef std::map<FrameType, cv::Mat> FrameMap;
    KinectCamera();
    ~KinectCamera();
    void grabFrames();
    int open();
    void close();
    FrameMap frames;
    KinectCamera::IRCameraParams getIRCameraParams() const;

    static std::pair<int, int> getFrameSize(FrameType type);

private:

#ifdef USE_KINECT_2
    libfreenect2::Freenect2Device *dev;
    libfreenect2::Freenect2 freenect2;
    libfreenect2::SyncMultiFrameListener *listener;
    libfreenect2::FrameMap frames_kinect2;
#else
    cv::VideoCapture capture;
#endif
};
