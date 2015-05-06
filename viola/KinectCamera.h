#pragma once

#include <map>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Werror"
#pragma GCC diagnostic ignored "-Wlong-long"

#pragma GCC diagnostic ignored "-pedantic"
#pragma GCC diagnostic ignored "-pedantic-errors"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <mrpt/otherlibs/do_opencv_includes.h>
#define USE_KINECT_2
#ifdef USE_KINECT_2
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#endif

#pragma GCC diagnostic pop

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

    typedef std::map<FrameType, cv::Mat> FrameMap;
    KinectCamera();
    ~KinectCamera();
    void grabFrames();
    int open();
    void close();
    FrameMap frames;

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

