#include "KinectCamera.h"
#include <thread>
int KinectCamera::open()
{
#ifdef USE_KINECT_2
    dev = freenect2.openDefaultDevice();

    if (dev == nullptr) {
        std::cout << "no device connected or failure opening the default one!" << std::endl;
        return -1;
    }

    dev->setColorFrameListener(listener);
    dev->setIrAndDepthFrameListener(listener);
    dev->start();

    std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
    std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;
#else
    capture.open(CV_CAP_OPENNI);
    capture.set(CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_SXGA_15HZ);

    if (!capture.isOpened()) {
        return -1;
    }

#endif
    return 0;
}

void KinectCamera::close()
{
#ifdef USE_KINECT_2
    dev->stop();
    dev->close();
    listener->release(frames_kinect2);
#else
    capture.release();
#endif
}

KinectCamera::KinectCamera()
{
    frames[FrameType::COLOR] = cv::Mat::ones(640, 480, CV_8UC3);
    frames[FrameType::DEPTH] = cv::Mat::ones(640, 480, CV_32F);
#ifdef USE_KINECT_2
    listener = new libfreenect2::SyncMultiFrameListener(libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);
    frames[FrameType::IR] = cv::Mat::ones(640, 480, CV_8UC3);
#else
    frames[FrameType::GRAY_IMAGE] = cv::Mat();
    frames[FrameType::DISPARITY_MAP] = cv::Mat();
    frames[FrameType::VALID_DEPTH_MASK] = cv::Mat();
#endif
}

KinectCamera::~KinectCamera()
{
    close();
    #ifdef USE_KINECT_2
    delete dev;
    delete listener;
    #endif
}

void KinectCamera::grabFrames()
{
#ifdef USE_KINECT_2
        listener->release(frames_kinect2);
        listener->waitForNewFrame(frames_kinect2);
        const libfreenect2::Frame *rgb = frames_kinect2[libfreenect2::Frame::Color];
        const libfreenect2::Frame *depth = frames_kinect2[libfreenect2::Frame::Depth];
        const libfreenect2::Frame *ir = frames_kinect2[libfreenect2::Frame::Ir];
        frames[FrameType::COLOR] = cv::Mat(rgb->height, rgb->width, CV_8UC3, rgb->data);
        frames[FrameType::DEPTH] = cv::Mat(depth->height, depth->width, CV_32FC1, depth->data);
        frames[FrameType::IR]    = cv::Mat(depth->height, depth->width, CV_32FC1, ir->data);
#else
        capture.grab();
        capture.retrieve(frames[FrameType::COLOR], CV_CAP_OPENNI_BGR_IMAGE);
        capture.retrieve(frames[FrameType::DEPTH], CV_CAP_OPENNI_DEPTH_MAP);
        capture.retrieve(frames[FrameType::GRAY_IMAGE], CV_CAP_OPENNI_GRAY_IMAGE);
        capture.retrieve(frames[FrameType::DISPARITY_MAP], CV_CAP_OPENNI_DISPARITY_MAP);
        capture.retrieve(frames[FrameType::VALID_DEPTH_MASK], CV_CAP_OPENNI_VALID_DEPTH_MASK);
#endif
}
