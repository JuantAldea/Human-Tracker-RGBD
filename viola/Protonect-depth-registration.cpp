/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2011 individual OpenKinect contributors. See the CONTRIB file
 * for details.
 *
 * This code is licensed to you under the terms of the Apache License, version
 * 2.0, or, at your option, the terms of the GNU General Public License,
 * version 2.0. See the APACHE20 and GPL2 files for the text of the licenses,
 * or the following URLs:
 * http://www.apache.org/licenses/LICENSE-2.0
 * http://www.gnu.org/licenses/gpl-2.0.txt
 *
 * If you redistribute this file in source form, modified or unmodified, you
 * may:
 *   1) Leave this header intact and distribute it under the same terms,
 *      accompanying it with the APACHE20 and GPL20 files, or
 *   2) Delete the Apache 2.0 clause and accompany it with the GPL2 file, or
 *   3) Delete the GPL v2 clause and accompany it with the APACHE20 file
 * In all cases you must keep the copyright notice intact and include a copy
 * of the CONTRIB file.
 *
 * Binary distributions must follow the binary distribution requirements of
 * either License.
 */

#include <cstdlib>
#include <iostream>
#include <signal.h>

#include <opencv2/opencv.hpp>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/threading.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>

#include "ImageRegistration.h"

bool protonect_shutdown = false;

void sigint_handler(int s)
{
    protonect_shutdown = true;
}

void dispDepth(const cv::Mat &in, cv::Mat &out, const float maxValue)
{
    cv::Mat tmp = cv::Mat(in.rows, in.cols, CV_8U);
    const uint32_t maxInt = 255;

    #pragma omp parallel for
    for (int r = 0; r < in.rows; ++r) {
        const uint16_t *itI = in.ptr<uint16_t>(r);
        uint8_t *itO = tmp.ptr<uint8_t>(r);

        for (int c = 0; c < in.cols; ++c, ++itI, ++itO) {
            *itO = (uint8_t)std::min((*itI * maxInt / maxValue), 255.0f);
        }
    }

    cv::applyColorMap(tmp, out, cv::COLORMAP_JET);
}

void combine(const cv::Mat &inC, const cv::Mat &inD, cv::Mat &out)
{
    out = cv::Mat(inC.rows, inC.cols, CV_8UC3);

    #pragma omp parallel for
    for (int r = 0; r < inC.rows; ++r) {
        const cv::Vec3b
        *itC = inC.ptr<cv::Vec3b>(r),
         *itD = inD.ptr<cv::Vec3b>(r);
        cv::Vec3b *itO = out.ptr<cv::Vec3b>(r);

        for (int c = 0; c < inC.cols; ++c, ++itC, ++itD, ++itO) {
            itO->val[0] = (itC->val[0] + itD->val[0]) >> 1;
            itO->val[1] = (itC->val[1] + itD->val[1]) >> 1;
            itO->val[2] = (itC->val[2] + itD->val[2]) >> 1;
        }
    }
}

int main(int argc, char *argv[])
{
    std::string program_path(argv[0]);
    size_t executable_name_idx = program_path.rfind("Protonect");

    std::string binpath = "/";

    if (executable_name_idx != std::string::npos) {
        binpath = program_path.substr(0, executable_name_idx);
    }

    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device *dev = 0;
    libfreenect2::PacketPipeline *pipeline = 0;

    if (freenect2.enumerateDevices() == 0) {
        std::cout << "no device connected!" << std::endl;
        return -1;
    }

    std::string serial = freenect2.getDefaultDeviceSerialNumber();

    for (int argI = 1; argI < argc; ++argI) {
        const std::string arg(argv[argI]);

        if (arg == "cpu") {
            if (!pipeline) {
                pipeline = new libfreenect2::CpuPacketPipeline();
            }
        } else if (arg == "gl") {
#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
            if (!pipeline) {
                pipeline = new libfreenect2::OpenGLPacketPipeline();
            }
#else
            std::cout << "OpenGL pipeline is not supported!" << std::endl;
#endif
        } else if (arg == "cl") {
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
            if (!pipeline) {
                pipeline = new libfreenect2::OpenCLPacketPipeline();
            }
#else
            std::cout << "OpenCL pipeline is not supported!" << std::endl;
#endif
        } else if (arg.find_first_not_of("0123456789") == std::string::npos) { //check if parameter could be a serial number
            serial = arg;
        } else {
            std::cout << "Unknown argument: " << arg << std::endl;
        }
    }

    float minDepth = 0;
    float maxDepth = 12.0;
    bool bilateral_filter = true;
    bool edge_aware_filter = true;
    
    if (pipeline) {
        libfreenect2::DepthPacketProcessor::Config config;
        config.EnableBilateralFilter = bilateral_filter;
        config.EnableEdgeAwareFilter = edge_aware_filter;
        config.MinDepth = minDepth;
        config.MaxDepth = maxDepth;
        pipeline->getDepthPacketProcessor()->setConfiguration(config);
        dev = freenect2.openDevice(serial, pipeline);
    }else{
        dev = freenect2.openDevice(serial);
    }

    

    if (dev == 0) {
        std::cout << "failure opening device!" << std::endl;
        return -1;
    }

    signal(SIGINT, sigint_handler);
    protonect_shutdown = false;

    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);
    libfreenect2::FrameMap frames;

    dev->setColorFrameListener(&listener);
    dev->setIrAndDepthFrameListener(&listener);
    dev->start();

    std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
    std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;

    libfreenect2::Registration* registration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());
    unsigned char* registered = NULL;

    char *calib_dir = getenv("HOME");
    std::string calib_path = std::string(calib_dir) + "/kinect2_calib/";
    std::string sensor = serial;

    ImageRegistration reg;
    reg.init(calib_path, sensor);

    
    //text



    while (!protonect_shutdown) {
        listener.waitForNewFrame(frames);
        libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
        libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
        libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];
        cv::Mat color_mat = cv::Mat(rgb->height, rgb->width, CV_8UC3, rgb->data);
        cv::Mat ir_mat = cv::Mat(ir->height, ir->width, CV_32FC1, ir->data);
        cv::Mat depth_mat = cv::Mat(depth->height, depth->width, CV_32FC1, depth->data);
        /*
        if (!registered) {
            registered = new unsigned char[depth->height * depth->width * rgb->bytes_per_pixel];
        }
        registration->apply(rgb, depth, registered);
        cv::imshow("rgb", color_mat);
        cv::imshow("ir", ir_mat / 20000.0f);
        cv::imshow("depth", depth_mat / 4500.0f);
        cv::imshow("registered", cv::Mat(depth->height, depth->width, CV_8UC3, registered));
        */

        cv::Mat registered_color_depth;
        reg.register_images(color_mat, depth_mat, registered_color_depth);
        double min, max;
        cv::minMaxLoc(registered_color_depth, &min, &max);
        std::cout << "min " << min << " max " << max << std::endl;
        
        cv::Mat depthDisp;
        dispDepth(registered_color_depth, depthDisp, 12000.0f);
        
        cv::flip(color_mat, color_mat, 1);
        cv::Mat combined;
        combine(color_mat, depthDisp, combined);
        cv::line(combined, cv::Point(combined.cols/2, 0), cv::Point(combined.cols/2, combined.rows - 1), cv::Scalar(0, 0, 255));
        cv::line(combined, cv::Point(0, combined.rows/2), cv::Point(combined.cols - 1, combined.rows/2), cv::Scalar(0, 255, 0));
        {
            std::ostringstream oss;
            oss << registered_color_depth.at<uint16_t>(combined.rows/2, combined.cols/2);
            int fontFace =  cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
            double fontScale = 2;
            int thickness = 3;

            int baseline = 0;
            cv::Size textSize = cv::getTextSize(oss.str(), fontFace, fontScale, thickness, &baseline);
            cv::Point textOrg((combined.cols - textSize.width)/2, (combined.rows + textSize.height)/2);
            putText(combined, oss.str(), textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);
        }
        
        cv::imshow("Display window", combined);

        int key = cv::waitKey(1);
        protonect_shutdown = protonect_shutdown || (key > 0 && ((key & 0xFF) == 27)); // shutdown on escape

        listener.release(frames);
        //libfreenect2::this_thread::sleep_for(libfreenect2::chrono::milliseconds(100));
    }

    // TODO: restarting ir stream doesn't work!
    // TODO: bad things will happen, if frame listeners are freed before dev->stop() :(
    dev->stop();
    dev->close();

    delete[] registered;
    delete registration;

    return 0;
}
