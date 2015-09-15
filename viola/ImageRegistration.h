#pragma once

#include <sys/stat.h>
#include <string>
#include <opencv2/opencv.hpp>

#include <depth_registration.h>

#define K2_DEFAULT_NS          "kinect2"
#define K2_CALIB_COLOR         "calib_color.yaml"
#define K2_CALIB_IR            "calib_ir.yaml"
#define K2_CALIB_POSE          "calib_pose.yaml"
#define K2_CALIB_DEPTH         "calib_depth.yaml"

#define K2_CALIB_CAMERA_MATRIX "cameraMatrix"
#define K2_CALIB_DISTORTION    "distortionCoefficients"
#define K2_CALIB_ROTATION      "rotation"
#define K2_CALIB_PROJECTION    "projection"
#define K2_CALIB_TRANSLATION   "translation"
#define K2_CALIB_ESSENTIAL     "essential"
#define K2_CALIB_FUNDAMENTAL   "fundamental"
#define K2_CALIB_DEPTH_SHIFT   "depthShift"

class ImageRegistration
{
public:
    cv::Size sizeColor;
    cv::Size sizeIr;
    cv::Size sizeLowRes;
    cv::Mat cameraMatrixColor, distortionColor, cameraMatrixIr, distortionIr;
    cv::Mat cameraMatrixLowRes;
    cv::Mat cameraMatrix;
    cv::Mat rotation, translation;
    cv::Mat map1Color, map2Color, map1Ir, map2Ir, map1LowRes, map2LowRes;
    cv::Mat lookupX, lookupY;
    double depthShift;
    double maxDepth;
    DepthRegistration *depthRegHighRes;
    DepthRegistration *depthRegLowRes;

public:
    ImageRegistration();
    ~ImageRegistration();
    void init(const std::string &calib_path, const std::string &sensor);
    void initCalibration(const std::string &calib_path, const std::string &sensor);
    bool loadCalibrationFile(const std::string &filename, cv::Mat &cameraMatrix, cv::Mat &distortion) const;
    bool loadCalibrationPoseFile(const std::string &filename, cv::Mat &rotation, cv::Mat &translation) const;
    bool loadCalibrationDepthFile(const std::string &filename, double &depthShift) const;

    void createLookup(const size_t width, const size_t height, const cv::Mat cameraMatrix);

    void register_images(const cv::Mat &color, const cv::Mat &ir_depth, cv::Mat &color_out, cv::Mat &ir_depth_out) const;
    void register_ir(const cv::Mat &ir_depth, cv::Mat &ir_out) const;
    void register_color(const cv::Mat &color, cv::Mat &color_out) const;
};
