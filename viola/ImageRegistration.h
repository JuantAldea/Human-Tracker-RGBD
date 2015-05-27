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
    cv::Mat cameraMatrixColor, distortionColor, cameraMatrixIr, distortionIr;
    cv::Mat rotation, translation;
    cv::Mat map1Color, map2Color, map1Ir, map2Ir;
    cv::Mat lookupX, lookupY;
    double depthShift;
    double maxDepth;
    DepthRegistration *depthRegHighRes;

public:
    ImageRegistration();
    ~ImageRegistration();
    void init(const std::string &calib_path, const std::string &sensor);
    void initCalibration(const std::string &calib_path, const std::string &sensor);
    bool loadCalibrationFile(const std::string &filename, cv::Mat &cameraMatrix, cv::Mat &distortion) const;
    bool loadCalibrationPoseFile(const std::string &filename, cv::Mat &rotation, cv::Mat &translation) const;
    bool loadCalibrationDepthFile(const std::string &filename, double &depthShift) const;
    void register_images(const cv::Mat &color, const cv::Mat &ir_depth, cv::Mat &out) const;
    void createLookup();
};
