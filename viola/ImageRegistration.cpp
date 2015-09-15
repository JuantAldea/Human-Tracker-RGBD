
#include "ImageRegistration.h"

ImageRegistration::ImageRegistration() :
    sizeColor(1920, 1080), sizeIr(512, 424),
    sizeLowRes(sizeColor.width / 2, sizeColor.height / 2),
    lookupX(cv::Mat(1, sizeColor.width, CV_32F)),
    lookupY(cv::Mat(1, sizeColor.height, CV_32F)),
    depthShift(0), maxDepth(12.0),
    depthRegHighRes(DepthRegistration::New(DepthRegistration::OPENCL)),
    depthRegLowRes(DepthRegistration::New(DepthRegistration::OPENCL))
{
    ;
}

ImageRegistration::~ImageRegistration()
{
    //delete depthRegHighRes;
}

void ImageRegistration::init(const std::string &calib_path, const std::string &sensor)
{
    initCalibration(calib_path, sensor);
}

void ImageRegistration::initCalibration(const std::string &calib_path, const std::string &sensor)
{
    std::string calibPath = calib_path + sensor + '/';
    std::cout << "Looking for camera parameters in folder: " << calibPath.c_str() << std::endl;
    struct stat fileStat;

    bool calibDirNotFound = stat(calibPath.c_str(), &fileStat) != 0 || !S_ISDIR(fileStat.st_mode);

    if (calibDirNotFound || !loadCalibrationFile(calibPath + K2_CALIB_COLOR, cameraMatrixColor, distortionColor)) {
        std::cerr << "using sensor defaults for color intrinsic parameters." << std::endl;
    }

    if (calibDirNotFound || !loadCalibrationFile(calibPath + K2_CALIB_IR, cameraMatrixIr, distortionIr)) {
        std::cerr << "using sensor defaults for ir intrinsic parameters." << std::endl;
    }

    if (calibDirNotFound || !loadCalibrationPoseFile(calibPath + K2_CALIB_POSE, rotation, translation)) {
        std::cerr << "using defaults for rotation and translation." << std::endl;
    }

    if (calibDirNotFound || !loadCalibrationDepthFile(calibPath + K2_CALIB_DEPTH, depthShift)) {
        std::cerr << "using defaults for depth shift." << std::endl;
        depthShift = 0.0;
    }

    cameraMatrixLowRes = cameraMatrixColor.clone();
    cameraMatrixLowRes.at<double>(0, 0) /= 2;
    cameraMatrixLowRes.at<double>(1, 1) /= 2;
    cameraMatrixLowRes.at<double>(0, 2) /= 2;
    cameraMatrixLowRes.at<double>(1, 2) /= 2;

    depthRegHighRes->init(cameraMatrixColor, sizeColor, cameraMatrixIr, sizeIr, distortionIr, rotation, translation, 0.5f, maxDepth, -1);
    depthRegLowRes->init(cameraMatrixLowRes, sizeLowRes, cameraMatrixIr, sizeIr, distortionIr, rotation, translation, 0.5f, maxDepth, -1);

    const int mapType = CV_16SC2;
    cv::initUndistortRectifyMap(cameraMatrixColor, distortionColor, cv::Mat(), cameraMatrixColor, sizeColor, mapType, map1Color, map2Color);
    cv::initUndistortRectifyMap(cameraMatrixColor, distortionColor, cv::Mat(), cameraMatrixLowRes, sizeLowRes, mapType, map1LowRes, map2LowRes);
    cv::initUndistortRectifyMap(cameraMatrixIr, distortionIr, cv::Mat(), cameraMatrixIr, sizeIr, mapType, map1Ir, map2Ir);

    createLookup(sizeColor.width, sizeColor.height, cameraMatrixColor);

    std::cout << std::endl
              << "Camera parameters:" << std::endl
              << "camera matrix color:" << std::endl << cameraMatrixColor << std::endl
              << "distortion coefficients color:" << std::endl << distortionColor << std::endl
              << "camera matrix ir:" << std::endl << cameraMatrixIr << std::endl
              << "distortion coefficients ir:" << std::endl << distortionIr << std::endl
              << "rotation:" << std::endl << rotation << std::endl
              << "translation:" << std::endl << translation << std::endl
              << "depth shift:" << std::endl << depthShift << std::endl
              << std::endl;
}

bool ImageRegistration::loadCalibrationFile(const std::string &filename, cv::Mat &cameraMatrix, cv::Mat &distortion) const
{
    cv::FileStorage fs;
    if (fs.open(filename, cv::FileStorage::READ)) {
        fs[K2_CALIB_CAMERA_MATRIX] >> cameraMatrix;
        fs[K2_CALIB_DISTORTION] >> distortion;
        fs.release();
    } else {
        std::cerr << "can't open calibration file: " << filename << std::endl;
        return false;
    }
    return true;
}

bool ImageRegistration::loadCalibrationPoseFile(const std::string &filename, cv::Mat &rotation, cv::Mat &translation) const
{
    cv::FileStorage fs;
    if (fs.open(filename, cv::FileStorage::READ)) {
        fs[K2_CALIB_ROTATION] >> rotation;
        fs[K2_CALIB_TRANSLATION] >> translation;
        fs.release();
    } else {
        std::cerr << "can't open calibration pose file: " << filename << std::endl;
        return false;
    }
    return true;
}

bool ImageRegistration::loadCalibrationDepthFile(const std::string &filename, double &depthShift) const
{
    cv::FileStorage fs;
    if (fs.open(filename, cv::FileStorage::READ)) {
        fs[K2_CALIB_DEPTH_SHIFT] >> depthShift;
        fs.release();
    } else {
        std::cerr << "can't open calibration depth file: " << filename << std::endl;
        return false;
    }
    return true;
}

void ImageRegistration::register_images(const cv::Mat &color, const cv::Mat &ir_depth, cv::Mat &color_out, cv::Mat &ir_depth_out) const
{
    register_color(color, color_out);
    register_ir(ir_depth, ir_depth_out);
}

void ImageRegistration::register_color(const cv::Mat &color, cv::Mat &color_out) const
{
    cv::Mat color_flipped;
    cv::flip(color, color_flipped, 1);
    cv::remap(color_flipped, color_out, map1Color, map2Color, cv::INTER_AREA);
}

void ImageRegistration::register_ir(const cv::Mat &ir_depth, cv::Mat &ir_out) const
{
    cv::Mat ir_depth_shifted;
    ir_depth.convertTo(ir_depth_shifted, CV_16U, 1, depthShift);
    cv::flip(ir_depth_shifted, ir_depth_shifted, 1);
    //cv::Mat depth_shifted_rect;
    //cv::remap(ir_depth_shifted, depth_shifted_rect, map1Ir, map2Ir, cv::INTER_NEAREST);
    depthRegHighRes->registerDepth(ir_depth_shifted, ir_out);
}

/*
void ImageRegistration::createLookup()
{
    const size_t width = sizeColor.width;
    const size_t height = sizeColor.height;
    const float inv_fx = 1.0f / cameraMatrixColor.at<double>(0, 0);
    const float inv_fy = 1.0f / cameraMatrixColor.at<double>(1, 1);
    const float cx = cameraMatrixColor.at<double>(0, 2);
    const float cy = cameraMatrixColor.at<double>(1, 2);

    float *it = lookupY.ptr<float>();
    for (size_t r = 0; r < height; ++r, ++it) {
        *it = (r - cy) * inv_fy;
    }

    it = lookupX.ptr<float>();
    for (size_t c = 0; c < width; ++c, ++it) {
        *it = (c - cx) * inv_fx;
    }
}
*/

void ImageRegistration::createLookup(const size_t width, const size_t height, const cv::Mat camera)
{
    cameraMatrix = camera;
    const float inv_fx = 1.0f / cameraMatrix.at<double>(0, 0);
    const float inv_fy = 1.0f / cameraMatrix.at<double>(1, 1);
    const float cx = cameraMatrix.at<double>(0, 2);
    const float cy = cameraMatrix.at<double>(1, 2);

    lookupY = cv::Mat(1, height, CV_32F);
    lookupX = cv::Mat(1, width, CV_32F);

    float *it = lookupY.ptr<float>();
    for (size_t r = 0; r < height; ++r, ++it) {
        *it = (r - cy) * inv_fy;
    }

    it = lookupX.ptr<float>();
    for (size_t c = 0; c < width; ++c, ++it) {
        *it = (c - cx) * inv_fx;
    }
}

/*
void createCloud(const cv::Mat &color, const cv::Mat &depth, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud) const
{
    const float badPoint = std::numeric_limits<float>::quiet_NaN();

    #pragma omp parallel for
    for (int r = 0; r < depth.rows; ++r) {
        pcl::PointXYZRGBA*itP = &cloud->points[r * depth.cols];
        const uint16_t *itD = depth.ptr<uint16_t>(r);
        const cv::Vec3b *itC = color.ptr<cv::Vec3b>(r);
        const float y = lookupY.at<float>(0, r);
        const float *itX = lookupX.ptr<float>();

        for (size_t c = 0; c < (size_t)depth.cols; ++c, ++itP, ++itD, ++itC, ++itX) {
            register const float depthValue = *itD / 1000.0f;
            // Check for invalid measurements
            if (isnan(depthValue) || depthValue <= 0.001) {
                // not valid
                itP->x = itP->y = itP->z = badPoint;
                itP->rgb = 0;
                continue;
            }
            itP->z = depthValue;
            itP->x = (*itX) * depthValue;
            itP->y = y * depthValue;
            itP->b = itC->val[0];
            itP->g = itC->val[1];
            itP->r = itC->val[2];
            itP->a = 255;
            //std::cout << itP->x << ' ' << itP->y << ' ' << itP->z << std::endl;
        }
    }
}
*/
