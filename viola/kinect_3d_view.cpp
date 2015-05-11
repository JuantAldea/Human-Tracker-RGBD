#include <signal.h>

#include <cstdlib>

#include "project_config.h"

IGNORE_WARNINGS_PUSH

#include <mrpt/otherlibs/do_opencv_includes.h>

#include <mrpt/gui/CDisplayWindow3D.h>

#include <mrpt/maps/CColouredPointsMap.h>
#include <mrpt/opengl/CGridPlaneXY.h>
#include <mrpt/opengl/stock_objects.h>
#include <mrpt/opengl/CPointCloudColoured.h>
#include <mrpt/base/include/mrpt/system/threads.h>

IGNORE_WARNINGS_POP

#include "KinectCamera.h"
#include "geometry_helpers.h"

using namespace mrpt;
using namespace mrpt::math;
using namespace mrpt::gui;
using namespace mrpt::obs;
using namespace mrpt::maps;
using namespace mrpt::utils;
using namespace mrpt::opengl;
using namespace std;



void kinect_3d_view()
{
    KinectCamera camera;
    camera.open();
    camera.grabFrames();

    mrpt::gui::CDisplayWindow3D win3D("Kinect 3D view", 800, 600);

    win3D.setCameraAzimuthDeg(0);
    win3D.setCameraElevationDeg(0);
    win3D.setCameraZoom(1);
    win3D.setFOV(90);
    win3D.setCameraPointingToPoint(1, 0, 0);

    mrpt::opengl::CPointCloudColouredPtr gl_points = mrpt::opengl::CPointCloudColoured::Create();
    gl_points->setPointSize(0.5);

    opengl::COpenGLViewportPtr viewRange, viewInt; // Extra viewports for the RGB & D images.

    {
        mrpt::opengl::COpenGLScenePtr &scene = win3D.get3DSceneAndLock();
        scene->insert(gl_points);
        scene->insert(mrpt::opengl::CGridPlaneXY::Create());
        scene->insert(mrpt::opengl::stock_objects::CornerXYZ());
        win3D.unlockAccess3DScene();
        win3D.repaint();
    }


    const KinectCamera::IRCameraParams &params = camera.getIRCameraParams();
    size_t depth_rows, depth_cols;
    std::tie(depth_cols, depth_rows) = KinectCamera::getFrameSize(KinectCamera::FrameType::DEPTH);
    
    cv::Mat reprojection = cv::Mat(cv::Size(depth_cols, depth_rows), CV_32FC3);    
    CColouredPointsMap pntsMap;
    pntsMap.colorScheme.scheme = CColouredPointsMap::cmFromHeightRelativeToSensor;

    while (win3D.isOpen()) {
        camera.grabFrames();
        cv::Mat depth = camera.frames[KinectCamera::FrameType::DEPTH];
        //cv::Mat color = camera.frames[KinectCamera::FrameType::COLOR];
        //cv::Mat scaled_color;
        //cv::resize(color, scaled_color,depth.size());
        //cout << "DEPTH " << depth.rows << ' ' << depth.cols << endl;
        //cout << "REPRO " << reprojection.rows << ' ' << reprojection.cols << endl;
        //cout << "COLOR " << color.rows << ' ' << color.cols << endl;
        
        reprojection = depth_3D_reprojection(depth, 1.f/params.fx, 1.f/params.fy, params.cx, params.cy);
        pntsMap.clear();
        for (int x = 0; x < reprojection.rows; x++) {
            for (int y = 0; y < reprojection.cols; y++) {
                //cv::Vect3b color = scaled_color.at<cv::Vec3b>(x, y);
                cv::Vec3f &v = reprojection.at<cv::Vec3f>(x, y);
                pntsMap.insertPoint(v[1], -v[2], -v[0]);
            }
        }

        win3D.get3DSceneAndLock();
        gl_points->loadFromPointsMap(&pntsMap);
        win3D.unlockAccess3DScene();

        win3D.repaint();
        //mrpt::system::sleep(1);
    }
}

//int main(int argc, char *argv[])
int main()
{
    kinect_3d_view();

    return 0;

    /*
    try {
        kinect_3d_view();
        return 0;
    } catch (std::exception &e) {
        std::cout << "MRPT exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        printf("Untyped exception!!");
        return -1;
    }
    */
}

