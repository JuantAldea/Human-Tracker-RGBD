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
    KinectCamera camera;
    camera.open();
    camera.grabFrames();
    KinectCamera::IRCameraParams params = camera.getIRCameraParams();
    while (win3D.isOpen()) {
        camera.grabFrames();
        cv::Mat depth = camera.frames[KinectCamera::FrameType::DEPTH];
        cv::Mat color = camera.frames[KinectCamera::FrameType::COLOR];
        cv::Mat scaled_color;
        cv::resize(color, scaled_color,depth.size());
        cout << depth.rows << ' ' << depth.cols << endl;
        cout << color.rows << ' ' << color.cols << endl;
        CColouredPointsMap pntsMap;
        pntsMap.colorScheme.scheme = CColouredPointsMap::cmFromHeightRelativeToSensor;
        float inv_fx = 1.f / params.fx;
        float inv_fy = 1.f / params.fy;
        for (int x = 0; x < depth.rows; x++) {
            for (int y = 0; y < depth.cols; y++) {
                //cv::Vect3b color = scaled_color.at<cv::Vec3b>(x, y);
                float v_z = depth.at<float>(x, y) / 1000.f;
                float v_x = ((x - params.cx) * v_z) * inv_fx;
                float v_y = ((y - params.cy) * v_z) * inv_fy;
                //std::cout << v_x << ' ' << v_y << ' ' << v_z << std::endl;
                pntsMap.insertPoint(v_y, -v_z, -v_x);
            }

        }

        win3D.get3DSceneAndLock();
        gl_points->loadFromPointsMap(&pntsMap);
        win3D.unlockAccess3DScene();

        win3D.repaint();
        mrpt::system::sleep(1);
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

