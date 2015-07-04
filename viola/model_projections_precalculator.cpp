#include "ellipse_functions.h"
#include "ImageRegistration.h"
#include "geometry_helpers.h"

#include "BoostSerializers.h"

#include <string>
#include <map>
#include <fstream>
#include <tuple>
#include <opencv2/opencv.hpp>
/*
#include <Eigen/Sparse>
#include <Eigen/Core>
*/
//#include <mrpt/otherlibs/do_opencv_includes.h>


using oarchive = boost::archive::binary_oarchive;
using iarchive = boost::archive::binary_iarchive;

std::string serial = "013572345247";


using EllipseData = std::tuple<cv::Mat, cv::Mat, cv::Mat, int>;
using EllipseDepthMap = std::map<int, EllipseData>;
int main(int argc, char *argv[])
{
    (void)(argc);
    (void)(argv);

    char *calib_dir = getenv("HOME");
    std::string calib_path = std::string(calib_dir) + "/kinect2_calib/";
    ImageRegistration reg;
    reg.init(calib_path, serial);

    float cx = reg.cameraMatrixColor.at<double>(0, 2);
    float cy = reg.cameraMatrixColor.at<double>(1, 2);

    //const float cx = 1000 * 0.5;
    //const float cy = 500 * 0.5;

    //std::cout << cx << ' ' << cy << ' ' << cxr << ' ' << cyr << std::endl;
    //exit(-1);
    const float X_SEMI_AXIS_METTERS = atof(argv[1]) * 0.5;
    const float Y_SEMI_AXIS_METTERS = atof(argv[2]) * 0.5;

    std::map<std::string, std::tuple<std::list<int>, Eigen::Vector2i>> ellipses;
    std::map<std::string, std::tuple<std::list<int>, EllipseData>> mask;
    std::cout << "#SIZE: " << X_SEMI_AXIS_METTERS << ' ' << Y_SEMI_AXIS_METTERS << std::endl;
    char filename[20];
    sprintf(filename, "ellipses_%fx%f.bin", atof(argv[1]), atof(argv[2]));
    char size[20];

    //for (int cxx = 0; cxx < 1920; cxx+=1)
    //    for (int cyx = 0; cyx < 1080; cyx+=1)
    std::map<int, std::string> depth_size;
    std::map<std::string, EllipseData> string_ellipse;

    for (int depth = 0; depth < 5000; depth += 1) {
        Eigen::Vector2i model_length;

        {
            Eigen::Vector2i top_corner, bottom_corner;
            std::tie(top_corner, bottom_corner) = project_model(Eigen::Vector2f(cx, cy), depth,
                                                  Eigen::Vector2f(X_SEMI_AXIS_METTERS, Y_SEMI_AXIS_METTERS),
                                                  reg.cameraMatrixColor, reg.lookupX, reg.lookupY);
            model_length = bottom_corner - top_corner;

            sprintf(size, "%d %d", model_length[0], model_length[1]);
            //std::cout << depth  << " " <<  size << std::endl;
        }

        /*
        Eigen::Vector2i model_length2;
        {
            Eigen::Vector2i top_corner, bottom_corner;
            std::tie(top_corner, bottom_corner) = project_model(Eigen::Vector2f(cxx, cyx), depth,
                Eigen::Vector2f(X_SEMI_AXIS_METTERS, Y_SEMI_AXIS_METTERS),
                reg.cameraMatrixColor, reg.lookupX, reg.lookupY);
            model_length2 = bottom_corner - top_corner;

            //sprintf(size, "%d %d %d %d", cxx, cyx, model_length2[0], model_length2[1]);
            //std::cout << depth  << " " <<  size << std::endl;
        }

        if (model_length[0] && model_length[1] && model_length2[0] && model_length2[1]){
            if ((model_length[0] - model_length2[0]) || model_length[1] - model_length2[1]){
                std::cout <<"DIFF " << model_length[0] - model_length2[0] << std::endl;
                std::cout <<"DIFF " << model_length[1] - model_length2[1] << std::endl;
            }
        }
        */

        int n_pixels;



        std::list<int> depths;
        Eigen::Vector2i lengths;
        std::tie(depths, lengths) = ellipses[std::string(size)];
        depths.push_back(depth);

        ellipses[std::string(size)] = std::make_tuple(depths, model_length);
        cv::Mat e3d = fast_create_ellipse_mask(cv::Rect(0, 0, model_length[0], model_length[1]), 3, n_pixels);
        cv::Mat e1d = cv::Mat(e3d.rows, e3d.cols, CV_8UC1);
        int from_to[] = {0,0};
        cv::mixChannels(&e3d, 1, &e1d, 1, from_to, 1);
        cv::Mat ew1d = create_ellipse_weight_mask(e1d);

        //e1d *= 255;
        mask[std::string(size)] = std::make_tuple(depths, std::make_tuple(e1d, e3d, ew1d, n_pixels));
        //std::cout << depth << ' ' << model_length[0] << ' ' << model_length[1] << std::endl;

        depth_size[depth] = std::string(size);
        string_ellipse[std::string(size)] = std::make_tuple(e1d, e3d, ew1d, n_pixels);
    }

    std::map<int, EllipseData*> depth_data;
    for (int depth = 0; depth < 5000; depth++){
        depth_data[depth] = &string_ellipse[depth_size[depth]];
    }

    std::list<Eigen::Vector2i> e;
    std::list<EllipseData> mats;

    std::map<int, Eigen::Vector2i> d_e;
    std::map<int, Eigen::Vector2i> d_e_2;

    EllipseDepthMap d_m_2;
    EllipseDepthMap d_m;

    /*
    for (typename decltype(ellipses)::iterator it = ellipses.begin(); it != ellipses.end(); ++it) {
        Eigen::Vector2i model_length;
        std::list<int> depths;
        std::tie(depths, model_length) = it->second;
        e.push_back(model_length);
        for (typename decltype(depths)::iterator it = depths.begin(); it != depths.end(); ++it) {
            d_e[*it] = e.back();
        }
    }
    */

    for (typename decltype(mask)::iterator it = mask.begin(); it != mask.end(); ++it) {
        EllipseData m;
        std::list<int> depths;
        std::tie(depths, m) = it->second;
        mats.push_back(m);
        for (typename decltype(depths)::iterator it = depths.begin(); it != depths.end(); ++it) {
            d_m[*it] = mats.back();
        }
    }

    /*
    for (typename decltype(d_e)::iterator it=d_e.begin(); it!=d_e.end(); ++it){
        std::cout << it->first << "=>" << (it->second)[0] << ' ' <<  (it->second)[1] << std::endl;
    }
    */

    {
        std::ofstream ofs(filename);
        oarchive ar(ofs);
        //ar & d_e;
        ar & d_m;
        ofs.close();
    }

    {
        std::ifstream ifs(filename);
        iarchive ar(ifs);
        //ar & d_e_2;
        ar & d_m_2;
        ifs.close();
    }

    /*
    for (typename decltype(d_e)::iterator it = d_e.begin(); it != d_e.end(); ++it) {
        if ((d_e_2[it->first]) != (it->second)) {
            std::cout << " ERROR E1" << std::endl;
            exit(-1);
        }
    }

    for (typename decltype(d_e_2)::iterator it = d_e_2.begin(); it != d_e_2.end(); ++it) {
        if ((d_e[it->first]) != (it->second)) {
            std::cout << " ERROR E2" << std::endl;
            exit(-1);
        }
    }
    */


    for (typename decltype(d_m)::iterator it = d_m.begin(); it != d_m.end(); ++it) {
        const EllipseData &a = d_m_2[it->first];
        const EllipseData &b = it->second;
        cv::Mat a1, a3, aw;
        cv::Mat b1, b3, bw;
        int  apix, bpix;
        std::tie(a1, a3, aw, apix) = a;
        std::tie(b1, b3, bw, bpix) = b;
        bool test = true;
        test &= (cv::norm(a1, b1) == 0);
        test &= (cv::norm(a3, b3) == 0);
        test &= (cv::norm(aw, bw) == 0);
        test &= apix == bpix;

        if (!test) {
            std::cout << " ERROR M1" << std::endl;
            exit(-1);
        }
    }

    for (typename decltype(d_m_2)::iterator it = d_m_2.begin(); it != d_m_2.end(); ++it) {
        const EllipseData &a = d_m[it->first];
        const EllipseData &b = it->second;
        cv::Mat a1, a3, aw;
        cv::Mat b1, b3, bw;
        int  apix, bpix;
        std::tie(a1, a3, aw, apix) = a;
        std::tie(b1, b3, bw, bpix) = b;
        bool test = true;
        test &= (cv::norm(a1, b1) == 0);
        test &= (cv::norm(a3, b3) == 0);
        test &= (cv::norm(aw, bw) == 0);
        test &= apix == bpix;
        if (!test) {
            std::cout << " ERROR M2" << std::endl;
            exit(-1);
        }
    }
    std::cout << "OK" << std::endl;

    //cv::namedWindow("Original Image", 1);
    //cv::namedWindow("New Image", 1);

    //cv::imshow("Original Image", d_m_2[800]);
    //cv::imshow("New Image", d_m[800]);


    //cv::waitKey(0);
}

