#include <string>
#include <map>


#include "model_parameters.h"
#include "ellipse_functions.h"
#include "ImageRegistration.h"
#include "geometry_helpers.h"


#include <fstream>
// include headers that implement a archive in simple text format
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <mrpt/gui/CDisplayWindow3D.h>
#include <mrpt/maps/CColouredPointsMap.h>
#include <mrpt/opengl/CGridPlaneXY.h>
#include <mrpt/opengl/stock_objects.h>
#include <mrpt/opengl/CPointCloudColoured.h>

#include <mrpt/gui/CDisplayWindow.h>
#include <mrpt/random.h>
#include <mrpt/bayes/CParticleFilterData.h>
#include <mrpt/obs/CSensoryFrame.h>
#include <mrpt/obs/CObservationImage.h>
#include <mrpt/otherlibs/do_opencv_includes.h>
using namespace mrpt;
using namespace mrpt::bayes;
using namespace mrpt::gui;
using namespace mrpt::obs;
using namespace mrpt::random;
using namespace mrpt::opengl;
using namespace mrpt::maps;


#ifndef EIGEN_BOOST_SERIALIZATION
#define EIGEN_BOOST_SERIALIZATION
//#include <Eigen/Sparse>
//#include <Eigen/Core>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/map.hpp>

template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void serialize(Archive & ar, Eigen::Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> & m, const unsigned int version) {
}

namespace boost{
namespace serialization{
    template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
        void save(Archive & ar, const Eigen::Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> & m, const unsigned int version) {
            (void)(version);
            int rows=m.rows(),cols=m.cols();
            ar & rows;
            ar & cols;
            ar & boost::serialization::make_array(m.data(), rows*cols);
        }
    template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
        void load(Archive & ar, Eigen::Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> & m, const unsigned int version) {
            (void)(version);
            int rows,cols;
            ar & rows;
            ar & cols;
            m.resize(rows,cols);
            ar & boost::serialization::make_array(m.data(), rows*cols);
        }

    template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
        void serialize(Archive & ar, Eigen::Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> & m, const unsigned int version) {
            (void)(version);
            split_free(ar,m,version);
        }

    /*
    template <class Archive, typename _Scalar>
        void save(Archive & ar, const Eigen::Triplet<_Scalar> & m, const unsigned int version) {
            ar & m.row();
            ar & m.col();
            ar & m.value();
        }
    template <class Archive, typename _Scalar>
        void load(Archive & ar, Eigen::Triplet<_Scalar> & m, const unsigned int version) {
            int row,col;
            _Scalar value;
            ar & row;
            ar & col;
            ar & value;
            m = Eigen::Triplet<_Scalar>(row,col,value);
        }

    template <class Archive, typename _Scalar>
        void serialize(Archive & ar, Eigen::Triplet<_Scalar> & m, const unsigned int version) {
            split_free(ar,m,version);
        }

    template <class Archive, typename _Scalar, int _Options,typename _Index>
        void save(Archive & ar, const Eigen::SparseMatrix<_Scalar,_Options,_Index> & m, const unsigned int version) {
            int innerSize=m.innerSize();
            int outerSize=m.outerSize();
            typedef typename Eigen::Triplet<_Scalar> Triplet;
            std::vector<Triplet> triplets;
            for(int i=0; i < outerSize; ++i) {
                for(typename Eigen::SparseMatrix<_Scalar,_Options,_Index>::InnerIterator it(m,i); it; ++it) {
                triplets.push_back(Triplet(it.row(), it.col(), it.value()));
                }
            }
            ar & innerSize;
            ar & outerSize;
            ar & triplets;
        }
    template <class Archive, typename _Scalar, int _Options, typename _Index>
        void load(Archive & ar, Eigen::SparseMatrix<_Scalar,_Options,_Index>  & m, const unsigned int version) {
            int innerSize;
            int outerSize;
            ar & innerSize;
            ar & outerSize;
            int rows = m.IsRowMajor?outerSize:innerSize;
            int cols = m.IsRowMajor?innerSize:outerSize;
            m.resize(rows,cols);
            typedef typename Eigen::Triplet<_Scalar> Triplet;
            std::vector<Triplet> triplets;
            ar & triplets;
            m.setFromTriplets(triplets.begin(), triplets.end());

        }
    template <class Archive, typename _Scalar, int _Options, typename _Index>
        void serialize(Archive & ar, Eigen::SparseMatrix<_Scalar,_Options,_Index> & m, const unsigned int version) {
            split_free(ar,m,version);
        }
    */

}}
#endif

#define BINARY
#ifdef BINARY
using oarchive = boost::archive::binary_oarchive;
using iarchive = boost::archive::binary_iarchive;
#else
using oarchive = boost::archive::text_oarchive;
using iarchive = boost::archive::text_iarchive;
#endif

std::string serial = "013572345247";

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
    std::map<std::string, std::tuple<std::list<int>, cv::Mat>> mask;
    std::cout << "#SIZE: " << X_SEMI_AXIS_METTERS << ' ' << Y_SEMI_AXIS_METTERS << std::endl;
    char filename[20];
    sprintf(filename, "ellipses_%fx%f.bin", atof(argv[1]), atof(argv[2]));
    char size[20];

    for (int cxx = 0; cxx < 1920; cxx+=1)
        for (int cyx = 0; cyx < 1080; cyx+=1)

    for (int depth = 500; depth < 5000; depth+=1){
        Eigen::Vector2i model_length1;
        Eigen::Vector2i model_length2;

        {
            Eigen::Vector2i top_corner, bottom_corner;
            std::tie(top_corner, bottom_corner) = project_model(Eigen::Vector2f(cx, cy), depth,
                Eigen::Vector2f(X_SEMI_AXIS_METTERS, Y_SEMI_AXIS_METTERS),
                reg.cameraMatrixColor, reg.lookupX, reg.lookupY);
            model_length1 = bottom_corner - top_corner;

            //sprintf(size, "%d %d %d %d", cxx, cyx, model_length1[0], model_length1[1]);
            //std::cout << depth  << " " <<  size << std::endl;
        }

        {
            Eigen::Vector2i top_corner, bottom_corner;
            std::tie(top_corner, bottom_corner) = project_model(Eigen::Vector2f(cxx, cyx), depth,
                Eigen::Vector2f(X_SEMI_AXIS_METTERS, Y_SEMI_AXIS_METTERS),
                reg.cameraMatrixColor, reg.lookupX, reg.lookupY);
            model_length2 = bottom_corner - top_corner;

            //sprintf(size, "%d %d %d %d", cxx, cyx, model_length2[0], model_length2[1]);
            //std::cout << depth  << " " <<  size << std::endl;
        }

        if (model_length1[0] && model_length1[1] && model_length2[0] && model_length2[1]){
            if ((model_length1[0] - model_length2[0]) || model_length1[1] - model_length2[1]){
                std::cout <<"DIFF " << model_length1[0] - model_length2[0] << std::endl;
                std::cout <<"DIFF " << model_length1[1] - model_length2[1] << std::endl;
            }
        }
        /*
        std::list<int> depths;
        Eigen::Vector2i lengths;
        std::tie(depths, lengths) = ellipses[std::string(size)];
        depths.push_back(depth);
        ellipses[std::string(size)] = std::make_tuple(depths, model_length);
        int n_pixels;
        mask[std::string(size)] = std::make_tuple(depths, fast_create_ellipse_mask(cv::Rect(0, 0, model_length[0], model_length[1]), 3, n_pixels));
        //std::cout << depth << ' ' << model_length[0] << ' ' << model_length[1] << std::endl;
        */
    }

    std::list<Eigen::Vector2i> e;
    std::list<cv::Mat> mats;


//#define POINTER

#ifdef POINTER
    std::map<int, Eigen::Vector2i*> d_e;
    std::map<int, Eigen::Vector2i*> d_e_2;
    std::map<int, cv::Mat*> d_m_2;
    std::map<int, cv::Mat*> d_m;
#else
    std::map<int, Eigen::Vector2i> d_e;
    std::map<int, Eigen::Vector2i> d_e_2;
    std::map<int, cv::Mat> d_m_2;
    std::map<int, cv::Mat> d_m;
#endif
    for (typename decltype(ellipses)::iterator it=ellipses.begin(); it!=ellipses.end(); ++it){
        Eigen::Vector2i model_length;
        std::list<int> depths;
        std::tie(depths, model_length) = it->second;
        e.push_back(model_length);
        Eigen::Vector2i *v = &e.back();
        for (typename decltype(depths)::iterator it=depths.begin(); it!=depths.end(); ++it) {
#ifdef POINTER
            d_e[*it] = v;
#else
            d_e[*it] = *v;
#endif
        }
    }

    for (typename decltype(mask)::iterator it=mask.begin(); it!=mask.end(); ++it){
        cv::Mat m;
        std::list<int> depths;
        std::tie(depths, m) = it->second;
        mats.push_back(m);
        cv::Mat *mm = &mats.back();
        for (typename decltype(depths)::iterator it=depths.begin(); it!=depths.end(); ++it) {
#ifdef POINTER
            d_m[*it] = mm;
#else
            d_m[*it] = *mm;
#endif
        }
    }

    for (typename decltype(d_e)::iterator it=d_e.begin(); it!=d_e.end(); ++it){
#ifdef POINTER
        std::cout << it->first << "=>" << (*it->second)[0] << ' ' <<  (*it->second)[1] << std::endl;
#else
        std::cout << it->first << "=>" << (it->second)[0] << ' ' <<  (it->second)[1] << std::endl;
#endif
    }
/*
    {
        std::ofstream ofs(filename);
        oarchive ar(ofs);
        ar & d_e;
        //ar & d_m;
        ofs.close();
    }

    {
        std::ifstream ifs(filename);
        iarchive ar(ifs);
        ar & d_e_2;
        //ar & d_m_2;
        ifs.close();
    }

    for (typename decltype(d_e)::iterator it=d_e.begin(); it!=d_e.end(); ++it){
#ifdef POINTER
        if (*(d_e_2[it->first]) != *(it->second)) {
#else
        if ((d_e_2[it->first]) != (it->second)) {
#endif
            std::cout << " ERROR E1" << std::endl;
            exit(-1);
        }
    }

    for (typename decltype(d_e_2)::iterator it=d_e_2.begin(); it!=d_e_2.end(); ++it){
#ifdef POINTER
        if (*(d_e[it->first]) != *(it->second)) {
#else
        if ((d_e[it->first]) != (it->second)) {
#endif
            std::cout << " ERROR E2" << std::endl;
            exit(-1);
        }
    }
/*
    for (typename decltype(d_m)::iterator it=d_m.begin(); it!=d_m.end(); ++it){
#ifdef POINTER
        if (*(d_m_2[it->first]) != *(it->second)) {
#else
        if ((d_m_2[it->first]) != (it->second)) {
#endif
            std::cout << " ERROR M1" << std::endl;
            exit(-1);
        }
    }

    for (typename decltype(d_m_2)::iterator it=d_m_2.begin(); it!=d_m_2.end(); ++it){
#ifdef POINTER
        if (*(d_e[it->first]) != *(it->second)) {
#else
        if ((d_e[it->first]) != (it->second)) {
#endif
            std::cout << " ERROR M2" << std::endl;
            exit(-1);
        }
    }
*/
    std::cout << "OK" << std::endl;
}


