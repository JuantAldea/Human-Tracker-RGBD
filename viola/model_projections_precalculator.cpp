#include <string>
#include <map>
#include <fstream>
#include <tuple>

#include "ellipse_functions.h"
#include "ImageRegistration.h"
#include "geometry_helpers.h"


//#include <mrpt/otherlibs/do_opencv_includes.h>
#include <opencv2/opencv.hpp>
/*
#include <Eigen/Sparse>
#include <Eigen/Core>
*/

#ifndef EIGEN_BOOST_SERIALIZATION
#define EIGEN_BOOST_SERIALIZATION

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/map.hpp>

namespace boost
{
namespace serialization
{

template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void save(Archive & ar, const
          Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & m,
          const unsigned int __attribute__((unused)) version)
{
    int rows = m.rows(), cols = m.cols();
    ar & rows;
    ar & cols;
    ar & boost::serialization::make_array(m.data(), rows * cols);
}

template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void load(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & m,
          const unsigned int __attribute__((unused)) version)
{
    int rows, cols;
    ar & rows;
    ar & cols;
    m.resize(rows, cols);
    ar & boost::serialization::make_array(m.data(), rows * cols);
}

template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void serialize(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>
               & m, const unsigned int __attribute__((unused)) version)
{
    split_free(ar, m, version);
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

}
}


#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>

BOOST_SERIALIZATION_SPLIT_FREE(cv::KeyPoint)
namespace boost
{
namespace serialization
{

/** Serialization support for cv::KeyPoint */
template<class Archive>
void save(Archive &ar, const cv::KeyPoint &p,
          const unsigned int __attribute__((unused)) version)
{
    ar & p.pt.x;
    ar & p.pt.y;
    ar & p.size;
    ar & p.angle;
    ar & p.response;
    ar & p.octave;
    ar & p.class_id;
}

/** Serialization support for cv::KeyPoint */
template<class Archive>
void load(Archive &ar, cv::KeyPoint &p, const unsigned int __attribute__((unused)) version)
{
    ar & p.pt.x;
    ar & p.pt.y;
    ar & p.size;
    ar & p.angle;
    ar & p.response;
    ar & p.octave;
    ar & p.class_id;
}
}

}

BOOST_SERIALIZATION_SPLIT_FREE(cv::Mat)
namespace boost
{
namespace serialization
{

/** Serialization support for cv::Mat */
template<class Archive>
void save(Archive &ar, const cv::Mat &m, const unsigned int __attribute__((unused)) version)
{
    size_t elem_size = m.elemSize();
    size_t elem_type = m.type();

    ar & m.cols;
    ar & m.rows;
    ar & elem_size;
    ar & elem_type;

    const size_t data_size = m.cols * m.rows * elem_size;
    ar & boost::serialization::make_array(m.ptr(), data_size);
}

/** Serialization support for cv::Mat */
template<class Archive>
void load(Archive &ar, cv::Mat &m, const unsigned int __attribute__((unused)) version)
{
    int    cols, rows;
    size_t elem_size, elem_type;

    ar & cols;
    ar & rows;
    ar & elem_size;
    ar & elem_type;

    m.create(rows, cols, elem_type);

    size_t data_size = m.cols * m.rows * elem_size;
    ar & boost::serialization::make_array(m.ptr(), data_size);
}

}
}

/*
Copyright 2011 Christopher Allen Ogden. All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:
   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.
   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.
THIS SOFTWARE IS PROVIDED BY CHRISTOPHER ALLEN OGDEN ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL CHRISTOPHER ALLEN OGDEN OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Christopher Allen Ogden.
*/

namespace boost
{
namespace serialization
{

template<uint N>
struct Serialize {
    template<class Archive, typename... Args>
    static void serialize(Archive & ar, std::tuple<Args...> &t, const unsigned int version)
    {
        ar & std::get < N - 1 > (t);
        Serialize < N - 1 >::serialize(ar, t, version);
    }
};

template<>
struct Serialize<0> {
    template<class Archive, typename... Args>
    static void serialize(Archive  __attribute__((unused)) &ar,
                          std::tuple<Args...>  __attribute__((unused)) &t,
                          const unsigned int __attribute__((unused)) version)
    {
        ;
    }
};

template<class Archive, typename... Args>
void serialize(Archive &ar, std::tuple<Args...> &t, const unsigned int version)
{
    Serialize<sizeof...(Args)>::serialize(ar, t, version);
}

}
}


#endif




using oarchive = boost::archive::binary_oarchive;
using iarchive = boost::archive::binary_iarchive;

std::string serial = "013572345247";

using MaskData = std::tuple<cv::Mat, cv::Mat, cv::Mat, int>;

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
    std::map<std::string, std::tuple<std::list<int>, MaskData>> mask;
    std::cout << "#SIZE: " << X_SEMI_AXIS_METTERS << ' ' << Y_SEMI_AXIS_METTERS << std::endl;
    char filename[20];
    sprintf(filename, "ellipses_%fx%f.bin", atof(argv[1]), atof(argv[2]));
    char size[20];

    //for (int cxx = 0; cxx < 1920; cxx+=1)
    //    for (int cyx = 0; cyx < 1080; cyx+=1)

    for (int depth = 0; depth < 5000; depth += 1) {
        Eigen::Vector2i model_length;

        {
            Eigen::Vector2i top_corner, bottom_corner;
            std::tie(top_corner, bottom_corner) = project_model(Eigen::Vector2f(cx, cy), depth,
                                                  Eigen::Vector2f(X_SEMI_AXIS_METTERS, Y_SEMI_AXIS_METTERS),
                                                  reg.cameraMatrixColor, reg.lookupX, reg.lookupY);
            model_length = bottom_corner - top_corner;

            //sprintf(size, "%d %d %d %d", cxx, cyx, model_length[0], model_length[1]);
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

        std::list<int> depths;
        Eigen::Vector2i lengths;
        std::tie(depths, lengths) = ellipses[std::string(size)];
        depths.push_back(depth);
        ellipses[std::string(size)] = std::make_tuple(depths, model_length);

        int n_pixels;
        cv::Mat e1d = fast_create_ellipse_mask(cv::Rect(0, 0, model_length[0], model_length[1]), 1,
                                               n_pixels);

        cv::Mat e3d = fast_create_ellipse_mask(cv::Rect(0, 0, model_length[0], model_length[1]), 3,
                                               n_pixels);

        cv::Mat ew = create_ellipse_weight_mask(e1d);
        //e1d *= 255;
        mask[std::string(size)] = std::make_tuple(depths, std::make_tuple(e1d, e3d, ew, n_pixels));
        //std::cout << depth << ' ' << model_length[0] << ' ' << model_length[1] << std::endl;
    }

    std::list<Eigen::Vector2i> e;
    std::list<std::tuple<cv::Mat, cv::Mat, cv::Mat, int>> mats;



    std::map<int, Eigen::Vector2i> d_e;
    std::map<int, Eigen::Vector2i> d_e_2;
    std::map<int, MaskData> d_m_2;
    std::map<int, MaskData> d_m;

    for (typename decltype(ellipses)::iterator it = ellipses.begin(); it != ellipses.end(); ++it) {
        Eigen::Vector2i model_length;
        std::list<int> depths;
        std::tie(depths, model_length) = it->second;
        e.push_back(model_length);
        Eigen::Vector2i *v = &e.back();
        for (typename decltype(depths)::iterator it = depths.begin(); it != depths.end(); ++it) {
            d_e[*it] = *v;
        }
    }

    for (typename decltype(mask)::iterator it = mask.begin(); it != mask.end(); ++it) {
        MaskData m;
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
        ar & d_e;
        ar & d_m;
        ofs.close();
    }

    {
        std::ifstream ifs(filename);
        iarchive ar(ifs);
        ar & d_e_2;
        ar & d_m_2;
        ifs.close();
    }

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


    for (typename decltype(d_m)::iterator it = d_m.begin(); it != d_m.end(); ++it) {
        const MaskData &a = d_m_2[it->first];
        const MaskData &b = it->second;
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
        const MaskData &a = d_m[it->first];
        const MaskData &b = it->second;
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

