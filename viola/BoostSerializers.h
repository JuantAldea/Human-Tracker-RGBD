#pragma once

#include <tuple>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/map.hpp>

#include <opencv2/opencv.hpp>

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

