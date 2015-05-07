project(viola)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${viola_SOURCE_DIR}/CMakeModules/)

set(CMAKE_CXX_STANDARD 11)

#set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS_DEBUG "${viola_SOURCE_DIR} -ggdb3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} --fast-math -O3 -flto")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -ggdb3 --fast-math -flto -O3 -std=c++11")

GET_FILENAME_COMPONENT(LIBUSB_DIR "${viola_SOURCE_DIR}/depends/libusb/" REALPATH)
INCLUDE_DIRECTORIES("${LIBUSB_DIR}/include/libusb-1.0/")
LINK_DIRECTORIES("${LIBUSB_DIR}/lib/")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

FIND_PACKAGE(PkgConfig QUIET)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(libfreenect REQUIRED)
FIND_PACKAGE(freenect2 REQUIRED)
PKG_CHECK_MODULES(TBB tbb)
FIND_PACKAGE(MRPT REQUIRED base bayes obs gui maps)

INCLUDE_DIRECTORIES({freenect2_INCLUDE_DIR})
LINK_DIRECTORIES(${viola_SOURCE_DIR})

ADD_LIBRARY(ImageParticleFilter SHARED CImageParticleFilter.cpp)
ADD_LIBRARY(FacesDetection SHARED faces_detection.cpp)
ADD_LIBRARY(KinectCamera SHARED KinectCamera.cpp)

ADD_EXECUTABLE(particle_filter_main particle_filter_main.cpp)
ADD_EXECUTABLE(kinect_3d_view kinect_3d_view.cpp)

#ADD_EXECUTABLE(particle_filter_cd particle_filter_cd.cpp)
#ADD_EXECUTABLE(objectDetection2 objectDetection2.cpp)
#ADD_EXECUTABLE(squareDetection  squareDetection.cpp)
#ADD_EXECUTABLE(particle_filter particle_filter.cpp)
#ADD_EXECUTABLE(objectDetection  objectDetection.cpp)
#ADD_EXECUTABLE(openni_capture openni_capture.cpp)
#ADD_EXECUTABLE(hog peopledetect.cpp)

TARGET_LINK_LIBRARIES(particle_filter_main
	 ImageParticleFilter
	 FacesDetection
	 KinectCamera
	 ${MRPT_LIBS} 
	 ${OpenCV_LIBS} 
	 ${TBB_LIBRARIES} 
	 ${freenect2_LIBRARY}
 )

TARGET_LINK_LIBRARIES(kinect_3d_view
	KinectCamera
	${MRPT_LIBS}
	${OpenCV_LIBS}
	${freenect2_LIBRARY}
)

#TARGET_LINK_LIBRARIES(particle_filter_cd ${MRPT_LIBS} ${OpenCV_LIBS} ${TBB_LIBRARIES} ${freenect2_LIBRARY})
#TARGET_LINK_LIBRARIES(objectDetection2 ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(squareDetection ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(particle_filter ${MRPT_LIBS} ${OpenCV_LIBS} ${TBB_LIBRARIES})
#TARGET_LINK_LIBRARIES(objectDetection ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(openni_capture  ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(hog ${OpenCV_LIBS})

