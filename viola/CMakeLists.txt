PROJECT(viola)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)


SET(CMAKE_MODULE_PATH ${viola_SOURCE_DIR}/CMakeModules/)
SET(CMAKE_CXX_STANDARD 11)

#SET(CMAKE_BUILD_TYPE Debug)

SET(CMAKE_CXX_FLAGS_DEBUG "${viola_SOURCE_DIR} -ggdb3 -O0 -pg -std=c++11")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} --fast-math -O3 -flto")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -ggdb3 --fast-math -flto -O3 -std=c++11")

FIND_PACKAGE(PkgConfig QUIET)

## CONFIG FILE SECTION

SET(USE_KINECT_2 1)

SET(USE_INTEL_TBB 1)
if(${USE_INTEL_TBB})
    SET(TBB_PARTITIONS 8)
    PKG_CHECK_MODULES(TBB tbb)
endif()

CONFIGURE_FILE(project_config.h.in project_config.h)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

#################
find_package(OpenMP)

if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

INCLUDE(depth_registration)
INCLUDE_DIRECTORIES(${DEPTH_REGISTRATION_INCLUDE_DIR})

INCLUDE(libusb_dep)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(libfreenect REQUIRED)
FIND_PACKAGE(freenect2 REQUIRED)
FIND_PACKAGE(MRPT REQUIRED base bayes obs gui maps)


#INCLUDE_DIRECTORIES(${freenect2_INCLUDE_DIR})
LINK_DIRECTORIES(${viola_SOURCE_DIR})

ADD_LIBRARY(ImageParticleFilter SHARED CImageParticleFilter.cpp)
ADD_LIBRARY(FacesDetection SHARED faces_detection.cpp)
ADD_LIBRARY(KinectCamera SHARED KinectCamera.cpp)
ADD_LIBRARY(ImageRegistration SHARED ImageRegistration.cpp)

ADD_EXECUTABLE(particle_filter_main particle_filter_main.cpp)
ADD_EXECUTABLE(kinect_3d_view kinect_3d_view.cpp)
ADD_EXECUTABLE(calibration_pairs calibration_pairs.cpp)
ADD_EXECUTABLE(image_registration_example image_registration_example.cpp)
ADD_EXECUTABLE(Protonect-reg Protonect-depth-registration.cpp)

#ADD_EXECUTABLE(particle_filter_cd particle_filter_cd.cpp)
#ADD_EXECUTABLE(objectDetection2 objectDetection2.cpp)
#ADD_EXECUTABLE(squareDetection  squareDetection.cpp)
#ADD_EXECUTABLE(particle_filter particle_filter.cpp)
#ADD_EXECUTABLE(objectDetection  objectDetection.cpp)
#ADD_EXECUTABLE(openni_capture openni_capture.cpp)
#ADD_EXECUTABLE(hog peopledetect.cpp)

FIND_PACKAGE(OpenCL REQUIRED)

TARGET_LINK_LIBRARIES(ImageRegistration
    ${DEPTH_REGISTRATION_LIBRARY}
)

TARGET_LINK_LIBRARIES(particle_filter_main
     ImageParticleFilter
     FacesDetection
     KinectCamera
     ${MRPT_LIBS} 
     ${OpenCV_LIBS} 
     ${TBB_LIBRARIES} 
     ${freenect2_LIBRARY}
     ${OpenCL_LIBRARIES}
     ${DEPTH_REGISTRATION_LIBRARY}
     ImageRegistration
)

TARGET_LINK_LIBRARIES(image_registration_example
    ${OpenCV_LIBS}
    ${OpenCL_LIBRARIES}
    ${DEPTH_REGISTRATION_LIBRARY}
    ImageRegistration
)

TARGET_LINK_LIBRARIES(calibration_pairs
    ${OpenCV_LIBS}
    ${freenect2_LIBRARY}
)

TARGET_LINK_LIBRARIES(kinect_3d_view
    KinectCamera
    ${MRPT_LIBS}
    ${OpenCV_LIBS}
    ${freenect2_LIBRARY}
)

TARGET_LINK_LIBRARIES(Protonect-reg
  ${OpenCV_LIBS}
  freenect2
  ${OpenCL_LIBRARIES}
  ${DEPTH_REGISTRATION_LIBRARY}
  ImageRegistration
)

#TARGET_LINK_LIBRARIES(particle_filter_cd ${MRPT_LIBS} ${OpenCV_LIBS} ${TBB_LIBRARIES} ${freenect2_LIBRARY})
#TARGET_LINK_LIBRARIES(objectDetection2 ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(squareDetection ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(particle_filter ${MRPT_LIBS} ${OpenCV_LIBS} ${TBB_LIBRARIES})
#TARGET_LINK_LIBRARIES(objectDetection ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(openni_capture  ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(hog ${OpenCV_LIBS})

