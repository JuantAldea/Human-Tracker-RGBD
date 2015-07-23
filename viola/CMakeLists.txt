PROJECT(viola)

CMAKE_MINIMUM_REQUIRED(VERSION 3.0)


SET(CMAKE_MODULE_PATH ${viola_SOURCE_DIR}/CMakeModules/)
SET(CMAKE_CXX_STANDARD 11)

#SET(CMAKE_BUILD_TYPE Debug)

#SET(CMAKE_CXX_FLAGS_DEBUG "${viola_SOURCE_DIR} -ggdb3 -O0 -pg -std=c++11 -fbuiltin -fno-inline")
#SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} --fast-math -O3 -flto")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -ggdb3 --fast-math -O3 -std=gnu++11 -pg -fbuiltin -fdiagnostics-color=always -Wno-unused-parameter -Wno-unused-variable")

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -ggdb3 --fast-math -O0 -std=gnu++11 -pg -fbuiltin -fno-inline -fno-omit-frame-pointer -fdiagnostics-color=always -Wno-unused-variable")
#SET(CMAKE_EXE_LINKER_FLAGS "-pg")

FIND_PACKAGE(PkgConfig QUIET)

## CONFIG FILE SECTION

#SET(VIEW_3D 1)

SET(USE_KINECT_2 1)
SET(USE_INTEL_TBB 1)
IF(${USE_INTEL_TBB})
    SET(TBB_PARTITIONS 8)
    PKG_CHECK_MODULES(TBB tbb)
ENDIF()

CONFIGURE_FILE(project_config.h.in project_config.h)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

#################
find_package(OpenMP)

if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

INCLUDE(depth_registration)
INCLUDE_DIRECTORIES(${DEPTH_REGISTRATION_INCLUDE_DIR})

SET(USE_AVX_INSTRUCTIONS 1)
INCLUDE(dlib-18.16/dlib/cmake)

INCLUDE(libusb_dep)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(libfreenect REQUIRED)
FIND_PACKAGE(freenect2 REQUIRED)
FIND_PACKAGE(MRPT REQUIRED base bayes obs gui maps)

MACRO(add_header_lib name)
   ADD_LIBRARY(${name} STATIC ${name}.h)
   SET_TARGET_PROPERTIES(${name} PROPERTIES LINKER_LANGUAGE CXX)
ENDMACRO()

#INCLUDE_DIRECTORIES(${freenect2_INCLUDE_DIR})
LINK_DIRECTORIES(${viola_SOURCE_DIR})

add_header_lib(EllipseStash)
add_header_lib(Tracker)
add_header_lib(MultiTracker)
add_header_lib(StateEstimation)
add_header_lib(BoostSerializers)
add_header_lib(Kinect2VideoReader)
add_header_lib(ModelParameters)
add_header_lib(ColorModel)
add_header_lib(GeometryHelpers)
add_header_lib(MiscHelpers)
add_header_lib(EllipseFunctions)

ADD_LIBRARY(FacesDetection STATIC FacesDetection.cpp)
ADD_LIBRARY(ImageRegistration STATIC ImageRegistration.cpp)

#add_header_lib(ImageRegistration)
#add_header_lib(FacesDetection)


ADD_LIBRARY(ImageParticleFilter STATIC CImageParticleFilter.h)
SET_TARGET_PROPERTIES(ImageParticleFilter PROPERTIES LINKER_LANGUAGE CXX)

#ADD_LIBRARY(KinectCamera SHARED KinectCamera.cpp)

ADD_EXECUTABLE(particle_filter_main particle_filter_main.cpp)
ADD_EXECUTABLE(kinect2_recorder kinect2_recorder.cpp)

ADD_EXECUTABLE(kinect2_video_replay Kinect2VideoReplay.cpp)

#ADD_EXECUTABLE(kinect_3d_view kinect_3d_view.cpp)
#ADD_EXECUTABLE(calibration_pairs calibration_pairs.cpp)
#ADD_EXECUTABLE(image_registration_example image_registration_example.cpp)
#ADD_EXECUTABLE(Protonect-reg Protonect-depth-registration.cpp)

#ADD_EXECUTABLE(particle_filter_cd particle_filter_cd.cpp)
#ADD_EXECUTABLE(objectDetection2 objectDetection2.cpp)
#ADD_EXECUTABLE(squareDetection  squareDetection.cpp)
#ADD_EXECUTABLE(particle_filter particle_filter.cpp)
#ADD_EXECUTABLE(objectDetection  objectDetection.cpp)
#ADD_EXECUTABLE(openni_capture openni_capture.cpp)
#ADD_EXECUTABLE(hog peopledetect.cpp)
#ADD_EXECUTABLE(Boost-normal Boost-normal.cpp)
ADD_EXECUTABLE(model_projections_precalculator model_projections_precalculator.cpp)

FIND_PACKAGE(OpenCL REQUIRED)
find_package(Boost REQUIRED system serialization)
include_directories(${Boost_INCLUDE_DIRS})

TARGET_LINK_LIBRARIES(ImageRegistration
    ${DEPTH_REGISTRATION_LIBRARY}
)

#[[
TARGET_LINK_LIBRARIES(Boost-normal
    ${Boost_LIBRARIES}
)
]]

TARGET_LINK_LIBRARIES(kinect2_recorder
    ${freenect2_LIBRARY}
    ${OpenCV_LIBS}
)

TARGET_LINK_LIBRARIES(kinect2_video_replay
    ${OpenCV_LIBS}
)

TARGET_LINK_LIBRARIES(model_projections_precalculator
    ModelParameters
    ImageRegistration
    EllipseFunctions
    GeometryHelpers
    MiscHelpers
    BoostSerializers

    ${MRPT_LIBS}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
)


TARGET_LINK_LIBRARIES(particle_filter_main
    ${MRPT_LIBS}
    ${OpenCV_LIBS}
    ${TBB_LIBRARIES}
    ${freenect2_LIBRARY}
    ${OpenCL_LIBRARIES}
    ${DEPTH_REGISTRATION_LIBRARY}
    ${Boost_LIBRARIES}

    ImageParticleFilter
    FacesDetection
    ImageRegistration
    EllipseFunctions
    ColorModel
    GeometryHelpers
    MiscHelpers
    Tracker
    MultiTracker
    StateEstimation
    EllipseStash
    BoostSerializers
    ModelParameters
    dlib
)

#[[
TARGET_LINK_LIBRARIES(image_registration_example
    ${OpenCV_LIBS}
    ${OpenCL_LIBRARIES}
    ${DEPTH_REGISTRATION_LIBRARY}
    ImageRegistration
)

TARGET_LINK_LIBRARIES(calibration_pairs
    ${OpenCV_LIBS}
    ${freenect2_LIBRARY}
    ${OpenCL_LIBRARIES}
)

TARGET_LINK_LIBRARIES(kinect_3d_view
    KinectCamera
    ${MRPT_LIBS}
    ${OpenCV_LIBS}
    ${OpenCL_LIBRARIES}
    ${freenect2_LIBRARY}
)

TARGET_LINK_LIBRARIES(Protonect-reg
  ${OpenCV_LIBS}
  freenect2
  ${OpenCL_LIBRARIES}
  ${DEPTH_REGISTRATION_LIBRARY}
  ImageRegistration
)
]]


#TARGET_LINK_LIBRARIES(particle_filter_cd ${MRPT_LIBS} ${OpenCV_LIBS} ${TBB_LIBRARIES} ${freenect2_LIBRARY})
#TARGET_LINK_LIBRARIES(objectDetection2 ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(squareDetection ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(particle_filter ${MRPT_LIBS} ${OpenCV_LIBS} ${TBB_LIBRARIES})
#TARGET_LINK_LIBRARIES(objectDetection ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(openni_capture  ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(hog ${OpenCV_LIBS})

