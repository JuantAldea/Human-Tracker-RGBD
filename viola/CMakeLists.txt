project(viola)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -O0 -ggdb3")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.8")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    endif()
endif()

#set(CMAKE_BUILD_TYPE Debug)

FIND_PACKAGE(MRPT REQUIRED base bayes obs gui)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(libfreenect REQUIRED)


ADD_EXECUTABLE(objectDetection  objectDetection.cpp)
ADD_EXECUTABLE(objectDetection2 objectDetection2.cpp)
ADD_EXECUTABLE(squareDetection  squareDetection.cpp)
ADD_EXECUTABLE(openni_capture openni_capture.cpp)
ADD_EXECUTABLE(hog peopledetect.cpp)
ADD_EXECUTABLE(particle_filter particle_filter.cpp)

ADD_LIBRARY (CObservationImageWithModel CObservationImageWithModel.cpp)

LINK_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

TARGET_LINK_LIBRARIES(particle_filter  ${MRPT_LIBS} ${OpenCV_LIBS} CObservationImageWithModel)
TARGET_LINK_LIBRARIES(objectDetection ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(objectDetection2 ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(squareDetection ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(openni_capture  ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(hog ${OpenCV_LIBS})

