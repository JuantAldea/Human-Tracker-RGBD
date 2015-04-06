project(va-05)
cmake_minimum_required(VERSION 2.8)
add_executable(objectDetection  objectDetection.cpp)
add_executable(objectDetection2  objectDetection2.cpp)
add_executable(openni_capture openni_capture.cpp)
add_executable(hog peopledetect.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pedantic-errors -Wall -Wextra -Werror")
find_package(OpenCV REQUIRED)

target_link_libraries(objectDetection ${OpenCV_LIBS})
target_link_libraries(objectDetection2 ${OpenCV_LIBS})
target_link_libraries(openni_capture  ${OpenCV_LIBS})
target_link_libraries(hog  ${OpenCV_LIBS})

set(CMAKE_BUILD_TYPE Debug)


