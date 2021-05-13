cmake_minimum_required(VERSION 3.4.1)

# Set your libs path here
set(OpenCV_DIR "E:/Libs/Android/OpenCV/sdk/native/jni")
set(OpenCL_LIBRARY "E:/Libs/Android/OpenCL/lib/${ANDROID_ABI}/libOpenCL.so")
set(OpenCL_INCLUDE_DIR "E:/Libs/Android/OpenCL/inc")
set(EIGEN3_INCLUDE_DIR "E:/Libs/eigen-3.4-rc1")

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/Anime4KCore/include ${OpenCL_INCLUDE_DIR} ${OpenCV_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})

aux_source_directory(${CMAKE_SOURCE_DIR}/Anime4KCore/src SOURCE)

add_definitions(-DBUILT_IN_KERNEL)
add_definitions(-DENABLE_VIDEO)
add_definitions(-DENABLE_OPENCL)
add_definitions(-DUSE_EIGEN3)

add_library(Anime4KCPPCore SHARED
        Anime4KCPPWrapper.cpp
        ${SOURCE})

target_link_libraries(Anime4KCPPCore
        ${OpenCV_LIBS}
        ${OpenCL_LIBRARIES})
