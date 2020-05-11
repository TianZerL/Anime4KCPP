cmake_minimum_required(VERSION 3.4.1)

# Set your libs path here
set(OpenCV_DIR "D:/Code/Lib/OpenCV-android-sdk/sdk/native/jni")
set(OpenCL_LIBRARY "D:/Code/Lib/opencl/lib/${ANDROID_ABI}/libOpenCL.so")
set(OpenCL_INCLUDE_DIR "D:/Code/Lib/opencl/inc")

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/Anime4KCore/include ${OpenCL_INCLUDE_DIR} ${OpenCV_INCLUDE_DIR})

aux_source_directory(${CMAKE_SOURCE_DIR}/Anime4KCore/src SOURCE)

add_definitions(-DBUILT_IN_KERNEL)

add_library(Anime4KCPPCore

        SHARED

        Anime4KCPPWarpper.cpp
        ${SOURCE})

target_link_libraries(Anime4KCPPCore

        ${OpenCV_LIBS}
        ${OpenCL_LIBRARIES})
