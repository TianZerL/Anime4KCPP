cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

# Set your libs path here
set(OpenCV_DIR "D:/Code/Lib/OpenCV-android-sdk/sdk/native/jni")
set(OpenCL_LIBRARY "D:/Code/Lib/opencl/lib/libOpenCL.so")
set(OpenCL_INCLUDE_DIR "D:/Code/Lib/opencl/inc")

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/Anime4KCore/include ${OpenCL_INCLUDE_DIR} ${OpenCV_INCLUDE_DIR})

aux_source_directory(${CMAKE_SOURCE_DIR}/Anime4KCore/src SOURCE)

file(GLOB INCLUDE ${CMAKE_SOURCE_DIR}/Anime4KCore/include/*.h)

add_definitions(-DBUILT_IN_KERNEL)

add_library( Anime4KCPP

             SHARED

             Anime4KCPP_Warpper.cpp
             ${INCLUDE}
             ${SOURCE}
        )

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log 
        )

target_link_libraries( Anime4KCPP

                       ${log-lib}
                       ${OpenCV_LIBS}
                       ${OpenCL_LIBRARIES}
        )