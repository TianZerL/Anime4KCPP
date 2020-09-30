cmake_minimum_required(VERSION 3.0)
project(Anime4KCPP)

if(APPLE)
    # using brew's llvm (see https://stackoverflow.com/a/54715120/1410221)
    include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
    link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "/EHsc")
else()
    set(CMAKE_CXX_FLAGS "-O3 -lpthread -fopenmp")
    if (NOT CMAKE_BUILD_TYPE)
        set (CMAKE_BUILD_TYPE "Release")
        message (
            STATUS "No CMAKE_BUILD_TYPE selected, defaulting to ${CMAKE_BUILD_TYPE}\n"
        )
    endif ()
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

if(WIN32)
    set(CMAKE_INSTALL_PREFIX ${TOP_DIR}/install)
endif(WIN32)

option(Build_GUI "Build GUI or not" OFF)
option(Build_CLI "Build CLI or not" ON)
option(Build_VapourSynth_plugin "Build Anime4KCPP for VapourSynth plugin or not" OFF)
option(Build_AviSynthPlus_plugin "Build Anime4KCPP for AviSynthPlus plugin or not" OFF)
option(Build_C_wrapper "Build C wrapper of Anime4KCPP or not" OFF)
option(Build_C_wrapper_with_core "Build C wrapper and core in one file" OFF)
option(Built_in_kernel "Built-in kernel or not" ON)
option(Use_Boost_filesystem "Use filesystem lib from boost instead of STL" OFF)
option(Use_TBB "Use TBB for parallel processing" OFF)

if(WIN32)
    set(OpenCL_Provider "Intel" CACHE STRING "one of Intel;Nvidia;AMD;Auto")
else()
    set(OpenCL_Provider "Auto" CACHE STRING "one of Intel;Nvidia;AMD;Auto")
endif()

if(WIN32)
    option(Build_DS_Filter "Build Anime4KCPP for DirectShow or not" OFF)
    set(DirectShow_SDK_PATH "DirectShow BaseClass SDK path" CACHE PATH "Where to look for DirectShow SDK")
endif()

set(VapourSynth_SDK_PATH "VapourSynth SDK path" CACHE PATH "Where to look for VapourSynth SDK")
set(AviSynthPlus_SDK_PATH "AviSynthPlus SDK path" CACHE PATH "Where to look for AviSynthPlus SDK")

if(Use_TBB)
    set(TBB_INCLUDE_PATH "TBB include path" CACHE PATH "Where to look for TBB include")
    set(TBB_LIB_PATH "TBB lib path" CACHE PATH "Where to look for TBB lib")
endif()

if (OpenCL_Provider MATCHES "Intel")
    if(DEFINED ENV{INTELOCLSDKROOT})
        set(CMAKE_PREFIX_PATH $ENV{INTELOCLSDKROOT} CACHE PATH "Intel OpenCL SDK")
    else()
        message (
            WARNING "Can't find Intel OpenCL SDK, set OpenCL_Provider to Auto\n"
        )
    endif()
elseif(OpenCL_Provider MATCHES "AMD")
    if(DEFINED ENV{OCL_ROOT})
        set(CMAKE_PREFIX_PATH $ENV{OCL_ROOT} CACHE PATH "AMD OpenCL SDK")
    else()
        message (
            WARNING "Can't find AMD OpenCL SDK, set OpenCL_Provider to Auto\n"
        )
    endif()
elseif(OpenCL_Provider MATCHES "Nvidia")
    if(DEFINED ENV{CUDA_PATH})
        set(CMAKE_PREFIX_PATH $ENV{CUDA_PATH} CACHE PATH "Nvidia OpenCL SDK")
    else()
        message (
            WARNING "Can't find Nvidia OpenCL SDK, set OpenCL_Provider to Auto\n"
        )
    endif()
elseif(NOT OpenCL_Provider MATCHES "Auto")
    message (
        WARNING "Unkonw value of OpenCL_Provider, set to Auto\n"
    )
endif()

message(STATUS  
"Building infomation:\n"
"   Build CLI ${Build_CLI}\n"
"   Build GUI ${Build_GUI}\n"
"   Build VapourSynth plugin ${Build_VapourSynth_plugin}\n"
"   Build AviSynthPlus plugin ${Build_AviSynthPlus_plugin}\n"
"   Build C wrapper ${Build_C_wrapper}\n\n"
"   Build C wrapper with core ${Build_C_wrapper_with_core}\n"
"   Built-in kernel ${Built_in_kernel}\n\n"
"   Use Boost filesystem ${Use_Boost_filesystem}\n"
"   Use TBB ${Use_TBB}\n"
)
if(WIN32)
message(STATUS
"Extra building infomation for Windows:\n"
"   Build DSFilter ${Build_DS_Filter}\n"
)
endif()

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(OS_64_Bit ON)
else()
    set(OS_64_Bit OFF)
endif()

macro(SUBDIRLIST result curdir)
    file(GLOB children RELATIVE ${curdir} ${curdir}/*)
    set(dirlist "")
    foreach(child ${children})
        if(IS_DIRECTORY ${curdir}/${child} AND EXISTS ${curdir}/${child}/CMakeLists.txt)
            list(APPEND dirlist ${child})
        endif()
    endforeach()
    set(${result} ${dirlist})
endmacro()

SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR})

foreach(SUBDIR ${SUBDIRS})
    add_subdirectory(${SUBDIR})
endforeach()
