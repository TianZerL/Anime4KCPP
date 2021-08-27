cmake_minimum_required(VERSION 3.9)
project(Anime4KCPP)

if(APPLE)
    # using brew's llvm (see https://stackoverflow.com/a/54715120/1410221)
    include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
    link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_INSTALL_PREFIX ${TOP_DIR}/install)

string(TIMESTAMP TODAY "%Y-%m-%d")

option(Build_GUI "Build GUI or not" OFF)
option(Build_CLI "Build CLI or not" ON)
option(Build_VapourSynth_Plugin "Build Anime4KCPP for VapourSynth plugin or not" OFF)
option(Build_AviSynthPlus_Plugin "Build Anime4KCPP for AviSynthPlus plugin or not" OFF)
option(Build_C_Wrapper "Build C wrapper of Anime4KCPP or not" OFF)
option(Build_Static_C_Wrapper "Build Anime4KCPP core c wrapper as static library" OFF)
option(Build_Static_Core "Build Anime4KCPP core as static library" ON)
option(Built_In_Kernel "Built-in kernel or not" ON)
option(Use_Eigen3 "Use Eigen3 for CPU processor acceleration" OFF)
option(Ryzen_Optimization "Optimization for AMD Ryzen CPU" OFF)
option(Native_Optimization "Optimization for native CPU arch" OFF)
option(Other_Optimization "Other optimizations may or may not useful" OFF)
option(Maximum_Optimization "Enable maximum optimizations for the current platform, may cause unknown bugs" OFF)
option(Enable_CUDA "Enable CUDA module" OFF)
option(Enable_OpenCL "Enable OpenCL module" ON)
option(Enable_NCNN "Enable ncnn module" OFF)
option(Enable_Fast_Math "Enable fast math" OFF)
option(Enable_AVX2 "Enable on AVX2 arch, AVX2 and FMA are required" OFF)
option(Enable_OpenCV_DNN "Use OpenCV DNN module for CPU processing, not for DSFilter, models need to be loaded from the outside" OFF)
option(Enable_Video "Enable video processor module" ON)
option(Enable_Preview_GUI "Enable image preview in GUI window for core" ON)
option(Enable_Image_IO "Enable image IO for core" ON)
option(Disable_Parallel "Disable parallel processing" OFF)

if(Build_Static_C_Wrapper)
    option(Build_Static_C_Wrapper_PIC "If Build_Static_C_Wrapper is ON, build C_Wrapper with -fPIC" ON)
endif()

if(Build_Static_Core)
    option(Build_Static_Core_PIC "If Build_Static_Core is ON, build core with -fPIC" OFF)
endif()

if(APPLE)
    option(Use_Legacy_OpenCL_API "Compatible with old GPU which only supports OpenCL1.2" ON)
else()
    option(Use_Legacy_OpenCL_API "Compatible with old GPU which only supports OpenCL1.2" OFF)
endif()

if(WIN32 AND (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    option(Use_OpenCV_With_MSVC_For_Clang "Use OpenCV build by MSVC for Clang" ON)
    option(Use_OpenCV_With_MINGW_For_Clang "Use OpenCV build by MinGW for Clang" OFF)
    if(Use_OpenCV_With_MSVC_For_Clang AND Use_OpenCV_With_MINGW_For_Clang)
        message(FATAL_ERROR "Use_OpenCV_With_MSVC_For_Clang and Use_OpenCV_With_MINGW_For_Clang cannot be true at same time")
    endif()
endif()

if(Build_GUI)
    set(Qt_Version "5" CACHE STRING "5 or 6")
endif()

if(Enable_CUDA)
    option(CUDA_Auto_CC "Use auto cuda compute capability" OFF)
    if(NOT CUDA_Auto_CC)
        set(CUDA_Minimum_CC "35" CACHE PATH "Set the minimum cuda compute capability")
        set(CUDA_CC "75" CACHE PATH "Set the cuda compute capability")
    endif()
    if(NOT ${CMAKE_MINOR_VERSION} LESS 18)
        cmake_policy(SET CMP0104 NEW)
    endif()
    enable_language(CUDA)
endif()

if(Maximum_Optimization)
    set(CMAKE_BUILD_TYPE Release)
    set(Other_Optimization ON)
    set(Ryzen_Optimization ON)
    set(Native_Optimization ON)
    set(Enable_Fast_Math ON)
endif()

if(Other_Optimization)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_Supported OUTPUT IPO_Error)

    if(IPO_Supported)
        message(STATUS "IPO enabled")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    else()
        message(STATUS "IPO not supported: ${IPO_Error}")
    endif()
endif()

if(MSVC)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        if(Enable_Fast_Math)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast /clang:-ffast-math")
        endif()
        if(Enable_AVX2)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
        elseif(Native_Optimization)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-march=native")
        elseif(Ryzen_Optimization)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
        endif()
    else()
        if(Enable_Fast_Math)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast")
        endif()
        if(Enable_AVX2 OR Ryzen_Optimization)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
        endif()
    endif()
else()
    if(Enable_Fast_Math)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
    endif()
    if(Enable_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    elseif(Native_Optimization)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    elseif(Ryzen_Optimization)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")  
    endif()
endif()

include(${TOP_DIR}/cmake/Detection.cmake)

if(HAS_FILESYSTEM)
    option(Use_Boost_filesystem "Use filesystem lib from boost instead of STL" OFF)
else()
    message (
        STATUS 
        "Failed to complie has_filesystem.cpp, will use boost::filesystem instead of std::filesystem\n"
        ${HAS_FILESYSTEM_MSG}
    )
    option(Use_Boost_filesystem "Use filesystem lib from boost instead of STL" ON)
endif()

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

if(NOT Disable_Parallel)
    set(Parallel_Library "Auto" CACHE STRING "Auto, PPL, OpenMP, TBB or Built-in")

    if(NOT Parallel_Library MATCHES "^(Auto|PPL|OpenMP|TBB|Built-in)$")
        message (
            FATAL_ERROR "Unkonw value of Parallel_Library\n"
        )
    endif()

    if(Parallel_Library MATCHES "Auto")
        if(MSVC)
            if(NOT (MSVC_VERSION LESS 1600))
                add_definitions(-DUSE_PPL)
                set(Parallel_Library_Type PPL)
            else()
                add_definitions(-DUSE_OPENMP)
                set(Parallel_Library_Type OpenMP)
            endif()
        else()
            add_definitions(-DUSE_OPENMP)
            set(Parallel_Library_Type OpenMP)
        endif()
    elseif(Parallel_Library MATCHES "PPL")
        if(MSVC AND NOT (MSVC_VERSION LESS 1600))
            add_definitions(-DUSE_PPL)
            set(Parallel_Library_Type PPL)
        else()
            message (
                FATAL_ERROR "PPL only available in MSVC10 or newer\n"
            )
        endif()
    elseif(Parallel_Library MATCHES "TBB")
        add_definitions(-DUSE_TBB)
        set(Parallel_Library_Type TBB)
    elseif(Parallel_Library MATCHES "OpenMP")
        add_definitions(-DUSE_OPENMP)
        set(Parallel_Library_Type OpenMP)
    else()
        set(Parallel_Library_Type Built-in)
    endif()
else()
    add_definitions(-DDISABLE_PARALLEL)
    set(Parallel_Library_Type None)
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
"Building information:\n"
"   Build date: ${TODAY}\n\n"
"   Build CLI ${Build_CLI}\n"
"   Build GUI ${Build_GUI}\n"
"   Build VapourSynth plugin ${Build_VapourSynth_Plugin}\n"
"   Build AviSynthPlus plugin ${Build_AviSynthPlus_Plugin}\n"
"   Build C wrapper ${Build_C_Wrapper}\n\n"
"   Build static C wrapper ${Build_Static_C_Wrapper}\n"
"   Build static core ${Build_Static_Core}\n"
"   Built-in kernel ${Built_In_Kernel}\n\n"
"   Use Boost filesystem ${Use_Boost_filesystem}\n"
"   Use Eigen3 ${Use_Eigen3}\n"
"   Use Legacy OpenCL API ${Use_Legacy_OpenCL_API}\n\n"
"   Ryzen Optimization ${Ryzen_Optimization}\n"
"   Native Optimization ${Native_Optimization}\n"
"   Other Optimization ${Other_Optimization}\n\n"
"   Enable OpenCL ${Enable_OpenCL}\n"
"   Enable CUDA ${Enable_CUDA}\n"
"   Enable NCNN ${Enable_NCNN}\n"
"   Enable AVX2 ${Enable_AVX2}\n"
"   Enable OpenCV DNN ${Enable_OpenCV_DNN}\n"
"   Enable video ${Enable_Video}\n"
"   Enable Preview GUI ${Enable_Preview_GUI}\n"
"   Enable image IO ${Enable_Image_IO}\n"
"   Disable parallel ${Disable_Parallel}\n\n"
"   Parallel library ${Parallel_Library_Type}\n"
)
if(WIN32)
    message(STATUS
    "Extra building information for Windows:\n"
    "   Build DSFilter ${Build_DS_Filter}\n"
    )
endif()

if(NOT MSVC)
    message(STATUS  
    "Build type: ${CMAKE_BUILD_TYPE}\n"
    )
endif()

message(STATUS  
"C++ compiler flags:\n ${CMAKE_CXX_FLAGS}\n"
)

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(OS_64_Bit ON)
else()
    set(OS_64_Bit OFF)
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

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
