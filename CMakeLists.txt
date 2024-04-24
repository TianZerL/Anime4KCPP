cmake_minimum_required(VERSION 3.18)
project(Anime4KCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_DIR ${TOP_DIR}/cmake)
set(DEPENDENCY_DIR ${CMAKE_DIR}/dependency)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_INSTALL_PREFIX ${TOP_DIR}/install)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build Type" FORCE)
endif()

if(POLICY CMP0135) #for downloading archive file, https://cmake.org/cmake/help/latest/policy/CMP0135.html
    cmake_policy(SET CMP0135 NEW)
endif()

string(TIMESTAMP DATE_TODAY "%Y-%m-%d")
string(TIMESTAMP DATE_YEAR "%Y")

include(${CMAKE_DIR}/CheckCompiler.cmake)

option(AC_SHARED_LIB "build to shared library" OFF)
option(AC_CORE_WITH_EIGEN3 "build core with eigen3" OFF)
option(AC_CORE_WITH_SSE "build core with x86 sse" ${AC_COMPILER_SUPPORT_SSE})
option(AC_CORE_WITH_AVX "build core with x86 avx" ${AC_COMPILER_SUPPORT_AVX})
option(AC_CORE_WITH_FMA "build core with x86 fma and avx" ${AC_COMPILER_SUPPORT_FMA})
option(AC_CORE_WITH_NEON "build core with neon" ${AC_COMPILER_SUPPORT_NEON})
option(AC_CORE_WITH_OPENCL "build core with opencl" OFF)
option(AC_CORE_WITH_CUDA "build core with cuda" OFF)
option(AC_CORE_ENABLE_IMAGE_IO "enable image file read and write for core" ON)
option(AC_BUILD_CLI "build cli" ON)
option(AC_BUILD_FILTER_VAPOURSYNTH "build vapoursynth filter" OFF)
option(AC_BUILD_FILTER_AVISYNTH "build avisynth filter" OFF)
option(AC_BUILD_FILTER_AVISYNTH_VAPOURSYNTH "build an avisynth and vapoursynth universal filter" OFF)
option(AC_LTO "enable LTO" OFF)
option(AC_DISABLE_RTTI "disable rtti" OFF)
option(AC_DISABLE_EXCEPTION "disable exception" OFF)

if(AC_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT LTO_CHECK OUTPUT LTO_ERROR)

    if(LTO_CHECK)
        message(STATUS "LTO enabled")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    else()
        message(STATUS "LTO not supported: ${LTO_ERROR}")
    endif()
endif()

if(AC_CORE_WITH_FMA)
    set(AC_CORE_WITH_AVX ON)
endif()

message(STATUS 
"SIMD support:
    SSE: ${AC_CORE_WITH_SSE}
    AVX: ${AC_CORE_WITH_AVX}
    FMA: ${AC_CORE_WITH_FMA}
    NEON: ${AC_CORE_WITH_NEON}
")

macro(ac_check_disable_flags TARGET)
    if(AC_DISABLE_RTTI)
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
            target_compile_options(${TARGET} PUBLIC /GR-)
        else()
            target_compile_options(${TARGET} PUBLIC -fno-rtti)
        endif()
    endif()

    if(AC_DISABLE_EXCEPTION)
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
            target_compile_options(${TARGET} PUBLIC /EHsc /D_HAS_EXCEPTIONS=0)
        else()
            target_compile_options(${TARGET} PUBLIC -fno-exceptions)
        endif()
    endif()
endmacro()

add_subdirectory(core)
add_subdirectory(util)
add_subdirectory(filter)
if(AC_BUILD_CLI)
    add_subdirectory(cli)
endif()

install(
    EXPORT AC
    FILE ACConfig.cmake
    NAMESPACE AC::
    DESTINATION ac/cmake
)
