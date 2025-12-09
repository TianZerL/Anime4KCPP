cmake_minimum_required(VERSION 3.19)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Set default build type to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build Type" FORCE)
endif()

project(Anime4KCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_DIR ${TOP_DIR}/cmake)
set(DEPENDENCY_DIR ${CMAKE_DIR}/dependency)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

string(TIMESTAMP DATE_TODAY "%Y-%m-%d")
string(TIMESTAMP DATE_YEAR "%Y")

if (CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(BUILD_ARCH_32BIT TRUE)
else()
    set(BUILD_ARCH_32BIT FALSE)
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(STATUS "Set default install path ${TOP_DIR}/install")
    set_property(CACHE CMAKE_INSTALL_PREFIX PROPERTY VALUE "${TOP_DIR}/install")
endif()

include(GNUInstallDirs)

include(${CMAKE_DIR}/CheckCompilerSIMD.cmake)

option(AC_SHARED_LIB "build as a shared library" OFF)
option(AC_CORE_WITH_EIGEN3 "build core with eigen3" OFF)
option(AC_CORE_WITH_SSE "build core with x86 sse" ${AC_COMPILER_SUPPORT_SSE})
option(AC_CORE_WITH_SSE2 "build core with x86 sse2" ${AC_COMPILER_SUPPORT_SSE2})
option(AC_CORE_WITH_AVX "build core with x86 avx" ${AC_COMPILER_SUPPORT_AVX})
option(AC_CORE_WITH_AVX2 "build core with x86 avx2" ${AC_COMPILER_SUPPORT_AVX2})
option(AC_CORE_WITH_FMA "build core with x86 fma and avx" ${AC_COMPILER_SUPPORT_FMA})
option(AC_CORE_WITH_NEON "build core with arm neon" ${AC_COMPILER_SUPPORT_NEON})
option(AC_CORE_WITH_WASM_SIMD128 "build core with wasm simd128" ${AC_COMPILER_SUPPORT_WASM_SIMD128})
option(AC_CORE_WITH_OPENCL "build core with opencl" OFF)
option(AC_CORE_WITH_CUDA "build core with cuda" OFF)
option(AC_CORE_ENABLE_FAST_MATH "enable fast math for core" OFF)
option(AC_CORE_DISABLE_IMAGE_IO "disable image file read and write for core" OFF)
option(AC_BUILD_CLI "build cli" ON)
option(AC_BUILD_GUI "build gui" OFF)
option(AC_BUILD_VIDEO "build video module" OFF)
option(AC_BUILD_FILTER_AVISYNTH "build avisynth filter" OFF)
option(AC_BUILD_FILTER_VAPOURSYNTH "build vapoursynth filter" OFF)
option(AC_BUILD_FILTER_DIRECTSHOW "build directshow filter" OFF)
option(AC_BUILD_FILTER_AVISYNTH_VAPOURSYNTH "build an avisynth and vapoursynth universal filter" OFF)
option(AC_BUILD_BINDING_C "build c binding for core" OFF)
option(AC_BUILD_BINDING_PYTHON "build python binding for core" OFF)
option(AC_TOOLS_BENCHMARK "build benchmark" OFF)
option(AC_TEST_UTIL "build util module test" OFF)
option(AC_TEST_VIDEO "build video module test" OFF)
option(AC_ENABLE_LTO "enable LTO" OFF)
option(AC_ENABLE_STATIC_CRT "enable static link crt" OFF)
option(AC_DISABLE_RTTI "disable rtti" OFF)
option(AC_DISABLE_EXCEPTION "disable exception" OFF)
option(AC_DISABLE_PIC "disable pic or pie" OFF)

if(AC_PRESET_RELEASE)
    set(AC_CORE_WITH_OPENCL ON)
    set(AC_CORE_WITH_CUDA ON)
    set(AC_CORE_ENABLE_FAST_MATH ON)
    set(AC_BUILD_CLI ON)
    set(AC_BUILD_GUI ON)
    set(AC_BUILD_VIDEO ON)
    set(AC_BUILD_FILTER_AVISYNTH_VAPOURSYNTH ON)
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(AC_BUILD_FILTER_DIRECTSHOW ON)
    endif()
endif()

if(AC_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT LTO_CHECK OUTPUT LTO_ERROR)

    if(LTO_CHECK)
        message(STATUS "LTO enabled")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    else()
        message(STATUS "LTO not supported: ${LTO_ERROR}")
    endif()
endif()

if(AC_CORE_WITH_SSE2)
    set(AC_CORE_WITH_SSE ON)
endif()

if(AC_CORE_WITH_AVX2)
    set(AC_CORE_WITH_FMA ON)
endif()

if(AC_CORE_WITH_FMA)
    set(AC_CORE_WITH_AVX ON)
endif()

if(AC_TEST_VIDEO)
    set(AC_BUILD_VIDEO ON)
endif()

message(STATUS
"SIMD support:
    SSE: ${AC_CORE_WITH_SSE}
    SSE2: ${AC_CORE_WITH_SSE2}
    AVX: ${AC_CORE_WITH_AVX}
    AVX2: ${AC_CORE_WITH_AVX2}
    FMA: ${AC_CORE_WITH_FMA}
    NEON: ${AC_CORE_WITH_NEON}
    WASM_SIMD128: ${AC_CORE_WITH_WASM_SIMD128}
")

add_compile_definitions(
    AC_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
    AC_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
    AC_BUILD_DATE="${DATE_TODAY}"
    AC_BUILD_YEAR="${DATE_YEAR}"
)

function(ac_check_disable_flags ARG_TARGET)
    if(AC_DISABLE_RTTI)
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
            target_compile_options(${ARG_TARGET} PRIVATE /GR-)
        else()
            target_compile_options(${ARG_TARGET} PRIVATE -fno-rtti)
        endif()
    endif()

    if(AC_DISABLE_EXCEPTION)
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
            target_compile_options(${ARG_TARGET} PRIVATE /EHsc /D_HAS_EXCEPTIONS=0)
        else()
            target_compile_options(${ARG_TARGET} PRIVATE -fno-exceptions)
        endif()
    endif()
endfunction()

function(ac_check_enable_static_crt ARG_TARGET)
    if(AC_ENABLE_STATIC_CRT)
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
            set_target_properties(${ARG_TARGET} PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        else()
            target_link_options(${ARG_TARGET} PRIVATE -static-libgcc -static-libstdc++ -static)
        endif()
    endif()
endfunction()

function(ac_check_disable_pic ARG_TARGET)
    if (NOT WIN32 AND NOT AC_SHARED_LIB AND NOT AC_DISABLE_PIC)
        set_target_properties(${ARG_TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
endfunction()

add_subdirectory(core)
add_subdirectory(util)
add_subdirectory(filter)
add_subdirectory(binding)
add_subdirectory(tools)
add_subdirectory(test)
if(AC_BUILD_VIDEO)
    add_subdirectory(video)
endif()
if(AC_BUILD_CLI)
    add_subdirectory(cli)
endif()
if(AC_BUILD_GUI)
    add_subdirectory(gui)
endif()

install(
    EXPORT AC
    FILE ACConfig.cmake
    NAMESPACE AC::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ac
)
