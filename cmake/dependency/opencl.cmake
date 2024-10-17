if(NOT TARGET dep::opencl)
    add_library(dep_opencl INTERFACE IMPORTED)
    find_package(OpenCL QUIET)
    if (NOT OpenCL_FOUND)
        message(STATUS "dep: opencl not found, will be fetched online.")
        include(FetchContent)
        FetchContent_Declare(
            opencl
            GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-SDK.git
            GIT_TAG main
        )
        set(OPENCL_SDK_BUILD_UTILITY_LIBRARIES OFF CACHE BOOL "OpenCL SDK option" FORCE)
        set(OPENCL_SDK_BUILD_SAMPLES OFF CACHE BOOL "OpenCL SDK option" FORCE)
        set(OPENCL_SDK_BUILD_OPENGL_SAMPLES OFF CACHE BOOL "OpenCL SDK option" FORCE)
        set(OPENCL_SDK_BUILD_CLINFO OFF CACHE BOOL "OpenCL SDK option" FORCE)
        set(BUILD_TESTING OFF CACHE BOOL "OpenCL SDK option" FORCE)
        set(BUILD_DOCS OFF CACHE BOOL "OpenCL SDK option" FORCE)
        set(BUILD_EXAMPLES OFF CACHE BOOL "OpenCL SDK option" FORCE)
        set(OPENCL_CLHPP_BUILD_TESTING OFF CACHE BOOL "OpenCL SDK option" FORCE)
        set(OPENCL_HEADERS_BUILD_TESTING OFF CACHE BOOL "OpenCL SDK option" FORCE)
        set(OPENCL_HEADERS_BUILD_CXX_TESTS OFF CACHE BOOL "OpenCL SDK option" FORCE)
        set(ENABLE_OPENCL_LAYERINFO OFF CACHE BOOL "OpenCL SDK option" FORCE)
        set(OPENCL_ICD_LOADER_BUILD_TESTING OFF CACHE BOOL "OpenCL SDK option" FORCE)
        FetchContent_MakeAvailable(opencl)
        target_link_libraries(dep_opencl INTERFACE OpenCL::HeadersCpp)
        set(dep_opencl_TARGET_VERSION 300)
    else()
        include(${CMAKE_DIR}/CheckOpenCLHPP.cmake)
        if(NOT dep_opencl_SUPPORT_HPP)
            message(STATUS "dep: opencl hpp headers not found or no supported, will be fetched online.")
            include(FetchContent)
            FetchContent_Declare(
                openclhpp
                GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-CLHPP.git
                GIT_TAG main
                SOURCE_SUBDIR do_not_find_cmake # To make sure CMakeLists.txt won't run
            )
            FetchContent_MakeAvailable(openclhpp)
            target_include_directories(dep_opencl INTERFACE $<BUILD_INTERFACE:${openclhpp_SOURCE_DIR}/include>)
        endif()
        set(dep_opencl_TARGET_VERSION ${OpenCL_VERSION_MAJOR}${OpenCL_VERSION_MINOR}0)
    endif()
    target_link_libraries(dep_opencl INTERFACE OpenCL::OpenCL)
    message(STATUS "dep: opencl target version ${dep_opencl_TARGET_VERSION}.")
    target_compile_definitions(dep_opencl INTERFACE
        CL_HPP_TARGET_OPENCL_VERSION=${dep_opencl_TARGET_VERSION}
        CL_HPP_MINIMUM_OPENCL_VERSION=110
    )
    add_library(dep::opencl ALIAS dep_opencl)
endif()
