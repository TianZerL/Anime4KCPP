if(NOT TARGET dep::opencl)
    add_library(dep_opencl INTERFACE IMPORTED)
    find_package(OpenCL QUIET)
    if(NOT OpenCL_FOUND)
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
        target_link_libraries(dep_opencl INTERFACE OpenCL::HeadersCpp OpenCL::OpenCL)
    else()
        include(${CMAKE_DIR}/CheckOpenCLHPP.cmake)
        if(NOT dep_opencl_SUPPORT_HPP)
            message(STATUS "dep: opencl hpp headers not found or no supported, will be fetched online.")
            include(FetchContent)
            FetchContent_Declare(
                openclheaders
                GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
                GIT_TAG main
            )
            set(OPENCL_HEADERS_BUILD_TESTING OFF CACHE BOOL "OpenCL SDK option" FORCE)
            set(OPENCL_HEADERS_BUILD_CXX_TESTS OFF CACHE BOOL "OpenCL SDK option" FORCE)
            FetchContent_MakeAvailable(openclheaders)
            FetchContent_Declare(
                openclhpp
                GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-CLHPP.git
                GIT_TAG main
            )
            set(BUILD_DOCS OFF CACHE BOOL "OpenCL SDK option" FORCE)
            set(BUILD_EXAMPLES OFF CACHE BOOL "OpenCL SDK option" FORCE)
            set(OPENCL_CLHPP_BUILD_TESTING OFF CACHE BOOL "OpenCL SDK option" FORCE)
            FetchContent_MakeAvailable(openclhpp)
            target_link_libraries(dep_opencl INTERFACE OpenCL::HeadersCpp ${OpenCL_LIBRARIES})
        else()
            target_link_libraries(dep_opencl INTERFACE OpenCL::OpenCL)
        endif()
        if(OpenCL_VERSION_STRING)
            set(dep_opencl_TARGET_VERSION "${OpenCL_VERSION_MAJOR}${OpenCL_VERSION_MINOR}0")
        endif()
    endif()
    if(NOT dep_opencl_TARGET_VERSION)
        set(dep_opencl_TARGET_VERSION 300)
    endif()
    message(STATUS "dep: opencl target version ${dep_opencl_TARGET_VERSION}.")
    target_compile_definitions(dep_opencl INTERFACE
        CL_HPP_TARGET_OPENCL_VERSION=${dep_opencl_TARGET_VERSION}
        CL_HPP_MINIMUM_OPENCL_VERSION=110 # set minimum version 110 to prevent compilation errors in certain versions of OpenCL HPP
    )
    add_library(dep::opencl ALIAS dep_opencl)
endif()
