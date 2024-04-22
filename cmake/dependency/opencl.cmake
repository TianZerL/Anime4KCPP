if(NOT TARGET dep::opencl)
    find_package(OpenCL QUIET)

    if (NOT OpenCL_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            OpenCL
            GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-SDK.git
            GIT_TAG main
        )
        set(BUILD_DOCS OFF)
        set(BUILD_EXAMPLES OFF)
        set(BUILD_TESTING OFF)
        set(ENABLE_OPENCL_LAYERINFO OFF)
        set(OPENCL_ICD_LOADER_BUILD_TESTING OFF)
        set(OPENCL_SDK_BUILD_OPENGL_SAMPLES OFF)
        set(OPENCL_SDK_BUILD_SAMPLES OFF)
        set(OPENCL_SDK_BUILD_UTILITY_LIBRARIES OFF)
        set(OPENCL_CLHPP_BUILD_TESTING OFF)
        set(OPENCL_HEADERS_BUILD_TESTING OFF)
        set(OPENCL_HEADERS_BUILD_CXX_TESTS OFF)
        FetchContent_MakeAvailable(OpenCL)
        target_link_libraries(OpenCL PUBLIC OpenCL::HeadersCpp)
    endif()

    add_library(dep_opencl INTERFACE)
    target_link_libraries(dep_opencl INTERFACE OpenCL::OpenCL)
    target_compile_definitions(dep_opencl INTERFACE 
        CL_HPP_TARGET_OPENCL_VERSION=300
        CL_HPP_MINIMUM_OPENCL_VERSION=110
    )
    add_library(dep::opencl ALIAS dep_opencl)
endif()
