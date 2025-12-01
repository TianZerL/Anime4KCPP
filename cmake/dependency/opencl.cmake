if(NOT TARGET dep::opencl)
    add_library(dep_opencl INTERFACE IMPORTED)
    if (AC_FETCH_OPENCL)
        message(STATUS "dep: opencl sdk will be fetched online.")
        set(dep_opencl_FETCH_SDK TRUE)
    else()
        find_package(OpenCL QUIET)
        if(NOT OpenCL_FOUND)
            message(STATUS "dep: opencl sdk not found, will be fetched online.")
            set(dep_opencl_FETCH_SDK TRUE)
        else()
            include(${CMAKE_DIR}/CheckOpenCLHPP.cmake)
            message(STATUS "dep: find opencl lib: ${OpenCL_LIBRARIES} headers: ${OpenCL_INCLUDE_DIRS}")
            check_opencl_hpp_support(dep_opencl_SUPPORT_HPP OpenCL::OpenCL)
            if(NOT dep_opencl_SUPPORT_HPP)
                message(STATUS "dep: system opencl sdk is incompatible, will fetch headers online, consider regenerating with 'AC_FETCH_OPENCL=TRUE' to fetch whole opencl sdk.")
                include(FetchContent)
                set(OPENCL_HEADERS_BUILD_TESTING OFF CACHE BOOL "OpenCL SDK option" FORCE)
                set(OPENCL_HEADERS_BUILD_CXX_TESTS OFF CACHE BOOL "OpenCL SDK option" FORCE)
                set(BUILD_DOCS OFF CACHE BOOL "OpenCL SDK option" FORCE)
                set(BUILD_EXAMPLES OFF CACHE BOOL "OpenCL SDK option" FORCE)
                set(OPENCL_CLHPP_BUILD_TESTING OFF CACHE BOOL "OpenCL SDK option" FORCE)
                if (CMAKE_VERSION VERSION_LESS 3.28)
                    FetchContent_Declare(
                        openclheaders
                        GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
                        GIT_TAG main
                    )
                    FetchContent_GetProperties(openclheaders)
                    if(NOT openclheaders_POPULATED)
                        FetchContent_Populate(openclheaders)
                        add_subdirectory(${openclheaders_SOURCE_DIR} ${openclheaders_BINARY_DIR} EXCLUDE_FROM_ALL)
                    endif()

                    FetchContent_Declare(
                        openclheaderscpp
                        GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-CLHPP.git
                        GIT_TAG main
                    )
                    FetchContent_GetProperties(openclheaderscpp)
                    if(NOT openclheaderscpp_POPULATED)
                        FetchContent_Populate(openclheaderscpp)
                        add_subdirectory(${openclheaderscpp_SOURCE_DIR} ${openclheaderscpp_BINARY_DIR} EXCLUDE_FROM_ALL)
                    endif()
                else()
                    FetchContent_Declare(
                        openclheaders
                        GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
                        GIT_TAG main
                        EXCLUDE_FROM_ALL
                    )
                    FetchContent_MakeAvailable(openclheaders)

                    FetchContent_Declare(
                        openclheaderscpp
                        GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-CLHPP.git
                        GIT_TAG main
                        EXCLUDE_FROM_ALL
                    )
                    FetchContent_MakeAvailable(openclheaderscpp)
                endif()
                target_link_libraries(dep_opencl INTERFACE OpenCL::HeadersCpp ${OpenCL_LIBRARIES})
            else()
                target_link_libraries(dep_opencl INTERFACE OpenCL::OpenCL)
            endif()
            if(OpenCL_VERSION_MAJOR)
                set(dep_opencl_TARGET_VERSION "${OpenCL_VERSION_MAJOR}${OpenCL_VERSION_MINOR}0")
            endif()
        endif()
    endif()

    if(dep_opencl_FETCH_SDK)
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
        include(FetchContent)
        if (CMAKE_VERSION VERSION_LESS 3.28)
            FetchContent_Declare(
                opencl
                GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-SDK.git
                GIT_TAG main
            )
            FetchContent_GetProperties(opencl)
            if(NOT opencl_POPULATED)
                FetchContent_Populate(opencl)
                add_subdirectory(${opencl_SOURCE_DIR} ${opencl_BINARY_DIR} EXCLUDE_FROM_ALL)
            endif()
        else()
            FetchContent_Declare(
                opencl
                GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-SDK.git
                GIT_TAG main
                EXCLUDE_FROM_ALL
            )
            FetchContent_MakeAvailable(opencl)
        endif()
        target_link_libraries(dep_opencl INTERFACE OpenCL::HeadersCpp OpenCL::OpenCL)
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
