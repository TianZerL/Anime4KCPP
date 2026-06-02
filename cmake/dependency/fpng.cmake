if(NOT TARGET dep::fpng)
    add_library(dep_fpng STATIC)
    if(AC_PATH_FPNG)
        set(dep_fpng_PATH ${AC_PATH_FPNG})
        find_path(dep_fpng_SOURCE
            NAMES fpng.h fpng.cpp
            PATHS ${dep_fpng_PATH}
        )
    endif()
    if(dep_fpng_SOURCE)
        target_sources(dep_fpng PRIVATE ${dep_fpng_SOURCE}/fpng.cpp)
        target_include_directories(dep_fpng PUBLIC $<BUILD_INTERFACE:${dep_fpng_SOURCE}>)
    else()
        message(STATUS "dep: fpng not found, will be fetched online.")
        include(FetchContent)
        FetchContent_Declare(
            fpng
            GIT_REPOSITORY https://github.com/richgel999/fpng.git
            GIT_TAG main
            SOURCE_SUBDIR do_not_find_cmake # To make sure CMakeLists.txt won't run
        )
        FetchContent_MakeAvailable(fpng)
        target_sources(dep_fpng PRIVATE ${fpng_SOURCE_DIR}/src/fpng.cpp)
        target_include_directories(dep_fpng PUBLIC $<BUILD_INTERFACE:${fpng_SOURCE_DIR}/src>)
    endif()
    if(AC_COMPILER_SUPPORT_SSE41 AND AC_COMPILER_SUPPORT_PCLMUL)
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC")
            if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                target_compile_options(dep_fpng PRIVATE /clang:-msse4.1 /clang:-mpclmul)
            endif()
        elseif(NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            target_compile_options(dep_fpng PRIVATE -msse4.1 -mpclmul -fno-strict-aliasing)
        endif()
    else()
        target_compile_definitions(dep_fpng PRIVATE FPNG_NO_SSE=1)
        if(NOT (CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
            target_compile_options(dep_fpng PRIVATE -fno-strict-aliasing)
        endif()
    endif()
    if (NOT WIN32 AND NOT AC_DISABLE_PIC)
        set_target_properties(dep_fpng PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
    add_library(dep::fpng ALIAS dep_fpng)
endif()
