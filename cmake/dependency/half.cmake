if(NOT TARGET dep::half)
    add_library(dep_half INTERFACE IMPORTED)
    if(AC_PATH_HALF)
        set(dep_half_PATH ${AC_PATH_HALF})
        find_path(dep_half_INCLUDE
            NAMES half.hpp
            PATHS ${dep_half_PATH}
        )
    endif()
    if(dep_half_INCLUDE)
        target_include_directories(dep_half INTERFACE $<BUILD_INTERFACE:${dep_half_INCLUDE}>)
    else()
        message(STATUS "dep: half not found, will be fetched online.")
        include(FetchContent)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
            FetchContent_Declare(
                half
                URL https://sourceforge.net/projects/half/files/latest/download
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            )
        else()
            FetchContent_Declare(
                half
                URL https://sourceforge.net/projects/half/files/latest/download
            )
        endif()
        FetchContent_MakeAvailable(half)
        target_include_directories(dep_half INTERFACE $<BUILD_INTERFACE:${half_SOURCE_DIR}/include>)
    endif()
    add_library(dep::half ALIAS dep_half)
endif()
