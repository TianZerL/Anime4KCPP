if(NOT TARGET dep::stb)
    add_library(dep_stb INTERFACE IMPORTED)
    if(AC_PATH_STB)
        set(dep_stb_PATH ${AC_PATH_STB})
    endif()
    find_path(dep_stb_INCLUDE
        NAMES stb_image.h
        PATHS ${dep_stb_PATH}
    )
    if(dep_stb_INCLUDE)
        target_include_directories(dep_stb INTERFACE $<BUILD_INTERFACE:${dep_stb_INCLUDE}>)
    else()
        message(STATUS "dep: stb not found, will be fetched online.")
        include(FetchContent)
        FetchContent_Declare(
            stb
            GIT_REPOSITORY https://github.com/nothings/stb.git
            GIT_TAG master
        )
        FetchContent_MakeAvailable(stb)
        target_include_directories(dep_stb INTERFACE $<BUILD_INTERFACE:${stb_SOURCE_DIR}>)
    endif()
    add_library(dep::stb ALIAS dep_stb)
endif()
