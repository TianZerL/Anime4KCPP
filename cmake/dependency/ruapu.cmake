if(NOT TARGET dep::ruapu)
    add_library(dep_ruapu INTERFACE IMPORTED)
    if(AC_PATH_RUAPU)
        set(dep_ruapu_PATH ${AC_PATH_RUAPU})
    endif()
    find_path(dep_ruapu_INCLUDE
        NAMES ruapu.h
        PATHS ${dep_ruapu_PATH}
    )
    if(dep_ruapu_INCLUDE)
        target_include_directories(dep_ruapu INTERFACE $<BUILD_INTERFACE:${dep_ruapu_INCLUDE}>)
    else()
        message(STATUS "dep: ruapu not found, will be fetched online.")
        include(FetchContent)
        FetchContent_Declare(
            ruapu
            GIT_REPOSITORY https://github.com/nihui/ruapu.git
            GIT_TAG master
        )
        FetchContent_MakeAvailable(ruapu)
        target_include_directories(dep_ruapu INTERFACE $<BUILD_INTERFACE:${ruapu_SOURCE_DIR}>)
    endif()
    add_library(dep::ruapu ALIAS dep_ruapu)
endif()
