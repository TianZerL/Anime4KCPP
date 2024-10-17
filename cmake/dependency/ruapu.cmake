if(NOT TARGET dep::ruapu)
    add_library(dep_ruapu INTERFACE IMPORTED)
    message(STATUS "dep: fetch ruapu online.")
    include(FetchContent)
    FetchContent_Declare(
        ruapu
        GIT_REPOSITORY https://github.com/nihui/ruapu.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(ruapu)
    target_include_directories(dep_ruapu INTERFACE $<BUILD_INTERFACE:${ruapu_SOURCE_DIR}>)
    add_library(dep::ruapu ALIAS dep_ruapu)
endif()
