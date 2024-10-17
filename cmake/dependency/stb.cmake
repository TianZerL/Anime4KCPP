if(NOT TARGET dep::stb)
    add_library(dep_stb INTERFACE IMPORTED)
    message(STATUS "dep: fetch stb online.")
    include(FetchContent)
    FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(stb)
    target_include_directories(dep_stb INTERFACE $<BUILD_INTERFACE:${stb_SOURCE_DIR}>)
    add_library(dep::stb ALIAS dep_stb)
endif()
