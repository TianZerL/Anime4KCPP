if(NOT TARGET dep::stb)
    add_library(dep_stb INTERFACE IMPORTED)
    message(STATUS "dep: fetch stb online.")
    include(FetchContent)
    FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG 013ac3beddff3dbffafd5177e7972067cd2b5083 # temporarily fix mingw compilation issue, see https://github.com/nothings/stb/issues/1663
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        TEST_COMMAND ""
        INSTALL_COMMAND ""
    )
    FetchContent_MakeAvailable(stb)
    target_include_directories(dep_stb INTERFACE $<BUILD_INTERFACE:${stb_SOURCE_DIR}>)
    add_library(dep::stb ALIAS dep_stb)
endif()
