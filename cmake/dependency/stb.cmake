if(NOT TARGET dep::stb)
    include(FetchContent)
    FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG master
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        TEST_COMMAND ""
        INSTALL_COMMAND ""
    )
    FetchContent_MakeAvailable(stb)

    add_library(dep_stb INTERFACE IMPORTED)
    target_include_directories(dep_stb INTERFACE
        $<BUILD_INTERFACE:${stb_SOURCE_DIR}>
    )
    add_library(dep::stb ALIAS dep_stb)
endif()
