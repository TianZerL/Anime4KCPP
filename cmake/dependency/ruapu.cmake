if(NOT TARGET dep::ruapu)
    include(FetchContent)
    FetchContent_Declare(
        ruapu
        GIT_REPOSITORY https://github.com/nihui/ruapu.git
        GIT_TAG master
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        TEST_COMMAND ""
        INSTALL_COMMAND ""
    )
    FetchContent_MakeAvailable(ruapu)

    add_library(dep_ruapu INTERFACE IMPORTED)
    target_include_directories(dep_ruapu INTERFACE
        $<BUILD_INTERFACE:${ruapu_SOURCE_DIR}>
    )  
    add_library(dep::ruapu ALIAS dep_ruapu)
endif()
