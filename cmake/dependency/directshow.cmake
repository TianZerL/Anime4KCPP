if(NOT TARGET dep::directshow)
    add_library(dep_directshow STATIC)
    if(AC_PATH_DIRECTSHOW_BASECLASSES)
        set(dep_directshow_PATH ${AC_PATH_DIRECTSHOW_BASECLASSES})
    endif()
    find_path(dep_directshow_INCLUDE
        NAMES streams.h
        PATHS ${dep_directshow_PATH}
    )
    if(dep_directshow_INCLUDE)
        set(directshow_baseclasses_SOURCE_DIR ${dep_directshow_INCLUDE})
    else()
        message(STATUS "dep: directshow baseclasses not found, will be fetched online.")
        include(FetchContent)
        FetchContent_Declare(
            directshow
            GIT_REPOSITORY https://github.com/microsoft/Windows-classic-samples.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(directshow)
        set(directshow_baseclasses_SOURCE_DIR ${directshow_SOURCE_DIR}/Samples/Win7Samples/multimedia/directshow/baseclasses)
    endif()
    file(GLOB dep_directshow_baseclasses_src "${directshow_baseclasses_SOURCE_DIR}/*.cpp")
    target_sources(dep_directshow PRIVATE ${dep_directshow_baseclasses_src})
    target_include_directories(dep_directshow PUBLIC $<BUILD_INTERFACE:${directshow_baseclasses_SOURCE_DIR}>)
    target_link_libraries(dep_directshow PRIVATE strmiids winmm)
    ac_check_enable_static_crt(dep_directshow)
    add_library(dep::directshow ALIAS dep_directshow)
endif()
