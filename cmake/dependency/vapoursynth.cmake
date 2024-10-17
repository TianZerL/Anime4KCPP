if(NOT TARGET dep::vapoursynth)
    add_library(dep_vapoursynth INTERFACE IMPORTED)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(VapourSynth QUIET IMPORTED_TARGET vapoursynth)
        if(VapourSynth_FOUND)
            target_link_libraries(dep_vapoursynth INTERFACE PkgConfig::VapourSynth)
        endif()
    endif()
    if(NOT PKG_CONFIG_FOUND OR NOT VapourSynth_FOUND)
        message(STATUS "dep: vapoursynth not found, will be fetched online.")
        include(FetchContent)
        FetchContent_Declare(
            vapoursynth
            GIT_REPOSITORY https://github.com/vapoursynth/vapoursynth.git
            GIT_TAG master
        )
        FetchContent_MakeAvailable(vapoursynth)
        target_include_directories(dep_vapoursynth INTERFACE $<BUILD_INTERFACE:${vapoursynth_SOURCE_DIR}/include>)
    endif()
    add_library(dep::vapoursynth ALIAS dep_vapoursynth)
endif()
