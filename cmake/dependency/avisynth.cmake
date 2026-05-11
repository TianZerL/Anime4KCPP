if(NOT TARGET dep::avisynth)
    add_library(dep_avisynth INTERFACE IMPORTED)
    if(AC_PATH_AVISYNTH_SDK)
        set(dep_avisynth_PATH ${AC_PATH_AVISYNTH_SDK}/include)
        find_path(dep_avisynth_INCLUDE
            NAMES avisynth.h
            PATHS ${dep_avisynth_PATH}
        )
    endif()
    if(dep_avisynth_INCLUDE)
        target_include_directories(dep_avisynth INTERFACE $<BUILD_INTERFACE:${dep_avisynth_INCLUDE}>)
    else()
        message(STATUS "dep: avisynth sdk not found, will be fetched online.")
        include(FetchContent)
        FetchContent_Declare(
            avisynth
            GIT_REPOSITORY https://github.com/AviSynth/AviSynthPlus.git
            GIT_TAG master
            SOURCE_SUBDIR do_not_find_cmake # To make sure CMakeLists.txt won't run
        )
        FetchContent_MakeAvailable(avisynth)
        target_include_directories(dep_avisynth INTERFACE $<BUILD_INTERFACE:${avisynth_SOURCE_DIR}/avs_core/include>)
    endif()
    add_library(dep::avisynth ALIAS dep_avisynth)
endif()
