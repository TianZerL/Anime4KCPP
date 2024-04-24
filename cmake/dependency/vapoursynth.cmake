if(NOT TARGET dep::vapoursynth)
    add_library(dep_vapoursynth INTERFACE IMPORTED)
    if(WIN32)
        if(AC_COMPILER_32BIT)
            set(AC_PATH_VAPOURSYNTH_LIB_SUBDIR lib32)
            set(AC_URL_VAPOURSYNTH_RELEASE https://github.com/vapoursynth/vapoursynth/releases/download/R63/VapourSynth32-Portable-R63.7z)
        else()
            set(AC_PATH_VAPOURSYNTH_LIB_SUBDIR lib64)
            set(AC_URL_VAPOURSYNTH_RELEASE https://github.com/vapoursynth/vapoursynth/releases/download/R66/VapourSynth64-Portable-R66.zip)
        endif()
        if(NOT AC_PATH_VAPOURSYNTH_SDK)
            FetchContent_Declare(
                vapoursynth
                URL ${AC_URL_VAPOURSYNTH_RELEASE}
                CONFIGURE_COMMAND ""
                BUILD_COMMAND ""
                TEST_COMMAND ""
            )
            FetchContent_MakeAvailable(vapoursynth)
            set(AC_PATH_VAPOURSYNTH_SDK ${vapoursynth_SOURCE_DIR}/sdk CACHE STRING "VapourSynth SDK path" FORCE)
        endif()
        find_library(dep_vapoursynth_LIBS 
            NAMES vapoursynth
            HINTS ${AC_PATH_VAPOURSYNTH_SDK}/${AC_PATH_VAPOURSYNTH_LIB_SUBDIR}
            REQUIRED
        )
        target_link_libraries(dep_vapoursynth INTERFACE ${dep_vapoursynth_LIBS})
        target_include_directories(dep_vapoursynth INTERFACE ${AC_PATH_VAPOURSYNTH_SDK}/include)
    elseif(UNIX)
        find_package(PkgConfig QUIET)
        if(PKG_CONFIG_FOUND)
            pkg_check_modules(VapourSynth REQUIRED IMPORTED_TARGET vapoursynth)
            target_link_libraries(dep_vapoursynth INTERFACE PkgConfig::VapourSynth)
        else()
            find_library(dep_vapoursynth_LIBS 
                NAMES vapoursynth
                REQUIRED
            )
            find_path(dep_vapoursynth_INCLUDE 
                NAMES VapourSynth4.h VSHelper4.h
                REQUIRED
            )
            target_link_libraries(dep_vapoursynth INTERFACE ${dep_vapoursynth_LIBS})
            target_include_directories(dep_vapoursynth INTERFACE ${dep_vapoursynth_INCLUDE})
        endif()
    endif()
    add_library(dep::vapoursynth ALIAS dep_vapoursynth)
endif()
