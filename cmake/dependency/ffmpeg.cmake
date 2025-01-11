if(NOT TARGET dep::ffmpeg)
    add_library(dep_ffmpeg INTERFACE IMPORTED)
    if(AC_PATH_FFMPEG)
        find_path(dep_ffmpeg_INCLUDE
            NAMES libavcodec libavformat libavutil libswscale
            HINTS ${AC_PATH_FFMPEG}/include
            REQUIRED
        )
        find_library(dep_ffmpeg_AVCODEC
            NAMES avcodec
            HINTS ${AC_PATH_FFMPEG}/lib
            REQUIRED
        )
        find_library(dep_ffmpeg_AVFORMAT
            NAMES avformat
            HINTS ${AC_PATH_FFMPEG}/lib
            REQUIRED
        )
        find_library(dep_ffmpeg_AVUTIL
            NAMES avutil
            HINTS ${AC_PATH_FFMPEG}/lib
            REQUIRED
        )
        find_library(dep_ffmpeg_SWSCALE
            NAMES swscale
            HINTS ${AC_PATH_FFMPEG}/lib
            REQUIRED
        )
        target_include_directories(dep_ffmpeg INTERFACE $<BUILD_INTERFACE:${dep_ffmpeg_INCLUDE}>)
        target_link_libraries(dep_ffmpeg INTERFACE
            ${dep_ffmpeg_AVCODEC}
            ${dep_ffmpeg_AVFORMAT}
            ${dep_ffmpeg_AVUTIL}
            ${dep_ffmpeg_SWSCALE}
        )
    else()
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
            # ffmpeg 4.0 at least
            libavcodec>=58.18.100
            libavformat>=58.12.100
            libavutil>=56.14.100
            libswscale>=5.1.100
        )
        target_link_libraries(dep_ffmpeg INTERFACE PkgConfig::LIBAV)
    endif()
    add_library(dep::ffmpeg ALIAS dep_ffmpeg)
endif()
