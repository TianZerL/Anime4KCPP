project(ac_video VERSION 1.1.0.0 LANGUAGES CXX)

set(VIDEO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(VIDEO_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include(${DEPENDENCY_DIR}/ffmpeg.cmake)

if(AC_SHARED_LIB)
    add_library(ac_video SHARED)
else()
    add_library(ac_video STATIC)
endif()
add_library(AC::Video ALIAS ac_video)

target_sources(ac_video PRIVATE
    ${VIDEO_SOURCE_DIR}/src/Pipeline.cpp
    ${VIDEO_SOURCE_DIR}/src/Filter.cpp
)

target_include_directories(ac_video PUBLIC
    $<BUILD_INTERFACE:${VIDEO_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${VIDEO_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(ac_video PRIVATE ac_util_misc ac_util_threads dep::ffmpeg)

target_compile_definitions(ac_video PUBLIC
    AC_VIDEO_VERSION_STR="${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
)

ac_check_disable_flags(ac_video)
ac_check_enable_static_crt(ac_video)
ac_check_disable_pic(ac_video)

set_target_properties(ac_video PROPERTIES EXPORT_NAME "Video")

include(GenerateExportHeader)
generate_export_header(ac_video
    BASE_NAME "AC_VIDEO"
    EXPORT_FILE_NAME ${VIDEO_BINARY_DIR}/include/ACVideoExport.hpp
)

install(
    TARGETS ac_video EXPORT AC
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY ${VIDEO_SOURCE_DIR}/include/ ${VIDEO_BINARY_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
