project(ac_filter_avs VERSION 3.1.0.0 LANGUAGES CXX)

set(FILTER_AVISYNTH_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FILTER_AVISYNTH_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include(${DEPENDENCY_DIR}/avisynth.cmake)

add_library(ac_filter_avs MODULE
    ${FILTER_AVISYNTH_SOURCE_DIR}/src/Filter.cpp
)

target_link_libraries(ac_filter_avs PRIVATE ac dep::avisynth)

ac_check_enable_static_crt(ac_filter_avs)

install(
    TARGETS ac_filter_avs
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
