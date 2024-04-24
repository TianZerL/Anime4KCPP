project(ac_filter_vs VERSION 1.0.0.0 LANGUAGES CXX)

set(FILTER_VAPOURSYNTH_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FILTER_VAPOURSYNTH_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include(${DEPENDENCY_DIR}/vapoursynth.cmake)

add_library(ac_filter_vs SHARED
    ${FILTER_VAPOURSYNTH_SOURCE_DIR}/src/Filter.cpp
)

target_link_libraries(ac_filter_vs PRIVATE ac dep::vapoursynth)

install(
    TARGETS ac_filter_vs
    RUNTIME DESTINATION filter
)
