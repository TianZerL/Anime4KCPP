project(ac_filter_ds VERSION 3.0.0.0 LANGUAGES CXX)

set(FILTER_DIRECTSHOW_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FILTER_DIRECTSHOW_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include(${DEPENDENCY_DIR}/directshow.cmake)
include(${CMAKE_DIR}/GenDSFilterRC.cmake)

add_library(ac_filter_ds SHARED
    ${FILTER_DIRECTSHOW_SOURCE_DIR}/src/Filter.cpp
    ${FILTER_DIRECTSHOW_BINARY_DIR}/rc/Filter.rc
    ${FILTER_DIRECTSHOW_BINARY_DIR}/rc/Filter.def
)

target_include_directories(ac_filter_ds PRIVATE ${FILTER_DIRECTSHOW_BINARY_DIR}/include)

target_link_libraries(ac_filter_ds PRIVATE ac dep::directshow)

install(
    TARGETS ac_filter_ds
    RUNTIME DESTINATION filter
)
