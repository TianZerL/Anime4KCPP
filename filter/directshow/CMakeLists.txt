project(ac_filter_ds VERSION 3.0.0.0 LANGUAGES CXX)

set(FILTER_DIRECTSHOW_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FILTER_DIRECTSHOW_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include(${DEPENDENCY_DIR}/directshow.cmake)
include(${CMAKE_DIR}/GenDSFilterRC.cmake)

add_library(ac_filter_ds MODULE
    ${FILTER_DIRECTSHOW_SOURCE_DIR}/src/Filter.cpp
    ${FILTER_DIRECTSHOW_BINARY_DIR}/rc/Filter.rc
    ${FILTER_DIRECTSHOW_BINARY_DIR}/rc/Filter.def
)

target_include_directories(ac_filter_ds PRIVATE ${FILTER_DIRECTSHOW_BINARY_DIR}/include)

target_link_libraries(ac_filter_ds PRIVATE ac dep::directshow)

ac_check_enable_static_crt(ac_filter_ds)

install(
    TARGETS ac_filter_ds
    ARCHIVE DESTINATION filter
    LIBRARY DESTINATION filter
    RUNTIME DESTINATION filter
)
