project(ac_filter_ds VERSION 3.1.0.0 LANGUAGES CXX)

set(FILTER_DIRECTSHOW_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FILTER_DIRECTSHOW_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include(${DEPENDENCY_DIR}/directshow.cmake)
include(${CMAKE_DIR}/GenDSFilterRC.cmake)

add_library(ac_filter_ds MODULE
    ${FILTER_DIRECTSHOW_SOURCE_DIR}/src/Filter.cpp
    ${FILTER_DIRECTSHOW_SOURCE_DIR}/src/Registry.cpp
    ${FILTER_DIRECTSHOW_SOURCE_DIR}/src/SideData.cpp
    ${FILTER_DIRECTSHOW_BINARY_DIR}/rc/Filter.rc
    ${FILTER_DIRECTSHOW_BINARY_DIR}/rc/Filter.def
)

target_include_directories(ac_filter_ds PRIVATE
    $<BUILD_INTERFACE:${FILTER_DIRECTSHOW_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${FILTER_DIRECTSHOW_BINARY_DIR}/include>
)

target_link_libraries(ac_filter_ds PRIVATE ac dep::directshow)

ac_check_enable_static_crt(ac_filter_ds)

install(
    TARGETS ac_filter_ds
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
