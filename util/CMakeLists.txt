project(ac_util VERSION 3.0.0.0 LANGUAGES CXX)

set(UTIL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(UTIL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_library(ac_util INTERFACE)

target_include_directories(ac_util INTERFACE
    $<BUILD_INTERFACE:${UTIL_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:ac/include>
)

set_target_properties(ac_util PROPERTIES EXPORT_NAME "Util")

install(TARGETS ac_util EXPORT AC)

install(DIRECTORY ${UTIL_SOURCE_DIR}/include DESTINATION ac)
