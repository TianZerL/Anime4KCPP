set(UTIL_MISC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(UTIL_MISC_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_library(ac_util_misc INTERFACE)
add_library(AC::Util::Misc ALIAS ac_util_misc)

target_include_directories(ac_util_misc INTERFACE
    $<BUILD_INTERFACE:${UTIL_MISC_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set_target_properties(ac_util_misc PROPERTIES EXPORT_NAME "Util::Misc")

install(TARGETS ac_util_misc EXPORT AC)

install(DIRECTORY ${UTIL_MISC_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
