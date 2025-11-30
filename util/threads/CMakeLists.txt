set(UTIL_THREADS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(UTIL_THREADS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include(${DEPENDENCY_DIR}/threads.cmake)

add_library(ac_util_threads INTERFACE)
add_library(AC::Util::Threads ALIAS ac_util_threads)

target_include_directories(ac_util_threads INTERFACE
    $<BUILD_INTERFACE:${UTIL_THREADS_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(ac_util_threads INTERFACE dep::threads)

set_target_properties(ac_util_threads PROPERTIES EXPORT_NAME "Util::Threads")

install(TARGETS ac_util_threads EXPORT AC)

install(DIRECTORY ${UTIL_THREADS_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
