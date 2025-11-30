set(UTIL_PARALLEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(UTIL_PARALLEL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include(${DEPENDENCY_DIR}/parallel.cmake)

add_library(ac_util_parallel INTERFACE)
add_library(AC::Util::Parallel ALIAS ac_util_parallel)

target_include_directories(ac_util_parallel INTERFACE
    $<BUILD_INTERFACE:${UTIL_PARALLEL_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(ac_util_parallel INTERFACE
    dep::parallel
    $<$<BOOL:${AC_DEP_PARALLEL_THREADPOOL}>:ac_util_threads>
)

set_target_properties(ac_util_parallel PROPERTIES EXPORT_NAME "Util::Parallel")

install(TARGETS ac_util_parallel EXPORT AC)

install(DIRECTORY ${UTIL_PARALLEL_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
