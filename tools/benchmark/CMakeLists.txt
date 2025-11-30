project(ac_tools_benchmark VERSION 1.0.0.0 LANGUAGES CXX)

set(TOOLS_BENCHMARK_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TOOLS_BENCHMARK_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_executable(ac_benchmark ${TOOLS_BENCHMARK_SOURCE_DIR}/src/Benchmark.cpp)

target_link_libraries(ac_benchmark PRIVATE ac ac_util_misc ac_util_threads)

ac_check_enable_static_crt(ac_benchmark)

install(
    TARGETS ac_benchmark
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
