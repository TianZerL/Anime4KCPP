project(ac_test_core VERSION 1.0.0.0 LANGUAGES CXX)

set(TEST_CORE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TEST_CORE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_executable(ac_test_core_benchmark ${TEST_CORE_SOURCE_DIR}/src/Benchmark.cpp)

target_link_libraries(ac_test_core_benchmark PRIVATE ac ac_util)

ac_check_enable_static_crt(ac_test_core_benchmark)
