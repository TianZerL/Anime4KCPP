project(ac_test_util VERSION 1.0.0.0 LANGUAGES CXX)

set(TEST_UTIL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TEST_UTIL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_executable(ac_test_util_channel ${TEST_UTIL_SOURCE_DIR}/src/Channel.cpp)

target_link_libraries(ac_test_util_channel PRIVATE ac_util_misc ac_util_threads ac_util_parallel)

ac_check_enable_static_crt(ac_test_util_channel)
