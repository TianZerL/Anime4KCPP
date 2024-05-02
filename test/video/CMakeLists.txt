project(ac_test_video VERSION 3.0.0.0 LANGUAGES CXX)

set(TEST_VIDEO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TEST_VIDEO_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_executable(ac_test_video ${TEST_VIDEO_SOURCE_DIR}/src/Video.cpp)

target_link_libraries(ac_test_video PRIVATE ac ac_util ac_video)
