project(ac_test_video VERSION 1.0.0.0 LANGUAGES CXX)

set(TEST_VIDEO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TEST_VIDEO_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_executable(ac_test_video_resize ${TEST_VIDEO_SOURCE_DIR}/src/Resize.cpp)

target_link_libraries(ac_test_video_resize PRIVATE ac ac_util_misc ac_video)

ac_check_enable_static_crt(ac_test_video_resize)
