project(ac_tests_core LANGUAGES CXX)

set(TESTS_CORE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TESTS_CORE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_executable(ac_tests_core
    ${TESTS_CORE_SOURCE_DIR}/src/ImageTest.cpp
    ${TESTS_CORE_SOURCE_DIR}/src/ImageProcessTest.cpp
    ${TESTS_CORE_SOURCE_DIR}/src/ImageResizeTest.cpp
    ${TESTS_CORE_SOURCE_DIR}/src/ModelTest.cpp
    ${TESTS_CORE_SOURCE_DIR}/src/ProcessorTest.cpp
    ${TESTS_CORE_SOURCE_DIR}/src/UtilTest.cpp
)

target_link_libraries(ac_tests_core PRIVATE ac dep::doctest)
add_test(NAME core COMMAND ac_tests_core)
