project(ac_test_wasm VERSION 1.0.0.0 LANGUAGES CXX)

set(TEST_WASM_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TEST_WASM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_executable(ac_test_wasm ${TEST_WASM_SOURCE_DIR}/src/Web.cpp)

target_link_libraries(ac_test_wasm PRIVATE ac ac_util_misc)
target_link_options(ac_test_wasm PRIVATE -lembind -sMODULARIZE -sNODERAWFS=1)
