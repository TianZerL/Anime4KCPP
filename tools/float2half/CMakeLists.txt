project(ac_tools_float2half VERSION 1.0.0.0 LANGUAGES CXX)

set(TOOLS_FLOAT2HALF_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TOOLS_FLOAT2HALF_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_executable(ac_float2half ${TOOLS_FLOAT2HALF_SOURCE_DIR}/src/Float2Half.cpp)

target_include_directories(ac_float2half PRIVATE
    $<BUILD_INTERFACE:${TOP_DIR}/core/include>
)

ac_check_enable_static_crt(ac_float2half)
