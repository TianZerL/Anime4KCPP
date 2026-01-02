project(ac_cli VERSION 3.1.0.0 LANGUAGES CXX)

set(CLI_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CLI_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include(${DEPENDENCY_DIR}/cli11.cmake)

add_executable(ac_cli
    ${CLI_SOURCE_DIR}/src/Main.cpp
    ${CLI_SOURCE_DIR}/src/Options.cpp
    ${CLI_SOURCE_DIR}/src/ProgressBar.cpp
)

target_include_directories(ac_cli PRIVATE
    $<BUILD_INTERFACE:${CLI_SOURCE_DIR}/include>
)

target_link_libraries(ac_cli PRIVATE
    ac
    ac_util_misc ac_util_threads
    $<$<TARGET_EXISTS:ac_video>:ac_video>
    dep::cli11
)

target_compile_definitions(ac_cli PRIVATE
    $<$<TARGET_EXISTS:ac_video>:AC_CLI_ENABLE_VIDEO>
)

ac_check_enable_static_crt(ac_cli)

install(
    TARGETS ac_cli
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
