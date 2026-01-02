project(ac_binding_c VERSION 3.1.0.0 LANGUAGES CXX)

set(BINDING_C_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BINDING_C_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_library(ac_c SHARED
    ${BINDING_C_SOURCE_DIR}/src/Binding.cpp
)
add_library(AC::Binding::C ALIAS ac_c)

target_include_directories(ac_c PUBLIC
    $<BUILD_INTERFACE:${BINDING_C_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${BINDING_C_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(ac_c PRIVATE ac)

set_target_properties(ac_c PROPERTIES EXPORT_NAME "Binding::C")

include(GenerateExportHeader)
generate_export_header(ac_c
    BASE_NAME "AC_C"
    EXPORT_FILE_NAME ${BINDING_C_BINARY_DIR}/include/ACCExport.h
)

install(
    TARGETS ac_c EXPORT AC
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY ${BINDING_C_SOURCE_DIR}/include/ ${BINDING_C_BINARY_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

ac_check_disable_flags(ac_c)
ac_check_enable_static_crt(ac_c)
