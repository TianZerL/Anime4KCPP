project(ac_binding_c VERSION 3.0.0.0 LANGUAGES CXX)

set(BINDING_C_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BINDING_C_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_library(cac SHARED
    ${BINDING_C_SOURCE_DIR}/src/Binding.cpp
)
add_library(CAC::Core ALIAS cac)

target_include_directories(cac PUBLIC
    $<BUILD_INTERFACE:${BINDING_C_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${BINDING_C_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:cac/include>
)

target_link_libraries(cac PRIVATE ac)

set_target_properties(cac PROPERTIES EXPORT_NAME "Core")

include(GenerateExportHeader)
generate_export_header(cac
    BASE_NAME "CAC"
    EXPORT_FILE_NAME ${BINDING_C_BINARY_DIR}/include/CACExport.h
)

install(
    TARGETS cac EXPORT CAC
    ARCHIVE DESTINATION cac/lib
    LIBRARY DESTINATION cac/lib
    RUNTIME DESTINATION cac/lib
)

install(DIRECTORY ${BINDING_C_SOURCE_DIR}/include ${BINDING_C_BINARY_DIR}/include DESTINATION cac)

ac_check_disable_flags(cac)
ac_check_enable_static_crt(cac)

install(
    EXPORT CAC
    FILE CACConfig.cmake
    NAMESPACE CAC::
    DESTINATION cac/cmake
)
