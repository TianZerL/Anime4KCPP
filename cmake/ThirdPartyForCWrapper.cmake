include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME} BASE_NAME "AC_C")

target_include_directories(
    ${PROJECT_NAME} 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:c_api/include>
)

target_link_libraries(${PROJECT_NAME} PRIVATE Anime4KCPPCore)
