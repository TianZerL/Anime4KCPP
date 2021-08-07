target_link_libraries(${PROJECT_NAME} PRIVATE Anime4KCPPCore)

target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:c_api/include>)
