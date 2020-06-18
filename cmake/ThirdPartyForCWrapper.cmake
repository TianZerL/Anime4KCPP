find_package(OpenCL REQUIRED)

include_directories(${TOP_DIR}/ThirdParty/include ${OpenCL_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} Anime4KCPPCore)
