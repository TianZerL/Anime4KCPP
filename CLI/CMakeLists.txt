project(Anime4KCPP_CLI)

find_package(OpenCL REQUIRED)

aux_source_directory(src SOURCE)

include_directories(../Anime4KCore/include ../ThirdParty/include ${OpenCL_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ${SOURCE})
target_link_libraries(${PROJECT_NAME} Anime4KCore)
