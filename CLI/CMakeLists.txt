project(Anime4KCPP_CLI)

aux_source_directory(src SOURCE)

include_directories(../Anime4KCore/include ../ThirdParty/include)

add_executable(${PROJECT_NAME} ${SOURCE})
target_link_libraries(${PROJECT_NAME} Anime4KCore)
