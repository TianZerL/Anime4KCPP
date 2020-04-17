project(Anime4KCore)

aux_source_directory(src SOURCE)

include_directories(include)

add_library(${PROJECT_NAME} SHARED ${SOURCE})

include(../cmake/ThirdParty.cmake)