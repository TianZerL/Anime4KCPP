project(Anime4KCPPCore)

aux_source_directory(src SOURCE)

include_directories(include)
file(GLOB INCLUDE include/*.h)

add_library(${PROJECT_NAME} SHARED ${INCLUDE} ${SOURCE})

include(../cmake/ThirdParty.cmake)
