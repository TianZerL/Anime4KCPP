project(Anime4K09)

aux_source_directory(src SOURCE)
file(GLOB INCLUDE include/*.h)

include_directories(include ../ThirdParty/include)

add_executable(${PROJECT_NAME} ${SOURCE} ${INCLUDE})

include(../cmake/ThirdParty.cmake)