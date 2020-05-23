project(Anime4KCPPCore LANGUAGES CXX)

aux_source_directory(src SOURCE)

include_directories(include)
file(GLOB INCLUDE include/*.h)

if(Built_in_kernel)
    add_compile_definitions(BUILT_IN_KERNEL) 
endif()

add_library(${PROJECT_NAME} SHARED ${INCLUDE} ${SOURCE})

include(${TOP_DIR}/cmake/ThirdPartyForCore.cmake)
