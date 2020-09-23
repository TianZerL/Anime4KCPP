project(Anime4KCPPCore LANGUAGES CXX)

aux_source_directory(src SOURCE)

include_directories(include)
file(GLOB INCLUDE include/*.h)

if(Built_in_kernel)
    add_definitions(-DBUILT_IN_KERNEL)
endif()

if(Use_TBB)
    add_definitions(-DUSE_TBB)
endif()

add_library(${PROJECT_NAME} SHARED ${INCLUDE} ${SOURCE})

include(${TOP_DIR}/cmake/ThirdPartyForCore.cmake)
