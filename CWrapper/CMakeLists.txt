project(ac LANGUAGES CXX)

if(Build_C_wrapper)

    aux_source_directory(src SOURCE)

    include_directories(include ${TOP_DIR}/Anime4KCore/include)

    file(GLOB INCLUDE include/*.h)

    add_library(${PROJECT_NAME} SHARED ${INCLUDE} ${SOURCE})

    include(${TOP_DIR}/cmake/ThirdPartyForCWrapper.cmake)  

endif()
