project(ac LANGUAGES CXX)

if(Build_C_wrapper)

    aux_source_directory(src SOURCE)

    include_directories(include ${TOP_DIR}/Anime4KCore/include)

    if(Build_C_wrapper_with_core)
        aux_source_directory(${TOP_DIR}/Anime4KCore/src Anime4KCPPCoreSource)
        list(APPEND SOURCE ${Anime4KCPPCoreSource})
        add_definitions(-DBUILT_IN_KERNEL)
    endif()

    file(GLOB INCLUDE include/*.h)

    add_library(${PROJECT_NAME} SHARED ${INCLUDE} ${SOURCE})

    include(${TOP_DIR}/cmake/ThirdPartyForCWrapper.cmake)  

endif()
