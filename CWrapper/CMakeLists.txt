project(ac LANGUAGES CXX)

if(Build_C_Wrapper)

    aux_source_directory(src SOURCE)

    include_directories(include ${TOP_DIR}/Anime4KCore/include)

    if(Build_C_Wrapper_With_Core)
        aux_source_directory(${TOP_DIR}/Anime4KCore/src Anime4KCPPCoreSource)
        list(APPEND SOURCE ${Anime4KCPPCoreSource})
        add_definitions(-DBUILT_IN_KERNEL)
        if(Use_Legacy_OpenCL_API)
            add_definitions(-DLEGACY_OPENCL_API)
        endif()
    endif()

    file(GLOB INCLUDE include/*.h)

    if(Enable_Video)
        add_definitions(-DENABLE_VIDEO)
    endif()

    if(Enable_Preview_GUI)
        add_definitions(-DENABLE_PREVIEW_GUI)
    endif()

    add_library(${PROJECT_NAME} SHARED ${INCLUDE} ${SOURCE})

    include(${TOP_DIR}/cmake/ThirdPartyForCWrapper.cmake)  

    install(
        TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION CWrapper/lib
        LIBRARY DESTINATION CWrapper/lib
        ARCHIVE DESTINATION CWrapper/lib
    )
    install(
        FILES include/AC.h  DESTINATION CWrapper/include
    )
    
endif()
