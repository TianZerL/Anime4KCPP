project(ac LANGUAGES CXX)

if(Build_C_Wrapper)

    aux_source_directory(src SOURCE)
    aux_source_directory(${TOP_DIR}/Anime4KCore/src Anime4KCPPCoreSource)

    include_directories(include ${TOP_DIR}/Anime4KCore/include)

    add_definitions(-DBUILT_IN_KERNEL)

    if(Use_Legacy_OpenCL_API)
        add_definitions(-DLEGACY_OPENCL_API)
    endif()

    file(GLOB INCLUDE include/*.h)

    if(Enable_Video)
        add_definitions(-DENABLE_VIDEO)
    endif()

    if(Enable_Preview_GUI)
        add_definitions(-DENABLE_PREVIEW_GUI)
    endif()

    if(Enable_Image_IO)
        add_definitions(-DENABLE_IMAGE_IO)
    endif()

    if(Build_Static_C_Wrapper)
        add_library(${PROJECT_NAME} STATIC ${INCLUDE} ${SOURCE} ${Anime4KCPPCoreSource})
        set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS -DAC_STATIC_DEFINE)
    else()
        add_library(${PROJECT_NAME} SHARED ${INCLUDE} ${SOURCE} ${Anime4KCPPCoreSource})
    endif()

    include(${TOP_DIR}/cmake/ThirdPartyForCWrapper.cmake)  

    install(
        TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION CWrapper/lib
        LIBRARY DESTINATION CWrapper/lib
        ARCHIVE DESTINATION CWrapper/lib
    )
    install(
        FILES include/AC.h ${CMAKE_CURRENT_BINARY_DIR}/ac_export.h  DESTINATION CWrapper/include
    )
    
endif()
