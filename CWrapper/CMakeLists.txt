project(ac LANGUAGES CXX)

if(Build_C_Wrapper)

    aux_source_directory(src SOURCE)

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
        add_library(${PROJECT_NAME} STATIC ${INCLUDE} ${SOURCE})
        set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS -DAC_STATIC_DEFINE)
    else()
        add_library(${PROJECT_NAME} SHARED ${INCLUDE} ${SOURCE})
    endif()

    include(${TOP_DIR}/cmake/ThirdPartyForCWrapper.cmake)  

    install(
        TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}
        ARCHIVE DESTINATION c_api/lib
        LIBRARY DESTINATION c_api/lib
        RUNTIME DESTINATION c_api/bin
    )
    
    install(
        FILES ${INCLUDE} ${CMAKE_CURRENT_BINARY_DIR}/ac_c_export.h
        DESTINATION c_api/include
    )

    install(EXPORT ${PROJECT_NAME} DESTINATION c_api/cmake)
    
    configure_file(${TOP_DIR}/cmake/CWrapperConfig.cmake.in acConfig.cmake @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/acConfig.cmake DESTINATION c_api/cmake)

endif()
