project(Anime4KCPP_CLI LANGUAGES CXX)

if(Build_CLI)

    aux_source_directory(src SOURCE)

    include_directories(include)
    
    file(GLOB INCLUDE include/*.hpp)

    add_definitions(-DENABLE_VIDEO)
    add_definitions(-DENABLE_IMAGE_IO)
    add_definitions(-DCOMPILER="${CMAKE_CXX_COMPILER_ID}")
    
    if(Use_Boost_filesystem)
        add_definitions(-DUSE_BOOST_FILESYSTEM) 
    endif()

    if(Enable_NCNN)
        add_definitions(-DENABLE_NCNN)
    endif()
    
    if(Enable_Preview_GUI)
        add_definitions(-DENABLE_PREVIEW_GUI)
    endif()

    add_executable(${PROJECT_NAME} ${INCLUDE} ${SOURCE})

    include(${TOP_DIR}/cmake/ThirdPartyForCLI.cmake)

    install(
        TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
    )

endif()
