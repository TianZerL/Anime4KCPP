project(Anime4KCPP_CLI LANGUAGES CXX)

if(Build_CLI)

    aux_source_directory(src SOURCE)

    include_directories(${TOP_DIR}/Anime4KCore/include include)
    file(GLOB INCLUDE include/*.hpp)

    if(Use_Boost_filesystem)
        add_definitions(-DUSE_BOOST_FILESYSTEM) 
    endif()

    if(Enable_NCNN)
        add_definitions(-DENABLE_NCNN)
    endif()

    add_definitions(-DCOMPILER="${CMAKE_CXX_COMPILER_ID}")

    add_executable(${PROJECT_NAME} ${INCLUDE} ${SOURCE})

    include(${TOP_DIR}/cmake/ThirdPartyForCLI.cmake)

    install(
        TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
    )

endif()
