project(Anime4KCPP_CLI LANGUAGES CXX)

if(Build_CLI)

    aux_source_directory(src SOURCE)

    include_directories(include)
    
    file(GLOB INCLUDE include/*.hpp)

    add_executable(${PROJECT_NAME} ${INCLUDE} ${SOURCE})

    target_compile_definitions(
        ${PROJECT_NAME} 
        PRIVATE 
            $<$<BOOL:${Use_Boost_filesystem}>:USE_BOOST_FILESYSTEM>
    )

    include(${TOP_DIR}/cmake/ThirdPartyForCLI.cmake)

    install(
        TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
    )

endif()
