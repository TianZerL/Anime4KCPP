if(Build_DS_Filter)
    project(Anime4KCPPDS LANGUAGES CXX)

    aux_source_directory(src SOURCE)
    
    file(GLOB INCLUDE include/*.h)

    enable_language(RC)
    list(APPEND SOURCE src/Anime4KCPPDS.rc src/Anime4KCPPDS.def)

    add_library(${PROJECT_NAME} SHARED ${INCLUDE} ${SOURCE})

    target_compile_definitions(${PROJECT_NAME} PRIVATE UNICODE _UNICODE)

    target_include_directories(${PROJECT_NAME} PRIVATE include)

    if((NOT MSVC) AND Ofast_Optimization_For_Other)
        target_compile_options(${PROJECT_NAME} PRIVATE -Ofast)
        target_link_options(${PROJECT_NAME} PRIVATE -Ofast)
    endif()

    include(${TOP_DIR}/cmake/ThirdPartyForDS.cmake)

    install(
        TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
    )

endif()
