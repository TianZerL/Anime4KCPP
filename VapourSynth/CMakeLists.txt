if(Build_VapourSynth_Plugin)
    project(Anime4KCPP_VapourSynth LANGUAGES CXX)

    aux_source_directory(src SOURCE)

    add_library(${PROJECT_NAME} SHARED ${SOURCE})

    if((NOT MSVC) AND Ofast_Optimization_For_Other)
        target_compile_options(${PROJECT_NAME} PRIVATE -Ofast)
        target_link_options(${PROJECT_NAME} PRIVATE -Ofast)
    endif()

    include(${TOP_DIR}/cmake/ThirdPartyForVS.cmake)

    install(
        TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin
    )

endif()
