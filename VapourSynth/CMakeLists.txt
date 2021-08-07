project(Anime4KCPP_VapourSynth LANGUAGES CXX)

if(Build_VapourSynth_Plugin)

    aux_source_directory(src SOURCE)

    add_library(${PROJECT_NAME} SHARED ${SOURCE})

    include(${TOP_DIR}/cmake/ThirdPartyForVS.cmake)

    install(
        TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin
    )

endif()
