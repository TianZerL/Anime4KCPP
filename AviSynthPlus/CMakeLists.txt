project(Anime4KCPP_AviSynthPlus LANGUAGES CXX)

if(Build_AviSynthPlus_Plugin)

    aux_source_directory(src SOURCE)

    add_library(${PROJECT_NAME} SHARED ${SOURCE})

    include(${TOP_DIR}/cmake/ThirdPartyForASP.cmake)

    install(
        TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin
    )

endif()
