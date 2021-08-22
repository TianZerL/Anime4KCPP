project(Anime4KCPPDS LANGUAGES CXX)

if(Build_DS_Filter)

    aux_source_directory(src SOURCE)
    
    file(GLOB INCLUDE include/*.h)

    enable_language(RC)
    list(APPEND SOURCE src/Anime4KCPPDS.rc src/Anime4KCPPDS.def)

    add_definitions(-DUNICODE)

    add_library(${PROJECT_NAME} SHARED ${INCLUDE} ${SOURCE})

    target_include_directories(${PROJECT_NAME} PRIVATE include)

    include(${TOP_DIR}/cmake/ThirdPartyForDS.cmake)

    install(
        TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
    )

endif()
