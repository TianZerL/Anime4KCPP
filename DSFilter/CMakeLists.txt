project(Anime4KCPPDS LANGUAGES CXX)

if(Build_DS_Filter)

    aux_source_directory(src SOURCE)
    aux_source_directory(${TOP_DIR}/Anime4KCore/src Anime4KCPPCoreSource)

    include_directories(${TOP_DIR}/Anime4KCore/include include)
    file(GLOB INCLUDE include/*.h)
    enable_language(RC)
    list(APPEND SOURCE src/Anime4KCPPDS.rc src/Anime4KCPPDS.def)

    add_definitions(-DBUILT_IN_KERNEL -DUNICODE)

    add_library(${PROJECT_NAME} SHARED ${INCLUDE} ${SOURCE} ${Anime4KCPPCoreSource})

    include(${TOP_DIR}/cmake/ThirdPartyForDS.cmake)

    install(
        TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION DSFilter
    )

endif()
