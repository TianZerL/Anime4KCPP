project(Anime4KCPP_AviSynthPlus LANGUAGES CXX)

if(Build_AviSynthPlus_plugin)

    aux_source_directory(src SOURCE)
    aux_source_directory(${TOP_DIR}/Anime4KCore/src Anime4KCPPCoreSource)

    include_directories(${TOP_DIR}/Anime4KCore/include)

    add_definitions(-DBUILT_IN_KERNEL)

    if(Use_Legacy_OpenCL_API)
        add_definitions(-DLEGACY_OPENCL_API)
    endif()

    add_library(${PROJECT_NAME} SHARED ${SOURCE} ${Anime4KCPPCoreSource})

    include(${TOP_DIR}/cmake/ThirdPartyForASP.cmake)

    install(
        TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION AviSynthPlusPlugin
        LIBRARY DESTINATION AviSynthPlusPlugin
    )

endif()
