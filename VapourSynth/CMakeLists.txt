if(Build_VapourSynth_Plugin)
    project(Anime4KCPP_VapourSynth LANGUAGES CXX)

    aux_source_directory(src SOURCE)

    find_package(PkgConfig REQUIRED)
    pkg_check_modules(VAPOURSYNTH REQUIRED IMPORTED_TARGET vapoursynth)

    add_library(${PROJECT_NAME} SHARED ${SOURCE})
    target_link_libraries(${PROJECT_NAME} PRIVATE PkgConfig::VAPOURSYNTH)
    target_include_directories(${PROJECT_NAME} PUBLIC ${VAPOURSYNTH_INCLUDE_DIRS})

    if(Other_Optimization_For_Other)
        if(MSVC)
            if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
                target_compile_options(${PROJECT_NAME} PRIVATE /O3)
            endif()
        else()
            target_compile_options(${PROJECT_NAME} PRIVATE -Ofast)
            target_link_options(${PROJECT_NAME} PRIVATE -Ofast)
        endif()
    endif()

    include(${TOP_DIR}/cmake/ThirdPartyForVS.cmake)

    install(
        TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin
    )

endif()
