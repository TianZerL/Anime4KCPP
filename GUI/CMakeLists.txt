if(Build_GUI)
    project(Anime4KCPP_GUI VERSION 1.13.0 LANGUAGES CXX)

    if(NOT Enable_Image_IO OR NOT Enable_Video)
        message (
            FATAL_ERROR "To compile the GUI, you cannot turn off image IO or Video support from the core\n"
        )
    endif()

    aux_source_directory(src SOURCE)

    file(GLOB INCLUDE include/*.h)

    find_package(Qt${Qt_Version} COMPONENTS Widgets LinguistTools REQUIRED)

    set(CMAKE_INCLUDE_CURRENT_DIR ON)
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)

    file(GLOB TS_FILES ./src/*.ts)
    set(UI_FILE ./src/mainwindow.ui)
    set(QRC_FILE ./src/logo.qrc)

    if (WIN32)
        if(NOT EXISTS src/logo.rc)
            file(WRITE src/logo.rc "IDI_ICON1               ICON    \"../../images/Logo.ico\"")
        endif()
        list(APPEND SOURCE src/logo.rc)
    endif()

    if(Qt_Version MATCHES "5")
        qt5_create_translation(QM_FILES include ${SOURCE} ${UI_FILE} ${TS_FILES})
    else()
        qt6_create_translation(QM_FILES include ${SOURCE} ${UI_FILE} ${TS_FILES})
    endif()

    add_executable(${PROJECT_NAME} WIN32 MACOSX_BUNDLE
        ${INCLUDE}
        ${SOURCE}
        ${QM_FILES}
        ${UI_FILE}
        ${QRC_FILE}
    )

    target_include_directories(${PROJECT_NAME} PRIVATE include)

    target_compile_definitions(
        ${PROJECT_NAME} 
        PRIVATE 
            ANIME4KCPP_GUI_VERSION="${PROJECT_VERSION}"
    )

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

    include(${TOP_DIR}/cmake/ThirdPartyForGUI.cmake)

    install(
        TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
    )
    install(
        FILES ${QM_FILES} DESTINATION  bin/language
    )

endif()
