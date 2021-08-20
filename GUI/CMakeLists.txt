project(Anime4KCPP_GUI LANGUAGES CXX)

if(Build_GUI)

    aux_source_directory(src SOURCE)

    include_directories(include)

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
        qt5_create_translation(QM_FILES ${CMAKE_SOURCE_DIR} ${TS_FILES})
    else()
        qt6_create_translation(QM_FILES ${CMAKE_SOURCE_DIR} ${TS_FILES})
    endif()

    add_executable(${PROJECT_NAME} WIN32 MACOSX_BUNDLE
        ${INCLUDE}
        ${SOURCE}
        ${QM_FILES}
        ${UI_FILE}
        ${QRC_FILE}
    )

    include(${TOP_DIR}/cmake/ThirdPartyForGUI.cmake)

    install(
        TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
    )
    install(
        FILES ${QM_FILES} DESTINATION  bin/language
    )

endif()
