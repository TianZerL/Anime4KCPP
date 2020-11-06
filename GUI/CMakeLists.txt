project(Anime4KCPP_GUI LANGUAGES CXX)

if(Build_GUI)
    find_package(Qt5 COMPONENTS Widgets LinguistTools REQUIRED)
    
    set(CMAKE_INCLUDE_CURRENT_DIR ON)

    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)

    set(TS_FILES 
    ./src/Anime4KCPP_GUI_zh_CN.ts 
    ./src/Anime4KCPP_GUI_zh_TW.ts 
    ./src/Anime4KCPP_GUI_ja_JP.ts 
    ./src/Anime4KCPP_GUI_fr_FR.ts)
    set(UI_FILE ./src/mainwindow.ui)
    set(QRC_FILE ./src/logo.qrc)

    include_directories(include ${TOP_DIR}/Anime4KCore/include)
    file(GLOB INCLUDE include/*.h)

    aux_source_directory(src SOURCE)

    if (WIN32)
        if(NOT EXISTS src/logo.rc)
            file(WRITE src/logo.rc "IDI_ICON1               ICON    \"../../images/Logo.ico\"")
        endif()
        list(APPEND SOURCE src/logo.rc)
    endif()

    qt5_create_translation(QM_FILES ${CMAKE_SOURCE_DIR} ${TS_FILES})

    add_executable(Anime4KCPP_GUI
        WIN32
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
