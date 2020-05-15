project(Anime4KCPP_GUI LANGUAGES CXX)

if(Build_GUI)

    set(CMAKE_INCLUDE_CURRENT_DIR ON)

    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)

    set(TS_FILES ./src/Anime4KCPP_GUI_zh_CN.ts)
    set(UI_FILE ./src/mainwindow.ui)

    include_directories(include ../Anime4KCore/include)
    file(GLOB INCLUDE include/*.h)

    aux_source_directory(src SOURCE)

    add_executable(Anime4KCPP_GUI
        WIN32
        ${INCLUDE}
        ${SOURCE}
        ${TS_FILES}
        ${UI_FILE}
    )

    include(../cmake/ThirdPartyForGUI.cmake)

    qt5_create_translation(QM_FILES ${CMAKE_SOURCE_DIR} ${TS_FILES})

endif()
