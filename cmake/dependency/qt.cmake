if(NOT TARGET dep::qt)
    add_library(dep_qt INTERFACE IMPORTED)

    find_package(QT NAMES Qt6 Qt5 COMPONENTS Widgets LinguistTools REQUIRED)
    find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets LinguistTools REQUIRED)

    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)

    macro(dep_qt_add_executable PARAM_TARGET PARAM_SRC_FILES PARAM_TS_FILES PARAM_INCLUDE_DIRECTORY)
        if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
            qt_create_translation(dep_qt_QM_FILES ${PARAM_SRC_FILES} ${PARAM_TS_FILES} OPTIONS -I ${PARAM_INCLUDE_DIRECTORY})
            qt_add_executable(${PARAM_TARGET} ${PARAM_SRC_FILES} ${dep_qt_QM_FILES})
        else()
            qt5_create_translation(dep_qt_QM_FILES ${PARAM_SRC_FILES} ${PARAM_TS_FILES} OPTIONS -I ${PARAM_INCLUDE_DIRECTORY})
            add_executable(${PARAM_TARGET} ${PARAM_SRC_FILES} ${dep_qt_QM_FILES})
        endif()

        set_target_properties(${PARAM_TARGET} PROPERTIES
            MACOSX_BUNDLE_GUI_IDENTIFIER tianzerl.anime4kcpp.gui
            MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
            MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
            MACOSX_BUNDLE TRUE
            WIN32_EXECUTABLE TRUE
        )
    endmacro()

    target_link_libraries(dep_qt INTERFACE Qt${QT_VERSION_MAJOR}::Widgets)
    add_library(dep::qt ALIAS dep_qt)
endif()
