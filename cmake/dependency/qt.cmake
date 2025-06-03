if(NOT TARGET dep::qt)
    add_library(dep_qt INTERFACE IMPORTED)

    find_package(QT NAMES Qt6 Qt5 COMPONENTS Widgets LinguistTools REQUIRED)
    find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets LinguistTools REQUIRED)

    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)

    function(dep_qt_add_executable ARG_TARGET)
        cmake_parse_arguments(
            ARG
            ""
            ""
            "SRC_FILES;TS_FILES;INCLUDE_DIRECTORY"
            ${ARGN}
        )

        if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
            if(${QT_VERSION_MINOR} GREATER_EQUAL 2)
                qt6_add_executable(${ARG_TARGET} ${ARG_SRC_FILES})
                qt6_add_translations(${ARG_TARGET} TS_FILES ${ARG_TS_FILES} QM_FILES_OUTPUT_VARIABLE dep_qt_QM_FILES) # set QM_FILES_OUTPUT_VARIABLE to disable automatic resource embedding
            else()
                qt6_create_translation(dep_qt_QM_FILES ${ARG_SRC_FILES} ${ARG_TS_FILES} OPTIONS -I ${ARG_INCLUDE_DIRECTORY})
                qt6_add_executable(${ARG_TARGET} ${ARG_SRC_FILES} ${dep_qt_QM_FILES})
            endif()
        else()
            qt5_create_translation(dep_qt_QM_FILES ${ARG_SRC_FILES} ${ARG_TS_FILES} OPTIONS -I ${ARG_INCLUDE_DIRECTORY})
            add_executable(${ARG_TARGET} ${ARG_SRC_FILES} ${dep_qt_QM_FILES})
        endif()

        set_target_properties(${ARG_TARGET} PROPERTIES
            MACOSX_BUNDLE_GUI_IDENTIFIER tianzerl.anime4kcpp.gui
            MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
            MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
            MACOSX_BUNDLE TRUE
            WIN32_EXECUTABLE TRUE
        )
    endfunction()

    target_link_libraries(dep_qt INTERFACE Qt${QT_VERSION_MAJOR}::Widgets)
    add_library(dep::qt ALIAS dep_qt)
endif()
