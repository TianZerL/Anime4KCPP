project(ac_gui VERSION 3.0.0.0 LANGUAGES CXX)

set(GUI_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(GUI_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include(${DEPENDENCY_DIR}/qt.cmake)

set(AC_GUI_LANGUAGES
    zh_CN
)

foreach(AC_GUI_LANGUAGE ${AC_GUI_LANGUAGES})
    list(APPEND AC_GUI_QRC_LANGUAGE_LIST "<file alias=\"ac_gui_${AC_GUI_LANGUAGE}.qm\">${GUI_BINARY_DIR}/ac_gui_${AC_GUI_LANGUAGE}.qm</file>")
endforeach()
list(JOIN AC_GUI_QRC_LANGUAGE_LIST "\n" AC_GUI_QRC_LANGUAGES)

if(TARGET ac_video)
    set(AC_GUI_LICENSE "${TOP_DIR}/LICENSE-GPLv3")
else()
    set(AC_GUI_LICENSE "${TOP_DIR}/LICENSE-MIT")
endif()

file(WRITE "${GUI_BINARY_DIR}/resources/qresource.qrc.in" [[
<RCC>
    <qresource prefix="/icon">
        <file alias="logo">@TOP_DIR@/images/Logo.ico</file>
    </qresource>
    <qresource prefix="/i18n">
        @AC_GUI_QRC_LANGUAGES@
    </qresource>
    <qresource prefix="/license">
        <file alias="ac">@AC_GUI_LICENSE@</file>
    </qresource>
</RCC>
]])

configure_file(
    ${GUI_BINARY_DIR}/resources/qresource.qrc.in
    ${GUI_BINARY_DIR}/resources/qresource.qrc @ONLY
)

if (WIN32)
    file(WRITE "${GUI_BINARY_DIR}/resources/resource.rc" "IDI_ICON1               ICON    \"${TOP_DIR}/images/Logo.ico\"")
endif()

foreach(AC_GUI_LANGUAGE ${AC_GUI_LANGUAGES})
    list(APPEND AC_GUI_TS_FILES "${GUI_SOURCE_DIR}/languages/ac_gui_${AC_GUI_LANGUAGE}.ts")
endforeach()

file(GLOB AC_GUI_UI_FILES ${GUI_SOURCE_DIR}/ui/*.ui)
file(GLOB AC_GUI_RC_FILES ${GUI_BINARY_DIR}/resources/*.*rc)

set(AC_GUI_SRC_FILES
    ${GUI_SOURCE_DIR}/src/QtMain.cpp
    ${GUI_SOURCE_DIR}/include/MainWindow.hpp
    ${GUI_SOURCE_DIR}/src/MainWindow.cpp
    ${GUI_SOURCE_DIR}/include/Config.hpp
    ${GUI_SOURCE_DIR}/src/Config.cpp
    ${GUI_SOURCE_DIR}/include/Upscaler.hpp
    ${GUI_SOURCE_DIR}/src/Upscaler.cpp
    ${GUI_SOURCE_DIR}/include/Logger.hpp
    ${GUI_SOURCE_DIR}/src/Logger.cpp
    ${AC_GUI_UI_FILES}
    ${AC_GUI_RC_FILES}
)

dep_qt_add_executable(ac_gui "${AC_GUI_SRC_FILES}" "${AC_GUI_TS_FILES}" "${GUI_SOURCE_DIR}/include")

target_include_directories(ac_gui PRIVATE
    $<BUILD_INTERFACE:${GUI_SOURCE_DIR}/include>
)

target_link_libraries(ac_gui PRIVATE
    ac
    ac_util
    $<$<TARGET_EXISTS:ac_video>:ac_video>
    dep::qt
)

target_compile_definitions(ac_gui PRIVATE
    $<$<TARGET_EXISTS:ac_video>:AC_CLI_ENABLE_VIDEO>
)

set_target_properties(ac_gui PROPERTIES
    AUTOUIC_SEARCH_PATHS ${GUI_SOURCE_DIR}/ui
)

ac_check_enable_static_crt(ac_gui)

install(
    TARGETS ac_gui
    RUNTIME DESTINATION bin
)
