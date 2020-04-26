project(Anime4KCPP_GUI LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(TS_FILES ./src/Anime4KCPP_GUI_zh_CN.ts)
set(UI_FILE ./src/mainwindow.ui)

include_directories(include ../ThirdParty/include ../Anime4KCore/include)
file(GLOB INCLUDE include/*.h)

aux_source_directory(src SOURCE)

add_executable(Anime4KCPP_GUI
    WIN32
    ${INCLUDE}
    ${SOURCE}
    ${TS_FILES}
    ${UI_FILE}
)

find_package(Qt5 COMPONENTS Widgets LinguistTools REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS})

target_link_libraries(Anime4KCPP_GUI PRIVATE Qt5::Widgets)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE Anime4KCore)

qt5_create_translation(QM_FILES ${CMAKE_SOURCE_DIR} ${TS_FILES})
