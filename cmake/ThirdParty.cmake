set(URL https://github.com/TianZerL/cmdline/raw/master/cmdline.h)
file(DOWNLOAD ${URL} ${TOP_DIR}/ThirdParty/include/cmdline.h)

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${OpenCL_LIBRARIES})
