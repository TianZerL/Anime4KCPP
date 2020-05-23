find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${OpenCL_LIBRARIES})
