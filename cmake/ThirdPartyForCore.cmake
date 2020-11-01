if(Use_TBB)
    include_directories(${TBB_INCLUDE_PATH})
    find_library(TBB_LIBS 
    NAMES tbb 
    PATHS ${TBB_LIB_PATH} 
    REQUIRED)
    target_link_libraries(${PROJECT_NAME} ${TBB_LIBS})
endif()

if(Enable_CUDA)
    target_link_libraries (${PROJECT_NAME} CUDA_Module)
endif()

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${OpenCL_LIBRARIES})
