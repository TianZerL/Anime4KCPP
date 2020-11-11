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

if(Use_OpenCV_with_MSVC_for_Clang)
    set(TMP_FALG ${MSVC})
    set(MSVC True)
elseif(Use_OpenCV_with_MINGW_for_Clang)
    set(TMP_FALG ${MINGW})
    set(MINGW True)
endif()

find_package(OpenCV REQUIRED)

if(Use_OpenCV_with_MSVC_for_Clang)
    set(MSVC ${TMP_FALG})
elseif(Use_OpenCV_with_MINGW_for_Clang)
    set(MINGW ${TMP_FALG})
endif()

find_package(OpenCL REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${OpenCL_LIBRARIES})
