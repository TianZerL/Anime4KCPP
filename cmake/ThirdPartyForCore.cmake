set(OPENCL_HPP_URL https://github.com/KhronosGroup/OpenCL-CLHPP/raw/master/include/CL/opencl.hpp)
set(SHA1_OPENCL_HPP "852d7cad4d5f479104c6426e6a837bf547ad033f")

if(EXISTS ${TOP_DIR}/ThirdParty/include/opencl/CL/opencl.hpp)
    file(SHA1 ${TOP_DIR}/ThirdParty/include/opencl/CL/opencl.hpp LOCAL_OPENCL_HPP)

    if(NOT ${LOCAL_OPENCL_HPP} STREQUAL ${SHA1_OPENCL_HPP})
        message("Warning:")
        message("   Local SHA1 for opencl.hpp:   ${LOCAL_SHA1_OPENCL_HPP}")
        message("   Expected SHA1:              ${SHA1_OPENCL_HPP}")
        message("   Mismatch SHA1 for opencl.hpp, trying to download it...")

        file(
            DOWNLOAD ${OPENCL_HPP_URL} ${TOP_DIR}/ThirdParty/include/opencl/CL/opencl.hpp 
            SHOW_PROGRESS 
            EXPECTED_HASH SHA1=${SHA1_OPENCL_HPP}
        )

    endif()
else()
    file(
        DOWNLOAD ${OPENCL_HPP_URL} ${TOP_DIR}/ThirdParty/include/opencl/CL/opencl.hpp 
        SHOW_PROGRESS 
        EXPECTED_HASH SHA1=${SHA1_OPENCL_HPP}
    )
endif()

if(Use_Eigen3)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

if(Use_TBB)
    include_directories(${TBB_INCLUDE_PATH})
    find_library(TBB_LIBS 
    NAMES tbb 
    PATHS ${TBB_LIB_PATH} 
    REQUIRED)
    target_link_libraries(${PROJECT_NAME} ${TBB_LIBS})
endif()

if(Enable_CUDA)
    if(NOT ${CMAKE_MINOR_VERSION} LESS 18)
        enable_language(CUDA)
    endif()
    target_link_libraries(${PROJECT_NAME} CUDA_Module)
endif()

if(Enable_NCNN)
    find_package(ncnn REQUIRED)
    target_link_libraries(${PROJECT_NAME} ncnn)
endif()

if(Use_OpenCV_with_MSVC_for_Clang)
    set(TMP_FALG ${MSVC})
    set(MSVC True)
elseif(Use_OpenCV_with_MINGW_for_Clang)
    set(TMP_FALG ${MINGW})
    set(MINGW True)
endif()

if(Enable_OpenCL)
    find_package(OpenCL REQUIRED)
    include_directories(${OpenCL_INCLUDE_DIRS} ${TOP_DIR}/ThirdParty/include/opencl)
    target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES})
endif()

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

if(Use_OpenCV_with_MSVC_for_Clang)
    set(MSVC ${TMP_FALG})
elseif(Use_OpenCV_with_MINGW_for_Clang)
    set(MINGW ${TMP_FALG})
endif()
