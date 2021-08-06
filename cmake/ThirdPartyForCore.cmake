set(OPENCL_HPP_URL https://github.com/KhronosGroup/OpenCL-CLHPP/raw/v2.0.15/include/CL/opencl.hpp)
set(SHA1_OPENCL_HPP "de739352c21ea9bf9b082bb903caec7de9212f97")

if(EXISTS ${TOP_DIR}/ThirdParty/include/opencl/CL/opencl.hpp)
    file(SHA1 ${TOP_DIR}/ThirdParty/include/opencl/CL/opencl.hpp LOCAL_SHA1_OPENCL_HPP)

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

include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME} BASE_NAME "AC")

target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

if(Use_Eigen3)
    target_include_directories(${PROJECT_NAME} PRIVATE $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>)
endif()

if(Use_TBB)
    target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${TBB_INCLUDE_PATH}>)
    find_library(TBB_LIBS 
    NAMES tbb 
    PATHS ${TBB_LIB_PATH} 
    REQUIRED)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${TBB_LIBS})
endif()

if(Enable_CUDA)
    if(NOT ${CMAKE_MINOR_VERSION} LESS 18)
        enable_language(CUDA)
    endif()
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA_Module)
endif()

if(Enable_NCNN)
    find_package(ncnn REQUIRED)
    target_link_libraries(${PROJECT_NAME} PUBLIC ncnn)
endif()

if(Enable_OpenCL)
    find_package(OpenCL REQUIRED)
    target_include_directories(${PROJECT_NAME} PRIVATE $<BUILD_INTERFACE:${TOP_DIR}/ThirdParty/include/opencl>)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)
endif()

if(Use_OpenCV_With_MSVC_For_Clang)
    set(TMP_FALG ${MSVC})
    set(MSVC True)
elseif(Use_OpenCV_With_MINGW_For_Clang)
    set(TMP_FALG ${MINGW})
    set(MINGW True)
endif()

find_package(OpenCV REQUIRED)

target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})

if(Use_OpenCV_With_MSVC_For_Clang)
    set(MSVC ${TMP_FALG})
elseif(Use_OpenCV_With_MINGW_For_Clang)
    set(MINGW ${TMP_FALG})
endif()
