project(ac_core VERSION 3.0.0.0 LANGUAGES CXX)

set(CORE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CORE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# third-party
include(${DEPENDENCY_DIR}/ruapu.cmake)
include(${DEPENDENCY_DIR}/stb.cmake)
if(AC_CORE_WITH_OPENCL)
    include(${DEPENDENCY_DIR}/opencl.cmake)
endif()
if(AC_CORE_WITH_EIGEN3)
    include(${DEPENDENCY_DIR}/eigen3.cmake)
endif()
if(AC_CORE_WITH_CUDA)
    include(${DEPENDENCY_DIR}/cuda.cmake)
endif()

if(AC_SHARED_LIB)
    add_library(ac SHARED)
else()
    add_library(ac STATIC)
endif()

target_sources(ac PRIVATE
    ${CORE_SOURCE_DIR}/src/Image.cpp
    ${CORE_SOURCE_DIR}/src/ImageProcess.cpp
    ${CORE_SOURCE_DIR}/src/ImageIO.cpp
    ${CORE_SOURCE_DIR}/src/ACNet.cpp
    ${CORE_SOURCE_DIR}/src/Processor.cpp
    ${CORE_SOURCE_DIR}/src/cpu/Dispatch.cpp
    ${CORE_SOURCE_DIR}/src/cpu/CPUProcessor.cpp
    ${CORE_SOURCE_DIR}/src/cpu/Generic.cpp
    $<$<BOOL:${AC_CORE_WITH_EIGEN3}>:${CORE_SOURCE_DIR}/src/cpu/Eigen3.cpp>
    $<$<BOOL:${AC_CORE_WITH_SSE}>:${CORE_SOURCE_DIR}/src/cpu/x86/SSE.cpp>
    $<$<BOOL:${AC_CORE_WITH_AVX}>:${CORE_SOURCE_DIR}/src/cpu/x86/AVX.cpp>
    $<$<BOOL:${AC_CORE_WITH_NEON}>:${CORE_SOURCE_DIR}/src/cpu/arm/NEON.cpp>
    $<$<BOOL:${AC_CORE_WITH_OPENCL}>:${CORE_SOURCE_DIR}/src/opencl/OpenCLProcessor.cpp>
    $<$<BOOL:${AC_CORE_WITH_CUDA}>:${CORE_SOURCE_DIR}/src/cuda/CUDAProcessor.cpp>
    $<$<BOOL:${AC_CORE_WITH_CUDA}>:${CORE_SOURCE_DIR}/src/cuda/Kernel.cu>
)

target_include_directories(ac PUBLIC
    $<BUILD_INTERFACE:${CORE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CORE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:ac/include>
)

if(NOT MSVC)
    if(AC_CORE_WITH_SSE)
        set_source_files_properties(${CORE_SOURCE_DIR}/src/cpu/x86/SSE.cpp PROPERTIES COMPILE_OPTIONS "-msse3")
    endif()
    if(AC_CORE_WITH_AVX)
        set_source_files_properties(${CORE_SOURCE_DIR}/src/cpu/x86/AVX.cpp PROPERTIES COMPILE_OPTIONS "-mavx;$<$<BOOL:${AC_CORE_WITH_FMA}>:-mfma>")
    endif()
    if(AC_CORE_WITH_NEON)
        set_source_files_properties(${CORE_SOURCE_DIR}/src/cpu/arm/NEON.cpp PROPERTIES COMPILE_OPTIONS "$<IF:$<BOOL:${AC_COMPILER_32BIT}>,-mfpu=neon,-march=armv8-a>")
    endif()
endif()

# parallel lib
if(MSVC)
    target_compile_definitions(ac PRIVATE AC_CORE_PARALLEL_PPL)
else()
    find_package(OpenMP COMPONENTS CXX QUIET)
    if(OpenMP_CXX_FOUND)
        target_compile_definitions(ac PRIVATE AC_CORE_PARALLEL_OPENMP)
    endif()
endif()

target_link_libraries(ac PRIVATE
    ac_util
    dep::ruapu
    dep::stb
    $<$<BOOL:${AC_CORE_WITH_EIGEN3}>:dep::eigen3>
    $<$<BOOL:${AC_CORE_WITH_OPENCL}>:dep::opencl>
    $<$<BOOL:${AC_CORE_WITH_CUDA}>:dep::cuda>
    $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

target_compile_definitions(ac PUBLIC
    $<$<BOOL:${AC_CORE_WITH_EIGEN3}>:AC_CORE_WITH_EIGEN3>
    $<$<BOOL:${AC_CORE_WITH_SSE}>:AC_CORE_WITH_SSE>
    $<$<BOOL:${AC_CORE_WITH_AVX}>:AC_CORE_WITH_AVX>
    $<$<BOOL:${AC_CORE_WITH_FMA}>:AC_CORE_WITH_FMA>
    $<$<BOOL:${AC_CORE_WITH_NEON}>:AC_CORE_WITH_NEON>
    $<$<BOOL:${AC_CORE_WITH_OPENCL}>:AC_CORE_WITH_OPENCL>
    $<$<BOOL:${AC_CORE_WITH_CUDA}>:AC_CORE_WITH_CUDA>
    $<$<BOOL:${AC_CORE_ENABLE_IMAGE_IO}>:AC_CORE_ENABLE_IMAGE_IO>

    AC_CORE_VERSION_STR="${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
    AC_CORE_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
    AC_CORE_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
    AC_CORE_BUILD_DATE="${DATE_TODAY}"
    AC_CORE_BUILD_YEAR="${DATE_YEAR}"
)

ac_check_disable_flags(ac)

file(WRITE "${CORE_BINARY_DIR}/Kernel.cl.hpp.in" [[
#ifndef AC_CORE_OPENCL_KERNEL_HPP
#define AC_CORE_OPENCL_KERNEL_HPP
namespace ac::core::opencl
{
    constexpr const char* KernelString = R"(@AC_CORE_OPENCL_KERNEL_CL@)";
}
#endif
]])

file(READ "${CORE_SOURCE_DIR}/src/opencl/Kernel.cl" AC_CORE_OPENCL_KERNEL_CL)

configure_file(
    ${CORE_BINARY_DIR}/Kernel.cl.hpp.in
    ${CORE_BINARY_DIR}/include/AC/Core/OpenCL/Kernel.hpp
    @ONLY
)

set_target_properties(ac PROPERTIES EXPORT_NAME "Core")

include(GenerateExportHeader)
generate_export_header(ac
    BASE_NAME "AC"
    EXPORT_FILE_NAME ${CORE_BINARY_DIR}/include/ACExport.hpp
)

install(
    TARGETS ac EXPORT AC
    ARCHIVE DESTINATION ac/lib
    LIBRARY DESTINATION ac/lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CORE_SOURCE_DIR}/include ${CORE_BINARY_DIR}/include DESTINATION ac)
