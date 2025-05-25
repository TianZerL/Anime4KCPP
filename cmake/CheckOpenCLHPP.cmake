include(CheckCXXCompilerFlag)

set(CMAKE_REQUIRED_LIBRARIES OpenCL::OpenCL)
if (dep_opencl_TARGET_VERSION)
    set(CHECK_OPENCL_HPP_TARGET_VERSION 300)
else()
    set(CHECK_OPENCL_HPP_TARGET_VERSION ${dep_opencl_TARGET_VERSION})
endif()
set(CMAKE_REQUIRED_DEFINITIONS -DCL_HPP_TARGET_OPENCL_VERSION=${CHECK_OPENCL_HPP_TARGET_VERSION} -DCL_HPP_MINIMUM_OPENCL_VERSION=110)
check_cxx_source_compiles("
#if __has_include(<CL/opencl.hpp>)
#   include <CL/opencl.hpp>
#elif __has_include(<CL/cl2.hpp>)
#   include <CL/cl2.hpp>
#else
#   include <CL/cl.hpp>
#endif
int main() { cl::size_type a = 0; return 0; }
" dep_opencl_SUPPORT_HPP)
unset(CMAKE_REQUIRED_DEFINITIONS)
unset(CHECK_OPENCL_HPP_TARGET_VERSION)
unset(CMAKE_REQUIRED_LIBRARIES)
