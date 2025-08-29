include(CheckCXXCompilerFlag)

set(CMAKE_REQUIRED_LIBRARIES OpenCL::OpenCL)
set(CMAKE_REQUIRED_DEFINITIONS
    -DCL_HPP_TARGET_OPENCL_VERSION=${OpenCL_VERSION_MAJOR}${OpenCL_VERSION_MINOR}0
    -DCL_HPP_MINIMUM_OPENCL_VERSION=110
)
check_cxx_source_compiles("
#if __has_include(<CL/opencl.hpp>)
#   include <CL/opencl.hpp>
#elif __has_include(<CL/cl2.hpp>)
#   include <CL/cl2.hpp>
#else
#   include <CL/cl.hpp>
#endif

#include <vector>

int main() {
    cl::size_type a = 0;
    cl::Platform::get(&std::vector<cl::Platform>());
    cl::Context context(cl::Device::getDefault());
    cl::CommandQueue queue(context, cl::Device::getDefault());
    return 0;
}
" dep_opencl_SUPPORT_HPP)
unset(CMAKE_REQUIRED_DEFINITIONS)
unset(CMAKE_REQUIRED_LIBRARIES)
