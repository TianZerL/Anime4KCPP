include(CheckCXXCompilerFlag)

set(CMAKE_REQUIRED_LIBRARIES OpenCL::OpenCL)
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
unset(CMAKE_REQUIRED_LIBRARIES)
