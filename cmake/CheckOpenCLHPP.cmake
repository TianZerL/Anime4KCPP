macro(check_opencl_hpp_support RESULT_VAR)
    include(CheckCXXSourceCompiles)

    set(CMAKE_REQUIRED_LIBRARIES ${ARGN})
    check_cxx_source_compiles("
    #define CL_HPP_TARGET_OPENCL_VERSION ${OpenCL_VERSION_MAJOR}${OpenCL_VERSION_MINOR}0
    #define CL_HPP_MINIMUM_OPENCL_VERSION 110

    #include <vector>

    #if __has_include(<CL/opencl.hpp>)
    #   include <CL/opencl.hpp>
    #elif __has_include(<CL/cl2.hpp>)
    #   include <CL/cl2.hpp>
    #else
    #   include <CL/cl.hpp>
    #endif

    int main() {
        cl::size_type a = 0;
        cl_int err = CL_SUCCESS;

        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

        cl::Context context(devices[0], nullptr, nullptr, nullptr, &err);
        cl::CommandQueue queue(context, devices[0], 0, err);
        return 0;
    }
    " ${RESULT_VAR})
    unset(CMAKE_REQUIRED_LIBRARIES)
endmacro()
