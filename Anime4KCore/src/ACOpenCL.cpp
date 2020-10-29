#define DLL

#include "ACOpenCL.hpp"

Anime4KCPP::OpenCL::GPUList::GPUList(
    const int platforms,
    const std::vector<int>& devices,
    const std::string& message
) :platforms(platforms), devices(devices), message(message) {}

int Anime4KCPP::OpenCL::GPUList::operator[](int pID) const
{
    return devices[pID];
}

std::string& Anime4KCPP::OpenCL::GPUList::operator()() noexcept
{
    return message;
}

Anime4KCPP::OpenCL::GPUInfo::GPUInfo(const bool supported, const std::string& message) :
    supported(supported), message(message) {};

std::string& Anime4KCPP::OpenCL::GPUInfo::operator()() noexcept
{
    return message;
}

Anime4KCPP::OpenCL::GPUInfo::operator bool() const noexcept
{
    return supported;
}

Anime4KCPP::OpenCL::GPUList Anime4KCPP::OpenCL::listGPUs()
{
    cl_int err = 0;
    cl_uint platforms = 0;
    cl_uint devices = 0;
    cl_platform_id* platform = nullptr;
    cl_device_id* device = nullptr;

    size_t platformNameLength = 0;
    size_t DeviceNameLength = 0;
    char* platformName = nullptr;
    char* DeviceName = nullptr;

    std::ostringstream msg;

    std::vector<int> devicesVector;

    err = clGetPlatformIDs(0, nullptr, &platforms);
    if (err != CL_SUCCESS || !platforms)
        return GPUList(0, { 0 }, "No suppoted platform");

    platform = new cl_platform_id[platforms];
    err = clGetPlatformIDs(platforms, platform, nullptr);
    if (err != CL_SUCCESS)
    {
        delete[] platform;
        return GPUList(0, { 0 }, "inital platform error");
    }

    for (cl_uint i = 0; i < platforms; i++)
    {
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, 0, nullptr, &platformNameLength);
        if (err != CL_SUCCESS)
        {
            delete[] platform;
            return GPUList(0, { 0 }, "Failed to get platform name length infomation");
        }


        platformName = new char[platformNameLength];
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, platformNameLength, platformName, nullptr);
        if (err != CL_SUCCESS)
        {
            delete[] platformName;
            delete[] platform;
            return GPUList(0, { 0 }, "Failed to get platform name infomation");
        }
        msg << "Platform " << i << ": " << platformName << std::endl;

        delete[] platformName;

        err = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &devices);
        if (err != CL_SUCCESS || !devices)
        {
            delete[] platform;
            return GPUList(0, { 0 }, "No supported GPU");
        }

        devicesVector.push_back(devices);

        device = new cl_device_id[devices];
        err = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_GPU, devices, device, nullptr);
        if (err != CL_SUCCESS)
        {
            delete[] device;
            delete[] platform;
            return GPUList(0, { 0 }, "inital GPU error");
        }

        for (cl_uint j = 0; j < devices; j++)
        {
            err = clGetDeviceInfo(device[j], CL_DEVICE_NAME, 0, nullptr, &DeviceNameLength);
            if (err != CL_SUCCESS)
            {
                clReleaseDevice(device[j]);
                delete[] device;
                delete[] platform;
                return GPUList(0, { 0 }, "Failed to get device name length infomation");
            }


            DeviceName = new char[DeviceNameLength];
            err = clGetDeviceInfo(device[j], CL_DEVICE_NAME, DeviceNameLength, DeviceName, nullptr);
            if (err != CL_SUCCESS)
            {
                clReleaseDevice(device[j]);
                delete[] DeviceName;
                delete[] device;
                delete[] platform;
                return GPUList(0, { 0 }, "Failed to get device name infomation");
            }
            msg << "Device " << j << ": " << DeviceName << std::endl;

            delete[] DeviceName;
            clReleaseDevice(device[j]);
        }
        delete[] device;
    }
    delete[] platform;

    return GPUList(platforms, devicesVector, msg.str());
}

Anime4KCPP::OpenCL::GPUInfo Anime4KCPP::OpenCL::checkGPUSupport(unsigned int pID, unsigned int dID)
{
    cl_int err = 0;
    cl_uint platforms = 0;
    cl_uint devices = 0;
    cl_platform_id firstPlatform = nullptr;
    cl_device_id device = nullptr;

    size_t platformNameLength = 0;
    size_t DeviceNameLength = 0;
    char* platformName = nullptr;
    char* DeviceName = nullptr;

    err = clGetPlatformIDs(0, nullptr, &platforms);
    if (err != CL_SUCCESS || !platforms)
        return GPUInfo(false, "No suppoted platform");

    cl_platform_id* tmpPlatform = new cl_platform_id[platforms];
    err = clGetPlatformIDs(platforms, tmpPlatform, nullptr);
    if (err != CL_SUCCESS)
    {
        delete[] tmpPlatform;
        return GPUInfo(false, "inital platform error");
    }


    if (pID >= 0 && pID < platforms)
        firstPlatform = tmpPlatform[pID];
    else
        firstPlatform = tmpPlatform[0];

    delete[] tmpPlatform;

    err = clGetPlatformInfo(firstPlatform, CL_PLATFORM_NAME, 0, nullptr, &platformNameLength);
    if (err != CL_SUCCESS)
        return GPUInfo(false, "Failed to get platform name length infomation");

    platformName = new char[platformNameLength];
    err = clGetPlatformInfo(firstPlatform, CL_PLATFORM_NAME, platformNameLength, platformName, nullptr);
    if (err != CL_SUCCESS)
    {
        delete[] platformName;
        return GPUInfo(false, "Failed to get platform name infomation");
    }


    err = clGetDeviceIDs(firstPlatform, CL_DEVICE_TYPE_GPU, 0, nullptr, &devices);
    if (err != CL_SUCCESS || !devices)
    {
        delete[] platformName;
        return GPUInfo(false, "No supported GPU");
    }

    cl_device_id* tmpDevice = new cl_device_id[devices];
    err = clGetDeviceIDs(firstPlatform, CL_DEVICE_TYPE_GPU, devices, tmpDevice, nullptr);
    if (err != CL_SUCCESS)
    {
        delete[] platformName;
        delete[] tmpDevice;
        return GPUInfo(false, "No supported GPU");
    }

    if (dID >= 0 && dID < devices)
        device = tmpDevice[dID];
    else
        device = tmpDevice[0];

    delete[] tmpDevice;

    err = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &DeviceNameLength);
    if (err != CL_SUCCESS)
    {
        delete[] platformName;
        clReleaseDevice(device);
        return GPUInfo(false, "Failed to get device name length infomation");
    }


    DeviceName = new char[DeviceNameLength];
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, DeviceNameLength, DeviceName, nullptr);
    if (err != CL_SUCCESS)
    {
        delete[] DeviceName;
        delete[] platformName;
        clReleaseDevice(device);
        return GPUInfo(false, "Failed to get device name infomation");
    }

    GPUInfo ret(true,
        std::string("Platform: ") + platformName + "\n" + "Device: " + DeviceName);

    delete[] DeviceName;
    delete[] platformName;
    clReleaseDevice(device);

    return ret;
}
