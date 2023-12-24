#include "HipHelper.hip"
#include "HipInterface.hpp"
#include <hip/amd_detail/amd_surface_functions.h>

int currHipDeviceID = 0;

void Anime4KCPP::Hip::cuSetDeviceID(const int id) noexcept
{
    if (id < 0 || id >= cuGetDeviceCount())
        currHipDeviceID = 0;
    else
        currHipDeviceID = id;
}

int Anime4KCPP::Hip::cuGetDeviceID() noexcept
{
    return currHipDeviceID;
}

void Anime4KCPP::Hip::cuReleaseHip() noexcept
{
    hipDeviceReset();
    currHipDeviceID = 0;
}

int Anime4KCPP::Hip::cuGetDeviceCount() noexcept
{
    int deviceCount;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess)
        return 0;
    return deviceCount;
}

std::string Anime4KCPP::Hip::cuGetDeviceInfo(const int id)
{
    hipDeviceProp_t deviceProp;
    hipError_t err =
        (id < 0 || id >= cuGetDeviceCount()) ? hipGetDeviceProperties(&deviceProp, 0) : hipGetDeviceProperties(&deviceProp, id);
    if (err != hipSuccess)
        return "Failed to find HIP device: " + std::to_string(id);

    return "Device id: " + std::to_string(id) +
           "\n Type: " + std::string(deviceProp.name) +
           "\n Video Memory: " + std::to_string(deviceProp.totalGlobalMem >> 20) + " mb" +
           "\n Compute Capability: " + std::to_string(deviceProp.major) + "." + std::to_string(deviceProp.minor);
}

std::string Anime4KCPP::Hip::cuGetHipInfo()
{
    std::string info;
    int deviceCount = cuGetDeviceCount();
    if (!deviceCount)
        info = "No HIP device found";
    else
        for (int i = 0; i < deviceCount; i++)
            info += cuGetDeviceInfo(i) + "\n";
    return info;
}

bool Anime4KCPP::Hip::cuCheckDeviceSupport(const int id) noexcept
{
    hipDeviceProp_t deviceProp;
    hipError_t err =
        (id < 0 || id >= cuGetDeviceCount()) ? hipGetDeviceProperties(&deviceProp, 0) : hipGetDeviceProperties(&deviceProp, id);
    if (err != hipSuccess || deviceProp.major < 2)
        return false;
    return true;
}
