#include "CudaHelper.cuh"
#include "CudaInterface.hpp"

int currCudaDeviceID = 0;

void Anime4KCPP::Cuda::cuSetDeviceID(const int id) noexcept
{
    if (id < 0 || id >= cuGetDeviceCount())
        currCudaDeviceID = 0;
    else
        currCudaDeviceID = id;
}

int Anime4KCPP::Cuda::cuGetDeviceID() noexcept
{
    return currCudaDeviceID;
}

void Anime4KCPP::Cuda::cuReleaseCuda() noexcept
{
    cudaDeviceReset();
    currCudaDeviceID = 0;
}

int Anime4KCPP::Cuda::cuGetDeviceCount() noexcept
{
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
        return 0;
    return deviceCount;
}

std::string Anime4KCPP::Cuda::cuGetDeviceInfo(const int id)
{
    cudaDeviceProp deviceProp;
    cudaError_t err =
        (id < 0 || id >= cuGetDeviceCount()) ? cudaGetDeviceProperties(&deviceProp, 0) : cudaGetDeviceProperties(&deviceProp, id);
    if (err != cudaSuccess)
        return "Failed to find CUDA device: " + std::to_string(id);

    return "Device id: " + std::to_string(id) +
           "\n Type: " + std::string(deviceProp.name) +
           "\n Video Memory: " + std::to_string(deviceProp.totalGlobalMem >> 20) + " mb" +
           "\n Compute Capability: " + std::to_string(deviceProp.major) + "." + std::to_string(deviceProp.minor);
}

std::string Anime4KCPP::Cuda::cuGetCudaInfo()
{
    std::string info;
    int deviceCount = cuGetDeviceCount();
    if (!deviceCount)
        info = "No CUDA device found";
    else
        for (int i = 0; i < deviceCount; i++)
            info += cuGetDeviceInfo(i) + "\n";
    return info;
}

bool Anime4KCPP::Cuda::cuCheckDeviceSupport(const int id) noexcept
{
    cudaDeviceProp deviceProp;
    cudaError_t err =
        (id < 0 || id >= cuGetDeviceCount()) ? cudaGetDeviceProperties(&deviceProp, 0) : cudaGetDeviceProperties(&deviceProp, id);
    if (err != cudaSuccess || deviceProp.major < 2)
        return false;
    return true;
}
