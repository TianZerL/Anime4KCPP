#include"CudaHelper.cuh"
#include"CudaInterface.hpp"

void cuInitCuda(const unsigned int id)
{
    cudaError_t err = cudaSetDevice(id);
    CheckCudaErr(err);
}

void cuReleaseCuda()
{
    cudaError_t err = cudaDeviceReset();
    CheckCudaErr(err);
}

inline int cuGetDeviceCount()
{
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    CheckCudaErr(err);
    return deviceCount;
}

inline std::string cuGetDeviceInfo(const unsigned int id)
{
    cudaDeviceProp deviceProp;
    cudaError_t err = cudaGetDeviceProperties(&deviceProp, id);
    CheckCudaErr(err);

    return
        "Device id: " + std::to_string(id) +
        "\n Type: " + std::string(deviceProp.name) +
        "\n Video Memory: " + std::to_string(deviceProp.totalGlobalMem >> 20) + " mb" +
        "\n Compute Capability: " + std::to_string(deviceProp.major) + "." + std::to_string(deviceProp.minor);
}

std::string cuGetCudaInfo()
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

bool cuCheckDeviceSupport(const unsigned int id)
{
    cudaDeviceProp deviceProp;
    cudaError_t err = cudaGetDeviceProperties(&deviceProp, id);
    if (err != cudaSuccess || deviceProp.major < 2)
        return false;
    return true;
}
