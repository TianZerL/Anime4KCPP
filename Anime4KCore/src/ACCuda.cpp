#ifdef ENABLE_CUDA

#define DLL

#include"CudaInterface.hpp"
#include"ACCuda.hpp"

Anime4KCPP::Cuda::GPUList Anime4KCPP::Cuda::listGPUs()
{
    return GPUList(cuGetDeviceCount(), cuGetCudaInfo());
}

Anime4KCPP::Cuda::GPUInfo Anime4KCPP::Cuda::checkGPUSupport(const int dID)
{
    return GPUInfo(cuCheckDeviceSupport(dID), cuGetDeviceInfo(dID));
}

Anime4KCPP::Cuda::GPUList::GPUList(const int devices, std::string message)
    : devices(devices), message(std::move(message)) {}

std::string& Anime4KCPP::Cuda::GPUList::operator()() noexcept
{
    return message;
}

Anime4KCPP::Cuda::GPUInfo::GPUInfo(const bool supported, std::string message) :
    supported(supported), message(std::move(message)) {};

std::string& Anime4KCPP::Cuda::GPUInfo::operator()() noexcept
{
    return message;
}

Anime4KCPP::Cuda::GPUInfo::operator bool() const noexcept
{
    return supported;
}

Anime4KCPP::Cuda::Manager::Manager(const int dID)
    : dID(dID) {}

void Anime4KCPP::Cuda::Manager::init()
{
    if (cuGetDeviceID() != dID)
        cuSetDeviceID(dID);
}

void Anime4KCPP::Cuda::Manager::release()
{
    if (cuGetDeviceID() == dID)
        cuReleaseCuda();
}

bool Anime4KCPP::Cuda::Manager::isInitialized()
{
    return cuGetDeviceID() == dID;
}

bool Anime4KCPP::Cuda::Manager::isSupport()
{
    return cuCheckDeviceSupport(dID);
}

#endif