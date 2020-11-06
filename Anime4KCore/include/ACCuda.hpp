#pragma once

#ifdef ENABLE_CUDA

#include"CudaAnime4K09.hpp"
#include"CudaACNet.hpp"
#include"ACManager.hpp"

namespace Anime4KCPP
{
    namespace Cuda
    {
        class Manager;

        struct GPUList;
        struct GPUInfo;

        //return platforms, devices of each platform, all devices infomation
        GPUList listGPUs();
        //return result and infomation
        GPUInfo checkGPUSupport(const unsigned int dID);
    }
}

class Anime4KCPP::Cuda::Manager : public Anime4KCPP::Processor::Manager
{
public:
    Manager(const unsigned int dID = 0);
    virtual void init() override;
    virtual void release() override;
    virtual bool isInitialized() override;
private:
    unsigned int dID;
    bool initialized = false;
};

inline Anime4KCPP::Cuda::Manager::Manager(const unsigned int dID)
    : dID(dID) {}

inline void Anime4KCPP::Cuda::Manager::init()
{
    cuInitCuda(dID);
    initialized = true;
}

inline void Anime4KCPP::Cuda::Manager::release()
{
    cuReleaseCuda();
    initialized = false;
}

inline bool Anime4KCPP::Cuda::Manager::isInitialized()
{
    return initialized;
}

struct Anime4KCPP::Cuda::GPUList
{
    int devices;
    std::string message;

    GPUList(const int devices, std::string message);
    std::string& operator()() noexcept;
};

inline Anime4KCPP::Cuda::GPUList::GPUList(const int devices, std::string message)
    : devices(devices), message(std::move(message)) {}

inline std::string& Anime4KCPP::Cuda::GPUList::operator()() noexcept
{
    return message;
}

struct Anime4KCPP::Cuda::GPUInfo
{
    bool supported;
    std::string message;

    GPUInfo(const bool supported, std::string message);
    std::string& operator()() noexcept;
    operator bool() const noexcept;
};

inline Anime4KCPP::Cuda::GPUInfo::GPUInfo(const bool supported, std::string message) :
    supported(supported), message(std::move(message)) {};

inline std::string& Anime4KCPP::Cuda::GPUInfo::operator()() noexcept
{
    return message;
}

inline Anime4KCPP::Cuda::GPUInfo::operator bool() const noexcept
{
    return supported;
}

inline Anime4KCPP::Cuda::GPUList Anime4KCPP::Cuda::listGPUs()
{
    return GPUList(cuGetDeviceCount(), cuGetCudaInfo());
}

inline Anime4KCPP::Cuda::GPUInfo Anime4KCPP::Cuda::checkGPUSupport(const unsigned int dID)
{
    return GPUInfo(cuCheckDeviceSupport(dID), cuGetDeviceInfo(dID));
}

#endif
