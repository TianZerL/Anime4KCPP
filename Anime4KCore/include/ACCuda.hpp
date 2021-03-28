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

        //return platforms, devices of each platform, all devices information
        GPUList listGPUs();
        //return result and information
        GPUInfo checkGPUSupport(const int dID);
    }

    namespace Processor
    {
        template<>
        struct GetManager<Cuda::ACNet> {
            using Manager = Cuda::Manager;
        };
        template<>
        struct GetManager<Cuda::Anime4K09> {
            using Manager = Cuda::Manager;
        };
    }
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

class Anime4KCPP::Cuda::Manager : public Anime4KCPP::Processor::Manager
{
public:
    Manager(const int dID = 0);
    virtual void init() override;
    virtual void release() override;
    virtual bool isInitialized() override;
    virtual bool isSupport() override;
private:
    int dID;
};

inline Anime4KCPP::Cuda::Manager::Manager(const int dID)
    : dID(dID) {}

inline void Anime4KCPP::Cuda::Manager::init()
{
    if (cuGetDeviceID() != dID)
        cuSetDeviceID(dID);
}

inline void Anime4KCPP::Cuda::Manager::release()
{
    if (cuGetDeviceID() == dID)
        cuReleaseCuda();
}

inline bool Anime4KCPP::Cuda::Manager::isInitialized()
{
    return cuGetDeviceID() == dID;
}

inline bool Anime4KCPP::Cuda::Manager::isSupport()
{
    return cuCheckDeviceSupport(dID);
}

inline Anime4KCPP::Cuda::GPUList Anime4KCPP::Cuda::listGPUs()
{
    return GPUList(cuGetDeviceCount(), cuGetCudaInfo());
}

inline Anime4KCPP::Cuda::GPUInfo Anime4KCPP::Cuda::checkGPUSupport(const int dID)
{
    return GPUInfo(cuCheckDeviceSupport(dID), cuGetDeviceInfo(dID));
}

#endif
