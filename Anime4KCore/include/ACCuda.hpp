#pragma once

#ifdef ENABLE_CUDA

#include"CudaAnime4K09.hpp"
#include"CudaACNet.hpp"
#include"ACManager.hpp"

namespace Anime4KCPP
{
    namespace Cuda
    {
        class DLL Manager;

        struct DLL GPUList;
        struct DLL GPUInfo;

        //return platforms, devices of each platform, all devices information
        DLL GPUList listGPUs();
        //return result and information
        DLL GPUInfo checkGPUSupport(const int dID);
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

struct Anime4KCPP::Cuda::GPUList
{
    int devices;
    std::string message;

    GPUList(const int devices, std::string message);
    std::string& operator()() noexcept;
};

struct Anime4KCPP::Cuda::GPUInfo
{
    bool supported;
    std::string message;

    GPUInfo(const bool supported, std::string message);
    std::string& operator()() noexcept;
    operator bool() const noexcept;
};

#endif
