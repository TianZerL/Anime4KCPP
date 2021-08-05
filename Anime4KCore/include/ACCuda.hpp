#pragma once

#ifdef ENABLE_CUDA

#include"CudaAnime4K09.hpp"
#include"CudaACNet.hpp"
#include"ACManager.hpp"

namespace Anime4KCPP
{
    namespace Cuda
    {
        class AC_EXPORT Manager;

        struct AC_EXPORT GPUList;
        struct AC_EXPORT GPUInfo;

        //return platforms, devices of each platform, all devices information
        AC_EXPORT GPUList listGPUs();
        //return result and information
        AC_EXPORT GPUInfo checkGPUSupport(const int dID);
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
    void init() override;
    void release() override;
    bool isInitialized() override;
    bool isSupport() override;
private:
    int dID;
};

struct Anime4KCPP::Cuda::GPUList
{
    int devices;
    std::string message;

    GPUList(int devices, std::string message);
    std::string& operator()() noexcept;
};

struct Anime4KCPP::Cuda::GPUInfo
{
    bool supported;
    std::string message;

    GPUInfo(bool supported, std::string message);
    std::string& operator()() noexcept;
    operator bool() const noexcept;
};

#endif
