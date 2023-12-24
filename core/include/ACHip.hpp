#ifndef ANIME4KCPP_CORE_AC_HIP_HPP
#define ANIME4KCPP_CORE_AC_HIP_HPP

#ifdef ENABLE_HIP

#include "HipAnime4K09.hpp"
#include "HipACNet.hpp"
#include "ACManager.hpp"

namespace Anime4KCPP
{
    namespace Hip
    {
        class AC_EXPORT Manager;

        struct AC_EXPORT GPUList;
        struct AC_EXPORT GPUInfo;

        //return platforms, devices of each platform, all devices information
        AC_EXPORT GPUList listGPUs() noexcept;
        //return result and information
        AC_EXPORT GPUInfo checkGPUSupport(int dID) noexcept;
    }

    namespace Processor
    {
        template<>
        struct GetManager<Hip::ACNet> {
            using Manager = Hip::Manager;
        };
        template<>
        struct GetManager<Hip::Anime4K09> {
            using Manager = Hip::Manager;
        };
    }
}

class Anime4KCPP::Hip::Manager : public Anime4KCPP::Processor::Manager
{
public:
    Manager(int dID = 0) noexcept;
    void init() override;
    void release() noexcept override;
    bool isInitialized() noexcept override;
    bool isSupport() noexcept override;
    const char* name() noexcept override { return "HIP Processor Manager"; };
private:
    int dID;
};

struct Anime4KCPP::Hip::GPUList
{
    int devices;
    std::string message;

    GPUList(int devices, std::string message);
    std::string& operator()() noexcept;
};

struct Anime4KCPP::Hip::GPUInfo
{
    bool supported;
    std::string message;

    GPUInfo(bool supported, std::string message);
    std::string& operator()() noexcept;
    operator bool() const noexcept;
};

#endif

#endif // !ANIME4KCPP_CORE_AC_HIP_HPP
