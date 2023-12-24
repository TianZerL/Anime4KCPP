#ifdef ENABLE_HIP

#include "HipInterface.hpp"
#include "ACHip.hpp"

Anime4KCPP::Hip::GPUList Anime4KCPP::Hip::listGPUs() noexcept
{
    try
    {
        return GPUList(cuGetDeviceCount(), cuGetHipInfo());
    }
    catch (const std::exception& e)
    {
        return GPUList(0, e.what());
    }
}

Anime4KCPP::Hip::GPUInfo Anime4KCPP::Hip::checkGPUSupport(const int dID) noexcept
{
    try
    {
        return GPUInfo(cuCheckDeviceSupport(dID), cuGetDeviceInfo(dID));
    }
    catch (const std::exception& e)
    {
        return GPUInfo(false, e.what());
    }
}

Anime4KCPP::Hip::GPUList::GPUList(const int devices, std::string message)
    : devices(devices), message(std::move(message)) {}

std::string& Anime4KCPP::Hip::GPUList::operator()() noexcept
{
    return message;
}

Anime4KCPP::Hip::GPUInfo::GPUInfo(const bool supported, std::string message) :
    supported(supported), message(std::move(message)) {};

std::string& Anime4KCPP::Hip::GPUInfo::operator()() noexcept
{
    return message;
}

Anime4KCPP::Hip::GPUInfo::operator bool() const noexcept
{
    return supported;
}

Anime4KCPP::Hip::Manager::Manager(const int dID) noexcept
    : dID(dID) {}

void Anime4KCPP::Hip::Manager::init()
{
    if (cuGetDeviceID() != dID)
        cuSetDeviceID(dID);
}

void Anime4KCPP::Hip::Manager::release() noexcept
{
    if (cuGetDeviceID() == dID)
        cuReleaseHip();
}

bool Anime4KCPP::Hip::Manager::isInitialized() noexcept
{
    return cuGetDeviceID() == dID;
}

bool Anime4KCPP::Hip::Manager::isSupport() noexcept
{
    return cuCheckDeviceSupport(dID);
}

#endif
