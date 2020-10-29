#pragma once

#include<vector>
#include<sstream>

#ifdef __APPLE__
#include<OpenCL/opencl.h>
#else
#include<CL/cl.h>
#endif // SPECIAL OS

#include"OpenCLAnime4K09.hpp"
#include"OpenCLACNet.hpp"
#include"ACManager.hpp"

namespace Anime4KCPP
{
    namespace OpenCL
    {
        template<typename T>
        class Manager;

        struct DLL GPUList;
        struct DLL GPUInfo;

        //return platforms, devices of each platform, all devices infomation
        DLL GPUList listGPUs();
        //return result and infomation
        DLL GPUInfo checkGPUSupport(unsigned int pID, unsigned int dID);
    }
}

template<typename T>
class Anime4KCPP::OpenCL::Manager : public Anime4KCPP::Processor::Manager
{
public:
    Manager(const unsigned int pID = 0, const unsigned int dID = 0);
    virtual void init() override;
    virtual void release() override;
private:
    unsigned int pID, dID;
};

template<typename T>
inline Anime4KCPP::OpenCL::Manager<T>::Manager(const unsigned int pID, const unsigned int dID)
    : pID(pID), dID(dID) {}

template<typename T>
inline void Anime4KCPP::OpenCL::Manager<T>::init()
{
    if (!T::isInitializedGPU())
        T::initGPU(pID, dID);
}

template<typename T>
inline void Anime4KCPP::OpenCL::Manager<T>::release()
{
    if (T::isInitializedGPU())
        T::releaseGPU();
}

template<>
class Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::ACNet> : public Anime4KCPP::Processor::Manager
{
public:
    Manager(const unsigned int pID = 0, const unsigned int dID = 0, const CNNType type = CNNType::Default);
    virtual void init() override;
    virtual void release() override;
private:
    unsigned int pID, dID;
    CNNType type;
};

inline Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::ACNet>::Manager(const unsigned int pID, const unsigned int dID, const CNNType type)
    : pID(pID), dID(dID), type(type) {}

inline void Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::ACNet>::init()
{
    if (!Anime4KCPP::OpenCL::ACNet::isInitializedGPU())
        Anime4KCPP::OpenCL::ACNet::initGPU(pID, dID, type);
}

inline void Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::ACNet>::release()
{
    if (Anime4KCPP::OpenCL::ACNet::isInitializedGPU())
        Anime4KCPP::OpenCL::ACNet::releaseGPU();
}

struct Anime4KCPP::OpenCL::GPUList
{
    int platforms;
    std::vector<int> devices;
    std::string message;

    GPUList(const int platforms, const std::vector<int>& devices, const std::string& message);
    int operator[](int pID) const;
    std::string& operator()() noexcept;
};

struct Anime4KCPP::OpenCL::GPUInfo
{
    bool supported;
    std::string message;

    GPUInfo(const bool supported, const std::string& message);
    std::string& operator()() noexcept;
    operator bool() const noexcept;
};
