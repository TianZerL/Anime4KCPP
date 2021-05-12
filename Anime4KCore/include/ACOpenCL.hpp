#pragma once

#ifdef ENABLE_OPENCL

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

        //return platforms, devices of each platform, all devices information
        DLL GPUList listGPUs();
        //return result and information
        DLL GPUInfo checkGPUSupport(const int pID, const int dID);
    }

    namespace Processor
    {
        template<>
        struct GetManager<OpenCL::ACNet> {
            using Manager = OpenCL::Manager<OpenCL::ACNet>;
        };
        template<>
        struct GetManager<OpenCL::Anime4K09> {
            using Manager = OpenCL::Manager<OpenCL::Anime4K09>;
        };
    }
}

struct Anime4KCPP::OpenCL::GPUList
{
    int platforms;
    std::vector<int> devices;
    std::string message;

    GPUList(const int platforms, std::vector<int> devices, std::string message);
    int operator[](int pID) const;
    std::string& operator()() noexcept;
};

struct Anime4KCPP::OpenCL::GPUInfo
{
    bool supported;
    std::string message;

    GPUInfo(const bool supported, std::string message);
    std::string& operator()() noexcept;
    operator bool() const noexcept;
};

template<typename T>
class Anime4KCPP::OpenCL::Manager : public Anime4KCPP::Processor::Manager
{
public:
    Manager(const int pID = 0, const int dID = 0, const int OpenCLQueueNum = 4, const bool OpenCLParallelIO = false);
    virtual void init() override;
    virtual void release() override;
    virtual bool isInitialized() override;
    virtual bool isSupport() override;
private:
    int pID, dID;
    int OpenCLQueueNum;
    bool OpenCLParallelIO;
};

template<typename T>
inline Anime4KCPP::OpenCL::Manager<T>::Manager(const int pID, const int dID, const int OpenCLQueueNum, const bool OpenCLParallelIO)
    : pID(pID), dID(dID), OpenCLQueueNum(OpenCLQueueNum), OpenCLParallelIO(OpenCLParallelIO) {}

template<typename T>
inline void Anime4KCPP::OpenCL::Manager<T>::init()
{
    if (!T::isInitializedGPU())
        T::initGPU(pID, dID, OpenCLQueueNum, OpenCLParallelIO);
}

template<typename T>
inline void Anime4KCPP::OpenCL::Manager<T>::release()
{
    if (T::isInitializedGPU())
        T::releaseGPU();
}

template<typename T>
inline bool Anime4KCPP::OpenCL::Manager<T>::isInitialized()
{
    return T::isInitializedGPU();
}

template<typename T>
inline bool Anime4KCPP::OpenCL::Manager<T>::isSupport()
{
    return Anime4KCPP::OpenCL::checkGPUSupport(pID, dID);
}

template<>
class Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::ACNet> : public Anime4KCPP::Processor::Manager
{
public:
    Manager(const int pID = 0, const int dID = 0, const CNNType type = CNNType::Default, const int OpenCLQueueNum = 4, const bool OpenCLParallelIO = false);
    virtual void init() override;
    virtual void release() override;
    virtual bool isInitialized() override;
    virtual bool isSupport() override;
private:
    int pID, dID;
    int OpenCLQueueNum;
    bool OpenCLParallelIO;
    CNNType type;
};

inline Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::ACNet>::Manager(const int pID, const int dID, const CNNType type, const int OpenCLQueueNum, const bool OpenCLParallelIO)
    : pID(pID), dID(dID), OpenCLQueueNum(OpenCLQueueNum), OpenCLParallelIO(OpenCLParallelIO), type(type) {}

inline void Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::ACNet>::init()
{
    if (!Anime4KCPP::OpenCL::ACNet::isInitializedGPU())
        Anime4KCPP::OpenCL::ACNet::initGPU(pID, dID, type, OpenCLQueueNum, OpenCLParallelIO);
}

inline void Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::ACNet>::release()
{
    if (Anime4KCPP::OpenCL::ACNet::isInitializedGPU())
        Anime4KCPP::OpenCL::ACNet::releaseGPU();
}

inline bool Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::ACNet>::isInitialized()
{
    return Anime4KCPP::OpenCL::ACNet::isInitializedGPU();
}

inline bool Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::ACNet>::isSupport()
{
    return Anime4KCPP::OpenCL::checkGPUSupport(pID, dID);
}

#endif
