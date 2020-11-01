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

        //struct DLL GPUList;
        //struct DLL GPUInfo;

        ////return platforms, devices of each platform, all devices infomation
        //DLL GPUList listGPUs();
        ////return result and infomation
        //DLL GPUInfo checkGPUSupport(unsigned int pID, unsigned int dID);
    }
}

class Anime4KCPP::Cuda::Manager : public Anime4KCPP::Processor::Manager
{
public:
    Manager(const unsigned int dID = 0);
    virtual void init() override;
    virtual void release() override;
private:
    unsigned int dID;
};

inline Anime4KCPP::Cuda::Manager::Manager(const unsigned int dID)
    : dID(dID) {}

inline void Anime4KCPP::Cuda::Manager::init()
{
    initCuda(dID);
}

inline void Anime4KCPP::Cuda::Manager::release()
{
    releaseCuda();
}

#endif
