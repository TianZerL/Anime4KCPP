#pragma once

#include"Anime4K.h"
#include"filterprocessor.h"

#include<fstream>

#ifdef __APPLE__
#include<OpenCL/opencl.h>
#else
#include<CL/cl.h>
#endif // SPECIAL OS

namespace Anime4KCPP
{
    struct DLL GPUList;
    struct DLL GPUInfo;
    class DLL Anime4KGPU;
}

struct Anime4KCPP::GPUList
{
    int platforms;
    std::vector<int> devices;
    std::string message;
    
    GPUList(const int platforms, const std::vector<int>& devices, const std::string& message);
    int operator[](int pID) const;
    std::string& operator()() noexcept;
};

struct Anime4KCPP::GPUInfo
{
    bool supported;
    std::string message;

    GPUInfo(const bool supported, const std::string& message);
    std::string& operator()() noexcept;
    operator bool() const noexcept;
};

class Anime4KCPP::Anime4KGPU :public Anime4K
{
public:
    Anime4KGPU(const Parameters& parameters = Parameters());
    virtual ~Anime4KGPU() = default;
    virtual void process() override;
    static void initGPU(unsigned int platformID = 0, unsigned int deviceID = 0);
    static void releaseGPU() noexcept;
    static bool isInitializedGPU();
    //return platforms, devices of each platform, all devices infomation
    static GPUList listGPUs();
    //return result and infomation
    static GPUInfo checkGPUSupport(unsigned int pID, unsigned int dID);
private:
    void runKernel(cv::InputArray orgImg, cv::OutputArray dstImg);
    static void initOpenCL();
    static void releaseOpenCL() noexcept;
    static std::string readKernel(const std::string &fileName);

    virtual ProcessorType getProcessorType() noexcept override;
private:
    static bool isInitialized;

    static cl_context context;
    static cl_command_queue commandQueue;
    static cl_program program;
    static cl_device_id device;

    static unsigned int pID;
    static unsigned int dID;

    static size_t workGroupSizeLog;

    double nWidth;
    double nHeight;

#ifdef BUILT_IN_KERNEL
    static const std::string Anime4KCPPKernelSourceString;
#endif // BUILT_IN_KERNEL

};
