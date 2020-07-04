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
    class DLL Anime4KGPU;
}

class Anime4KCPP::Anime4KGPU :public Anime4K
{
public:
    Anime4KGPU(const Parameters& parameters = Parameters());
    virtual ~Anime4KGPU() = default;
    virtual void process() override;
    static void initGPU(unsigned int platformID = 0, unsigned int deviceID = 0);
    static void releaseGPU();
    static bool isInitializedGPU();
    //return platforms, devices of each platform, all devices infomation
    static std::pair<std::pair<int, std::vector<int>>, std::string> listGPUs();
    //return result and infomation
    static std::pair<bool, std::string> checkGPUSupport(unsigned int pID, unsigned int dID);
private:
    void runKernel(cv::InputArray orgImg, cv::OutputArray dstImg);
    static void initOpenCL();
    static void releaseOpenCL();
    static std::string readKernel(const std::string &fileName);

    virtual ProcessorType getProcessorType() override;
private:
    static bool isInitialized;

    static cl_context context;
    static cl_command_queue commandQueue;
    static cl_program program;
    static cl_device_id device;

    static unsigned int pID;
    static unsigned int dID;

    double nWidth;
    double nHeight;

#ifdef BUILT_IN_KERNEL
    static const std::string Anime4KCPPKernelSourceString;
#endif // BUILT_IN_KERNEL

};
