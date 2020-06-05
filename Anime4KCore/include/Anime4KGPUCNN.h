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
    class Anime4KGPUCNN;
}

class Anime4KCPP::Anime4KGPUCNN :public Anime4K
{
public:
    Anime4KGPUCNN(const Parameters& parameters = Parameters());
    virtual ~Anime4KGPUCNN() = default;
    virtual void process() override;
    static void initGPU(unsigned int platformID = 0, unsigned int deviceID = 0);
    static void releaseGPU();
    static bool isInitializedGPU();
private:
    void runKernel(cv::InputArray orgImg, cv::OutputArray dstImg);
    static void initOpenCL();
    static void releaseOpenCL();
    static std::string readKernel(const std::string& fileName);
private:
    static bool isInitialized;

    static cl_context context;
    static cl_command_queue commandQueue;
    static cl_program program;
    static cl_device_id device;

    static unsigned int pID;
    static unsigned int dID;

#ifdef BUILT_IN_KERNEL
    static const std::string ACNetKernelSourceString;
#endif // BUILT_IN_KERNEL

};
