#pragma once

#include<fstream>

#ifdef __APPLE__
#include<OpenCL/opencl.h>
#else
#include<CL/cl.h>
#endif // SPECIAL OS

#include"FilterProcessor.hpp"

namespace Anime4KCPP
{
    namespace OpenCL
    {
        class DLL Anime4K09;
    }
}

class Anime4KCPP::OpenCL::Anime4K09 :public AC
{
public:
    Anime4K09(const Parameters& parameters = Parameters());
    virtual ~Anime4K09() = default;

    virtual std::string getInfo() override;
    virtual std::string getFiltersInfo() override;

    static void initGPU(unsigned int platformID = 0, unsigned int deviceID = 0);
    static void releaseGPU() noexcept;
    static bool isInitializedGPU();
private:
    virtual void processYUVImage() override;
    virtual void processRGBImage() override;
    virtual void processRGBVideo() override;

    virtual Processor::Type getProcessorType() noexcept override;

    void runKernel(const cv::Mat& orgImg, cv::Mat& dstImg);

    static void initOpenCL();
    static void releaseOpenCL() noexcept;
    static std::string readKernel(const std::string &fileName);
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
