#pragma once

#include<fstream>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
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

    static void initGPU(const int platformID = 0, const int deviceID = 0, const int OpenCLQueueNum = 4, const bool OpenCLParallelIO = false);
    static void releaseGPU() noexcept;
    static bool isInitializedGPU();
private:
    virtual void processYUVImageB() override;
    virtual void processRGBImageB() override;
    virtual void processGrayscaleB() override;
    virtual void processRGBVideoB() override;

    virtual void processYUVImageW() override;
    virtual void processRGBImageW() override;
    virtual void processGrayscaleW() override;

    virtual void processYUVImageF() override;
    virtual void processRGBImageF() override;
    virtual void processGrayscaleF() override;

    virtual Processor::Type getProcessorType() noexcept override;
    virtual std::string getProcessorInfo() override;

    void runKernelB(const cv::Mat& orgImg, cv::Mat& dstImg);
    void runKernelW(const cv::Mat& orgImg, cv::Mat& dstImg);
    void runKernelF(const cv::Mat& orgImg, cv::Mat& dstImg);

    void runKernelPB(const cv::Mat& orgImg, cv::Mat& dstImg);
    void runKernelPW(const cv::Mat& orgImg, cv::Mat& dstImg);
    void runKernelPF(const cv::Mat& orgImg, cv::Mat& dstImg);

    static void initOpenCL();
    static void releaseOpenCL() noexcept;
    static std::string readKernel(const std::string &fileName);
private:
    double nWidth;
    double nHeight;

    static bool isInitialized;

    static cl_context context;

    static int commandQueueNum;
    static int commandQueueCount;
    static std::vector<cl_command_queue> commandQueueList;
    static bool parallelIO;
    static cl_command_queue commandQueueIO;

    static cl_program program;
    static cl_device_id device;

    static int pID;
    static int dID;

    static size_t workGroupSizeLog;
};
