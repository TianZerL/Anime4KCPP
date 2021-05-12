#pragma once

#ifdef ENABLE_OPENCL

#include"AC.hpp"
#include"CNN.hpp"

namespace Anime4KCPP
{
    namespace OpenCL
    {
        class DLL ACNet;
    }
}

class Anime4KCPP::OpenCL::ACNet :public AC
{
public:
    ACNet(const Parameters& parameters = Parameters());
    virtual ~ACNet() = default;
    virtual void setParameters(const Parameters& parameters) override;

    virtual std::string getInfo() override;
    virtual std::string getFiltersInfo() override;

    static void initGPU(
        const int platformID = 0,
        const int deviceID = 0,
        const CNNType type = CNNType::Default, 
        const int OpenCLQueueNum = 4,
        const bool OpenCLParallelIO = false);
    static void releaseGPU() noexcept;
    static bool isInitializedGPU() noexcept;
private:
    virtual void processYUVImageB() override;
    virtual void processRGBImageB() override;
    virtual void processGrayscaleB() override;

    virtual void processYUVImageW() override;
    virtual void processRGBImageW() override;
    virtual void processGrayscaleW() override;

    virtual void processYUVImageF() override;
    virtual void processRGBImageF() override;
    virtual void processGrayscaleF() override;

    virtual Processor::Type getProcessorType() noexcept override;
    virtual std::string getProcessorInfo() override;

    void runKernel(const cv::Mat& orgImg, cv::Mat& dstImg, int channelType);
    void runKernelP(const cv::Mat& orgImg, cv::Mat& dstImg, int channelType);

    void runKernelB(const cv::Mat& orgImg, cv::Mat& dstImg);
    void runKernelW(const cv::Mat& orgImg, cv::Mat& dstImg);
    void runKernelF(const cv::Mat& orgImg, cv::Mat& dstImg);

    void runKernelPB(const cv::Mat& orgImg, cv::Mat& dstImg);
    void runKernelPW(const cv::Mat& orgImg, cv::Mat& dstImg);
    void runKernelPF(const cv::Mat& orgImg, cv::Mat& dstImg);

    static void initOpenCL(const CNNType type);
    static std::string readKernel(const std::string& fileName);
private:
    int currACNetypeIndex;
};

#endif
