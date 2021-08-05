#pragma once

#ifdef ENABLE_OPENCL

#include"AC.hpp"
#include"CNN.hpp"

namespace Anime4KCPP::OpenCL
{
    class AC_EXPORT ACNet;
}

class Anime4KCPP::OpenCL::ACNet :public AC
{
public:
    explicit ACNet(const Parameters& parameters = Parameters());
    ~ACNet() override = default;
    void setParameters(const Parameters& parameters) override;

    std::string getInfo() override;
    std::string getFiltersInfo() override;

    static void init(
        const int platformID = 0,
        const int deviceID = 0,
        const CNNType type = CNNType::Default,
        const int OpenCLQueueNum = 4,
        const bool OpenCLParallelIO = false);
    static void release() noexcept;
    static bool isInitialized() noexcept;
private:
    void processYUVImageB() override;
    void processRGBImageB() override;
    void processGrayscaleB() override;

    void processYUVImageW() override;
    void processRGBImageW() override;
    void processGrayscaleW() override;

    void processYUVImageF() override;
    void processRGBImageF() override;
    void processGrayscaleF() override;

    Processor::Type getProcessorType() noexcept override;
    std::string getProcessorInfo() override;

    void runKernel(const cv::Mat& orgImg, cv::Mat& dstImg, int channelType) const;
    void runKernelP(const cv::Mat& orgImg, cv::Mat& dstImg, int channelType) const;

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
