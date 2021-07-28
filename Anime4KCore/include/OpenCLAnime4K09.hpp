#pragma once

#ifdef ENABLE_OPENCL

#include"AC.hpp"

namespace Anime4KCPP::OpenCL
{
    class DLL Anime4K09;
}

class Anime4KCPP::OpenCL::Anime4K09 :public AC
{
public:
    explicit Anime4K09(const Parameters& parameters = Parameters());
    ~Anime4K09() override = default;

    std::string getInfo() override;
    std::string getFiltersInfo() override;

    static void init(int platformID = 0, int deviceID = 0, int OpenCLQueueNum = 4, bool OpenCLParallelIO = false);
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

    static void initOpenCL();
    static std::string readKernel(const std::string &fileName);
private:
    double nWidth;
    double nHeight;
};

#endif
