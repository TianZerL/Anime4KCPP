#pragma once

#ifdef ENABLE_CUDA

#include"AC.hpp"

namespace Anime4KCPP::Cuda
{
    class DLL Anime4K09;
}

class Anime4KCPP::Cuda::Anime4K09 :public AC
{
public:
    explicit Anime4K09(const Parameters& parameters = Parameters());
    ~Anime4K09() override = default;

    std::string getInfo() override;
    std::string getFiltersInfo() override;
private:
    void runKernelB(const cv::Mat& orgImg, cv::Mat& dstImg);
    void runKernelW(const cv::Mat& orgImg, cv::Mat& dstImg);
    void runKernelF(const cv::Mat& orgImg, cv::Mat& dstImg);

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
};

#endif
