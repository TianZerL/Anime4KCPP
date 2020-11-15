#pragma once

#ifdef ENABLE_CUDA

#include"CudaInterface.hpp"
#include"AC.hpp"
#include"CNN.hpp"

namespace Anime4KCPP
{
    namespace Cuda
    {
        class DLL ACNet;
    }
}

class Anime4KCPP::Cuda::ACNet :public AC
{
public:
    ACNet(const Parameters& parameters = Parameters());
    virtual ~ACNet() = default;

    virtual std::string getInfo() override;
    virtual std::string getFiltersInfo() override;
private:
    void runKernel(const cv::Mat& orgImg, cv::Mat& dstImg);

    virtual void processYUVImage() override;
    virtual void processRGBImage() override;
    virtual void processRGBVideo() override;

    virtual Processor::Type getProcessorType() noexcept override;
};

#endif
