#pragma once

#ifdef ENABLE_CUDA

#include"CudaInterface.hpp"
#include"FilterProcessor.hpp"

namespace Anime4KCPP
{
    namespace Cuda
    {
        class DLL Anime4K09;
    }
}

class Anime4KCPP::Cuda::Anime4K09 :public AC
{
public:
    Anime4K09(const Parameters& parameters = Parameters());
    virtual ~Anime4K09() = default;

    virtual std::string getInfo() override;
    virtual std::string getFiltersInfo() override;
private:
    void runKernel(cv::Mat& orgImg, cv::Mat& dstImg);

    virtual void processYUVImage() override;
    virtual void processRGBImage() override;
    virtual void processRGBVideo() override;

    virtual Processor::Type getProcessorType() noexcept override;
};

#endif
