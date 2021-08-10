#pragma once

#ifdef ENABLE_CUDA

#include"AC.hpp"
#include"CNN.hpp"

namespace Anime4KCPP::Cuda
{
    class AC_EXPORT ACNet;
}

class Anime4KCPP::Cuda::ACNet :public AC
{
public:
    explicit ACNet(const Parameters& parameters = Parameters());
    ~ACNet() override = default;
    void setParameters(const Parameters& parameters) override;

    std::string getInfo() override;
    std::string getFiltersInfo() override;
private:
    void processYUVImage() override;
    void processRGBImage() override;
    void processGrayscale() override;

    Processor::Type getProcessorType() noexcept override;
    std::string getProcessorInfo() override;
private:
    int ACNetTypeIndex;
};

#endif
