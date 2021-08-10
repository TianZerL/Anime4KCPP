#pragma once

#ifdef ENABLE_CUDA

#include"AC.hpp"

namespace Anime4KCPP::Cuda
{
    class AC_EXPORT Anime4K09;
}

class Anime4KCPP::Cuda::Anime4K09 :public AC
{
public:
    explicit Anime4K09(const Parameters& parameters = Parameters());
    ~Anime4K09() override = default;

    std::string getInfo() override;
    std::string getFiltersInfo() override;
private:
    void processYUVImage() override;
    void processRGBImage() override;
    void processGrayscale() override;

    Processor::Type getProcessorType() noexcept override;
    std::string getProcessorInfo() override;
};

#endif
