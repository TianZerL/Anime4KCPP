#ifndef ANIME4KCPP_CORE_CUDA_ANIME4K09_HPP
#define ANIME4KCPP_CORE_CUDA_ANIME4K09_HPP

#ifdef ENABLE_CUDA

#include"AC.hpp"

namespace Anime4KCPP::Cuda
{
    class AC_EXPORT Anime4K09;
}

class Anime4KCPP::Cuda::Anime4K09 :public AC
{
public:
    using AC::AC;

    std::string getInfo() override;
    std::string getFiltersInfo() override;
private:
    void processYUVImage() override;
    void processRGBImage() override;
    void processGrayscale() override;

    Processor::Type getProcessorType() noexcept override;
    std::string getProcessorInfo() override;
};

#endif // ENABLE_CUDA

#endif // !ANIME4KCPP_CORE_CUDA_ANIME4K09_HPP
