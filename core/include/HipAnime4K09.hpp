#ifndef ANIME4KCPP_CORE_HIP_ANIME4K09_HPP
#define ANIME4KCPP_CORE_HIP_ANIME4K09_HPP

#ifdef ENABLE_HIP

#include <hip/hip_runtime_api.h>
#include "AC.hpp"

namespace Anime4KCPP::Hip
{
    class AC_EXPORT Anime4K09;
}

class Anime4KCPP::Hip::Anime4K09 :public AC
{
public:
    using AC::AC;

    explicit Anime4K09(const Parameters& parameters);
    virtual ~Anime4K09();

    std::string getInfo() const override;
    std::string getFiltersInfo() const override;
private:
	hipStream_t stream;
    void processYUVImage() override;
    void processRGBImage() override;
    void processGrayscale() override;

    Processor::Type getProcessorType() const noexcept override;
    std::string getProcessorInfo() const override;
};

#endif // ENABLE_HIP

#endif // !ANIME4KCPP_CORE_HIP_ANIME4K09_HPP
