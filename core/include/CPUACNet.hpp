#ifndef ANIME4KCPP_CORE_CPU_ACNET_HPP
#define ANIME4KCPP_CORE_CPU_ACNET_HPP

#include "CPUACNetProcessor.hpp"

namespace Anime4KCPP::CPU
{
    class AC_EXPORT ACNet;
}

class Anime4KCPP::CPU::ACNet :public AC
{
public:
    explicit ACNet(const Parameters& parameters);
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
    std::unique_ptr<ACNetProcessor> processor;
};

#endif // !ANIME4KCPP_CORE_CPU_ACNET_HPP
