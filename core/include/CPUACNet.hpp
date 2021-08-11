#pragma once

#include"CPUACNetProcessor.hpp"

namespace Anime4KCPP::CPU
{
    class AC_EXPORT ACNet;
}

class Anime4KCPP::CPU::ACNet :public AC
{
public:
    explicit ACNet(const Parameters& parameters);
    ~ACNet() override;
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
    ACNetProcessor* processor;
};
