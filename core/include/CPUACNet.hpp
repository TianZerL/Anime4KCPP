#pragma once

#include"CPUACNetProcessor.hpp"

namespace Anime4KCPP::CPU
{
    class AC_EXPORT ACNet;
}

class Anime4KCPP::CPU::ACNet :public AC
{
public:
    explicit ACNet(const Parameters& parameters = Parameters());
    ~ACNet() override;
    void setParameters(const Parameters& parameters) override;
    std::string getInfo() override;
    std::string getFiltersInfo() override;
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
private:
    ACNetProcessor* processor;
};
