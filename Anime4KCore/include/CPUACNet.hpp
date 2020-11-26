#pragma once

#include"CPUACNetProcessor.hpp"

namespace Anime4KCPP
{
    namespace CPU
    {
        class DLL ACNet;
    }
}

class Anime4KCPP::CPU::ACNet :public AC
{
public:
    ACNet(const Parameters& parameters = Parameters());
    virtual ~ACNet();
    virtual void setArguments(const Parameters& parameters) override;
    virtual std::string getInfo() override;
    virtual std::string getFiltersInfo() override;
private:
    virtual void processYUVImageB() override;
    virtual void processRGBImageB() override;
    virtual void processRGBVideoB() override;

    virtual void processYUVImageF() override;
    virtual void processRGBImageF() override;

    virtual Processor::Type getProcessorType() noexcept override;
private:
    ACNetProcessor* processor;
};
