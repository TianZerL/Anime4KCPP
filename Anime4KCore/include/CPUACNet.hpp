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
    virtual ~ACNet() = default;
    virtual void process() override;
    virtual std::string getInfo() override;
    virtual std::string getFiltersInfo() override;
private:
    virtual Processor::Type getProcessorType() noexcept override;
};
