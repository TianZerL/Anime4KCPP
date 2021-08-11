#pragma once

#include"AC.hpp"

namespace Anime4KCPP::CPU
{
    class AC_EXPORT Anime4K09;
}

class Anime4KCPP::CPU::Anime4K09 :public AC
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
