#pragma once

#ifdef ENABLE_OPENCL

#include"AC.hpp"
#include"CNN.hpp"

namespace Anime4KCPP::OpenCL
{
    class AC_EXPORT ACNet;
}

class Anime4KCPP::OpenCL::ACNet :public AC
{
public:
    explicit ACNet(const Parameters& parameters);
    void setParameters(const Parameters& parameters) override;

    std::string getInfo() override;
    std::string getFiltersInfo() override;

    static void init(
        const int platformID = 0,
        const int deviceID = 0,
        const CNNType type = CNNType::Default,
        const int OpenCLQueueNum = 4,
        const bool OpenCLParallelIO = false);
    static void release() noexcept;
    static bool isInitialized() noexcept;
private:
    void processYUVImage();
    void processRGBImage();
    void processGrayscale();

    Processor::Type getProcessorType() noexcept override;
    std::string getProcessorInfo() override;

    static void initOpenCL(const CNNType type);
private:
    int ACNetTypeIndex;
};

#endif
