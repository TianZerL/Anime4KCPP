#pragma once

#ifdef ENABLE_OPENCL

#include"AC.hpp"

namespace Anime4KCPP::OpenCL
{
    class AC_EXPORT Anime4K09;
}

class Anime4KCPP::OpenCL::Anime4K09 :public AC
{
public:
    using AC::AC;

    std::string getInfo() override;
    std::string getFiltersInfo() override;

    static void init(int platformID = 0, int deviceID = 0, int OpenCLQueueNum = 4, bool OpenCLParallelIO = false);
    static void release() noexcept;
    static bool isInitialized() noexcept;
private:
    void processYUVImage() override;
    void processRGBImage() override;
    void processGrayscale() override;

    Processor::Type getProcessorType() noexcept override;
    std::string getProcessorInfo() override;

    static void initOpenCL();
};

#endif
