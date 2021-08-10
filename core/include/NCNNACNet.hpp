#pragma once

#ifdef ENABLE_NCNN

#include"AC.hpp"
#include"CNN.hpp"

namespace Anime4KCPP::NCNN
{
    class AC_EXPORT ACNet;
}

class Anime4KCPP::NCNN::ACNet :public AC
{
public:
    explicit ACNet(const Parameters& parameters = Parameters());
    ~ACNet() override;
    void setParameters(const Parameters& parameters) override;

    std::string getInfo() override;
    std::string getFiltersInfo() override;

    static void init(
        std::string& modelPath, std::string& paramPath, 
        int type, int deviceID, int threads);

    static void init(int type, int deviceID, int threads);

    static void init(int deviceID, int threads);

    static void release();
    static bool isInitialized();
private:
    void processYUVImage() override;
    void processRGBImage() override;
    void processGrayscale() override;

    Processor::Type getProcessorType() noexcept override;
    std::string getProcessorInfo() override;
private:
    int ACNetTypeIndex;

    struct DataHolder;
    std::unique_ptr<DataHolder> dataHolder;
};

#endif // ENABLE_NCNN
