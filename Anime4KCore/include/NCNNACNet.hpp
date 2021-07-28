#pragma once

#ifdef ENABLE_NCNN

#include<net.h>

#include"AC.hpp"
#include"CNN.hpp"

namespace Anime4KCPP::NCNN
{
    class DLL ACNet;
}

class Anime4KCPP::NCNN::ACNet :public AC
{
public:
    explicit ACNet(const Parameters& parameters = Parameters());
    ~ACNet() override = default;
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
    void processCPU(const cv::Mat& orgImg, cv::Mat& dstImg, int scaleTimes = 1, ncnn::Mat* dataHolder = nullptr);
    void processVK(const cv::Mat& orgImg, cv::Mat& dstImg, int scaleTimes = 1, ncnn::Mat* dataHolder = nullptr);

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
    int currACNetypeIndex;

    ncnn::Mat defaultDataHolder;
};

#endif // ENABLE_NCNN
