#pragma once

#ifdef ENABLE_NCNN

#include<net.h>

#include"AC.hpp"
#include"CNN.hpp"

namespace Anime4KCPP
{
    namespace NCNN
    {
        class DLL ACNet;
    }
}

class Anime4KCPP::NCNN::ACNet :public AC
{
public:
    ACNet(const Parameters& parameters = Parameters());
    virtual ~ACNet() = default;
    virtual void setParameters(const Parameters& parameters) override;

    virtual std::string getInfo() override;
    virtual std::string getFiltersInfo() override;

    static void init(
        std::string& modelPath, std::string& paramPath, 
        int type, const int deviceID, const int threads);

    static void init(int type, const int deviceID, const int threads);

    static void init(const int deviceID, const int threads);

    static void release();
    static bool isInitialized();
private:
    void processCPU(const cv::Mat& orgImg, cv::Mat& dstImg, const int scaleTimes = 1, ncnn::Mat* dataHolder = nullptr);
    void processVK(const cv::Mat& orgImg, cv::Mat& dstImg, const int scaleTimes = 1, ncnn::Mat* dataHolder = nullptr);

    virtual void processYUVImageB() override;
    virtual void processRGBImageB() override;
    virtual void processGrayscaleB() override;

    virtual void processYUVImageW() override;
    virtual void processRGBImageW() override;
    virtual void processGrayscaleW() override;

    virtual void processYUVImageF() override;
    virtual void processRGBImageF() override;
    virtual void processGrayscaleF() override;

    virtual Processor::Type getProcessorType() noexcept override;
    virtual std::string getProcessorInfo() override;
private:
    int currACNetypeIndex;

    ncnn::Mat defaultDataHolder;
};

#endif // ENABLE_NCNN
