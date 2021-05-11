#pragma once

#include"AC.hpp"

namespace Anime4KCPP
{
    namespace CPU
    {
        class DLL Anime4K09;
    }
}

class Anime4KCPP::CPU::Anime4K09 :public AC
{
public:
    Anime4K09(const Parameters& parameters = Parameters());
    virtual ~Anime4K09() = default;

    virtual std::string getInfo() override;
    virtual std::string getFiltersInfo() override;
private:
    virtual void processYUVImageB() override;
    virtual void processRGBImageB() override;
    virtual void processGrayscaleB() override;

    virtual void processYUVImageW() override;
    virtual void processRGBImageW() override;
    virtual void processGrayscaleW() override;

    virtual void processYUVImageF() override;
    virtual void processRGBImageF() override;
    virtual void processGrayscaleF() override;

    void getGrayB(cv::Mat& img);
    void pushColorB(cv::Mat& img);
    void getGradientB(cv::Mat& img);
    void pushGradientB(cv::Mat& img);

    void getGrayW(cv::Mat& img);
    void pushColorW(cv::Mat& img);
    void getGradientW(cv::Mat& img);
    void pushGradientW(cv::Mat& img);

    void getGrayF(cv::Mat& img);
    void pushColorF(cv::Mat& img);
    void getGradientF(cv::Mat& img);
    void pushGradientF(cv::Mat& img);

    void getLightest(PixelB mc, const PixelB a, const PixelB b, const PixelB c) noexcept;
    void getLightest(PixelW mc, const PixelW a, const PixelW b, const PixelW c) noexcept;
    void getLightest(PixelF mc, const PixelF a, const PixelF b, const PixelF c) noexcept;
    void getAverage(PixelB mc, const PixelB a, const PixelB b, const PixelB c) noexcept;
    void getAverage(PixelW mc, const PixelW a, const PixelW b, const PixelW c) noexcept;
    void getAverage(PixelF mc, const PixelF a, const PixelF b, const PixelF c) noexcept;

    virtual Processor::Type getProcessorType() noexcept override;
    virtual std::string getProcessorInfo() override;
};
