#pragma once

#include"AC.hpp"

namespace Anime4KCPP::CPU
{
    class DLL Anime4K09;
}

class Anime4KCPP::CPU::Anime4K09 :public AC
{
public:
    explicit Anime4K09(const Parameters& parameters = Parameters());
    ~Anime4K09() override = default;

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

    void getLightest(PixelB mc, PixelB a, PixelB b, PixelB c) noexcept;
    void getLightest(PixelW mc, PixelW a, PixelW b, PixelW c) noexcept;
    void getLightest(PixelF mc, PixelF a, PixelF b, PixelF c) noexcept;
    void getAverage(PixelB mc, PixelB a, PixelB b, PixelB c) noexcept;
    void getAverage(PixelW mc, PixelW a, PixelW b, PixelW c) noexcept;
    void getAverage(PixelF mc, PixelF a, PixelF b, PixelF c) noexcept;

    Processor::Type getProcessorType() noexcept override;
    std::string getProcessorInfo() override;
};
