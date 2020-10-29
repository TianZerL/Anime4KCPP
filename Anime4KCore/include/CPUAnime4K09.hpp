#pragma once

#include"FilterProcessor.hpp"

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
    virtual void process() override;
    virtual std::string getInfo() override;
    virtual std::string getFiltersInfo() override;
private:
    void getGray(cv::Mat& img);
    void pushColor(cv::Mat& img);
    void getGradient(cv::Mat& img);
    void pushGradient(cv::Mat& img);
    void changEachPixelBGRA(cv::Mat& src, const std::function<void(int, int, PixelB, LineB)>&& callBack);
    void getLightest(PixelB mc, PixelB a, PixelB b, PixelB c) noexcept;
    void getAverage(PixelB mc, PixelB a, PixelB b, PixelB c) noexcept;

    virtual Processor::Type getProcessorType() noexcept override;
};
