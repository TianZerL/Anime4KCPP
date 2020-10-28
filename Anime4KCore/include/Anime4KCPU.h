#pragma once

#include"Anime4K.h"
#include"filterprocessor.h"

#define MAX3(a, b, c) std::max({a, b, c})
#define MIN3(a, b, c) std::min({a, b, c})
#define UNFLOAT(n) ((n) >= 255 ? 255 : ((n) <= 0 ? 0 : uint8_t((n) + 0.5)))

namespace Anime4KCPP
{
    class DLL Anime4KCPU;
}

class Anime4KCPP::Anime4KCPU :public Anime4K
{
public:
    Anime4KCPU(const Parameters& parameters = Parameters());
    virtual ~Anime4KCPU() = default;
    virtual void process() override;
private:
    void getGray(cv::Mat& img);
    void pushColor(cv::Mat& img);
    void getGradient(cv::Mat& img);
    void pushGradient(cv::Mat& img);
    void changEachPixelBGRA(cv::Mat& src, const std::function<void(int, int, PixelB, LineB)>&& callBack);
    void getLightest(PixelB mc, PixelB a, PixelB b, PixelB c) noexcept;
    void getAverage(PixelB mc, PixelB a, PixelB b, PixelB c) noexcept;

    virtual ProcessorType getProcessorType() noexcept override;
};
