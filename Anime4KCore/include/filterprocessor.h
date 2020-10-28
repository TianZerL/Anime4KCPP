#pragma once

#include"Anime4K.h"

#define MAX5(a, b, c, d, e) std::max({a, b, c, d, e})
#define MIN5(a, b, c, d, e) std::min({a, b, c, d, e})
#define LERP(x, y, w) ((x) * (1.0 - (w)) + (y) * (w))
#define REC(n) ((n) < 1 ? 1.0 : 1.0 / (n))
#define UNFLOAT(n) ((n) >= 255 ? 255 : ((n) <= 0 ? 0 : uint8_t((n) + 0.5)))

namespace Anime4KCPP
{
    class FilterProcessor;
}

class Anime4KCPP::FilterProcessor
{
public:
    FilterProcessor(cv::Mat& srcImg, uint8_t _filters);
    void process();
private:
    void CASSharpening(cv::Mat& src);
    void changEachPixelBGR(cv::Mat& src, const std::function<void(const int, const int, PixelB, LineB)>&& callBack);
private:
    int H, W;
    cv::Mat img, tmpImg;
    cv::Mat& srcImgRef;
    uint8_t filters;
};
