#pragma once

#include"AC.hpp"

namespace Anime4KCPP
{
    class FilterProcessor;
}

class Anime4KCPP::FilterProcessor
{
public:
    FilterProcessor(cv::Mat& srcImg, uint8_t _filters);
    void process();
    static std::vector<std::string> filterToString(uint8_t filters);
private:
    void CASSharpening(cv::Mat& src);
    void changEachPixelBGR(cv::Mat& src, const std::function<void(const int, const int, PixelB, LineB)>&& callBack);
private:
    int H, W;
    cv::Mat img, tmpImg;
    cv::Mat& srcImgRef;
    uint8_t filters;
};
