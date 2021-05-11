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
private:
    int H, W;
    cv::Mat& srcImgRef;
    uint8_t filters;
};
