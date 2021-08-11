#pragma once

#include"AC.hpp"

namespace Anime4KCPP
{
    class FilterProcessor;
}

class Anime4KCPP::FilterProcessor
{
public:
    FilterProcessor(cv::Mat& srcImg, std::uint8_t filters);
    void process();
    static std::vector<std::string> filterToString(std::uint8_t filters);
private:
    void CASSharpening(cv::Mat& src);
private:
    int H, W;
    cv::Mat& srcImgRef;
    std::uint8_t filters;
};
