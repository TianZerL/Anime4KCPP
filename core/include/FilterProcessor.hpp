#pragma once

#include"AC.hpp"

namespace Anime4KCPP
{
    class FilterProcessor;
}

class Anime4KCPP::FilterProcessor
{
public:
    FilterProcessor(cv::Mat& srcImg, unsigned char filters);
    void process();
    static std::vector<std::string> filterToString(unsigned char filters);
private:
    void CASSharpening(cv::Mat& src);
private:
    int H, W;
    cv::Mat& srcImgRef;
    unsigned char filters;
};
