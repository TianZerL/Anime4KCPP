#pragma once
#include<opencv2/opencv.hpp>

enum FilterType : uint8_t
{
    MEDIAN_BLUR = 1, MEAN_BLUR = 2, GAUSSIAN_BLUR = 4, 
    BILATERAL_FILTER = 8, BILATERAL_FILTER_FAST = 16
};


class PostProcessor
{
public:
    PostProcessor(cv::InputArray srcImg, uint8_t _filters);
    void process();
private:
    cv::Mat img,tmpImg;
    uint8_t filters;
};