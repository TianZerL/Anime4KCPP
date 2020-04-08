#include "postprocessing.h"

PostProcessor::PostProcessor(cv::InputArray srcImg, uint8_t _filters) :
    filters(_filters)
{
    if (!((filters & BILATERAL_FILTER) || (filters & BILATERAL_FILTER_FAST)))
        img = srcImg.getMat();
    else
    {
        srcImg.copyTo(img);
        tmpImg = srcImg.getMat();
    }

}

void PostProcessor::process()
{
        if (filters & MEDIAN_BLUR)
            cv::medianBlur(img, img, 3);
        if (filters & MEAN_BLUR)
            cv::blur(img, img, cv::Size(3, 3));
        if (filters & GAUSSIAN_BLUR_WEAK)
            cv::GaussianBlur(img, img, cv::Size(3, 3), 0.5);
        else if (filters & GAUSSIAN_BLUR)
            cv::GaussianBlur(img, img, cv::Size(3, 3), 1);
        if (filters & BILATERAL_FILTER)
            cv::bilateralFilter(img, tmpImg, 9, 30, 30);
        else if (filters & BILATERAL_FILTER_FAST)
            cv::bilateralFilter(img, tmpImg, 5, 35, 35);
}
