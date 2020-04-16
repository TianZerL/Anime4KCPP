#pragma once
#include<opencv2/opencv.hpp>

#ifdef _MSC_VER
#include<ppl.h>
#else
#include<omp.h>
#endif

#define MAX5(a, b, c, d, e) (a > b && a > c && a > d && a > e ? a : (b > c && b > d && b > e ? b : (c > d && c > e ? c : ( d > e ? d : e))))
#define MIN5(a, b, c, d, e) (a < b && a < c && a < d && a < e ? a : (b < c && b < d && b < e ? b : (c < d && c < e ? c : ( d < e ? d : e))))
#define LERP(x, y, w) ((x) * (1 - (w)) + (y) * (w))
#define REC(n) ((n) < 1 ? 1.0 : 1.0 / (n))
#define UNFLOAT(n) (n >= 255 ? 255 : (n <= 0 ? 0 : uint8_t(n + 0.5)))

typedef unsigned char* RGBA;
typedef unsigned char* Line;

enum FilterType : uint8_t
{
    MEDIAN_BLUR = 1, MEAN_BLUR = 2, CAS_SHARPENING = 4,
    GAUSSIAN_BLUR_WEAK = 8, GAUSSIAN_BLUR = 16,
    BILATERAL_FILTER = 32, BILATERAL_FILTER_FAST = 64
};


class FilterProcessor
{
public:
    FilterProcessor(cv::InputArray srcImg, uint8_t _filters);
    void process();
private:
    void CASSharpening(cv::InputArray src);
    void changEachPixelBGR(cv::InputArray _src, const std::function<void(int, int, RGBA, Line)>&& callBack);
private:
    const static int B = 0, G = 1, R = 2, A = 3;
    int H, W;
    cv::Mat img, tmpImg;
    uint8_t filters;
};
