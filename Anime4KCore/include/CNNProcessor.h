#pragma once
#include "Anime4K.h"

#ifdef _MSC_VER
#include<ppl.h>
#else
#include<omp.h>
#endif

#define RULE(x) std::max(x, 0.0)
#define NORM(X) (double(X) / 255.0)
#define UNNORM(n) ((n) >= 255.0? uint8_t(255) : ((n) <= 0.0 ? uint8_t(0) : uint8_t(n)))

namespace Anime4KCPP
{
    class CNNProcessor;

    enum Layer
    {
        L2 = 0, L3 = 1, L4 = 2, L5 = 3, L6 = 4, L7 = 5, L8 = 6, L9 = 7
    };

    enum class CNNType;
}

enum class Anime4KCPP::CNNType
{
    Default = 0, ACNet = 1, ACNetHDN = 2
};

class Anime4KCPP::CNNProcessor
{
public:
    CNNProcessor() = default;;
    virtual ~CNNProcessor() = default;

    virtual void process(const cv::Mat& src, cv::Mat& dst) = 0;
protected:
    void conv1To8(cv::InputArray img, const double* kernels, const double* biases, std::pair<cv::Mat, cv::Mat>& tmpMats);
    void conv8To8(const double* kernels, const double* biases, std::pair<cv::Mat, cv::Mat>& tmpMats);
    void convTranspose8To1(cv::Mat& img, const double* kernels, std::pair<cv::Mat, cv::Mat>& tmpMats);

    void changEachPixel1To8(cv::InputArray _src, const std::function<void(int, int, Chan, Chan, LineC)>&& callBack, std::pair<cv::Mat, cv::Mat>& tmpMats);
    void changEachPixel8To8(const std::function<void(int, int, Chan, Chan, LineF, LineF)>&& callBack, std::pair<cv::Mat, cv::Mat>& tmpMats);
    void changEachPixel8To1(cv::Mat& img, const std::function<void(int, int, PIXEL, LineF, LineF)>&& callBack, std::pair<cv::Mat, cv::Mat>& tmpMats);
};
