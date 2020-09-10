#pragma once
#include "Anime4K.h"

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
    Default, ACNet, ACNetHDNL1, ACNetHDNL2, ACNetHDNL3
};

class Anime4KCPP::CNNProcessor
{
public:
    CNNProcessor() = default;;
    virtual ~CNNProcessor() = default;

    virtual void process(const cv::Mat& src, cv::Mat& dst) = 0;
protected:
    void conv1To8(const cv::Mat& img, const double* kernels, const double* biases, cv::Mat& tmpMat);
    void conv8To8(const double* kernels, const double* biases, cv::Mat& tmpMat);
    void convTranspose8To1(cv::Mat& img, const double* kernels, cv::Mat& tmpMat);

    void changEachPixel1ToN(const cv::Mat& src, const std::function<void(int, int, ChanF, LineB)>&& callBack, cv::Mat& tmpMat, int outChannels);
    void changEachPixelNToN(const std::function<void(int, int, ChanF, LineF)>&& callBack, cv::Mat& tmpMat);
    void changEachPixelNTo1(cv::Mat& img, const std::function<void(int, int, ChanB, LineF)>&& callBack, const cv::Mat& tmpMat);
};
