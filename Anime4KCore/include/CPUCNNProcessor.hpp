#pragma once

#include "AC.hpp"

namespace Anime4KCPP
{
    namespace CPU
    {
        class CNNProcessor;
    }
}

class Anime4KCPP::CPU::CNNProcessor
{
protected:
    void conv1To8(const cv::Mat& img, const double* kernels, const double* biases, cv::Mat& tmpMat);
    void conv8To8(const double* kernels, const double* biases, cv::Mat& tmpMat);
    void convTranspose8To1(cv::Mat& img, const double* kernels, cv::Mat& tmpMat);

    void changEachPixel1ToN(const cv::Mat& src, const std::function<void(int, int, ChanF, LineB)>&& callBack, cv::Mat& tmpMat, int outChannels);
    void changEachPixelNToN(const std::function<void(int, int, ChanF, LineF)>&& callBack, cv::Mat& tmpMat);
    void changEachPixelNTo1(cv::Mat& img, const std::function<void(int, int, ChanB, LineF)>&& callBack, const cv::Mat& tmpMat);
};
