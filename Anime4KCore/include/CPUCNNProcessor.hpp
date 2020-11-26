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
    void conv1To8B(const cv::Mat& img, const double* kernels, const double* biases, cv::Mat& tmpMat);
    void conv1To8F(const cv::Mat& img, const double* kernels, const double* biases, cv::Mat& tmpMat);
    void conv8To8(const double* kernels, const double* biases, cv::Mat& tmpMat);
    void convTranspose8To1B(cv::Mat& img, const double* kernels, cv::Mat& tmpMat);
    void convTranspose8To1F(cv::Mat& img, const double* kernels, cv::Mat& tmpMat);

    void changEachPixel1ToN(const cv::Mat& src, const std::function<void(int, int, ChanD, LineB)>&& callBack, cv::Mat& tmpMat, int outChannels);
    void changEachPixel1ToN(const cv::Mat& src, const std::function<void(int, int, ChanD, LineF)>&& callBack, cv::Mat& tmpMat, int outChannels);
    void changEachPixelNToN(const std::function<void(int, int, ChanD, LineD)>&& callBack, cv::Mat& tmpMat);
    void changEachPixelNTo1(cv::Mat& img, const std::function<void(int, int, ChanB, LineD)>&& callBack, const cv::Mat& tmpMat);
    void changEachPixelNTo1(cv::Mat& img, const std::function<void(int, int, ChanF, LineD)>&& callBack, const cv::Mat& tmpMat);
};
