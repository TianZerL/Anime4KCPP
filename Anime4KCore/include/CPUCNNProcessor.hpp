#pragma once

#ifndef ENABLE_OPENCV_DNN

#include"AC.hpp"

namespace Anime4KCPP
{
    namespace CPU
    {
        typedef float* ChanFP;
        typedef float* LineFP;
        typedef float* PixelFP;

        class CNNProcessor;
    }
}

class Anime4KCPP::CPU::CNNProcessor
{
protected:
    void conv1To8B(const cv::Mat& img, const float* kernels, const float* biases, cv::Mat& tmpMat);
    void conv1To8W(const cv::Mat& img, const float* kernels, const float* biases, cv::Mat& tmpMat);
    void conv1To8F(const cv::Mat& img, const float* kernels, const float* biases, cv::Mat& tmpMat);
    void conv8To8(const float* kernels, const float* biases, cv::Mat& tmpMat);
    void convTranspose8To1B(cv::Mat& img, const float* kernels, cv::Mat& tmpMat);
    void convTranspose8To1W(cv::Mat& img, const float* kernels, cv::Mat& tmpMat);
    void convTranspose8To1F(cv::Mat& img, const float* kernels, cv::Mat& tmpMat);
};

#endif // !ENABLE_OPENCV_DNN
