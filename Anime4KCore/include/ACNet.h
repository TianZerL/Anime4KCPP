#pragma once
#include "CNNProcessor.h"

namespace Anime4KCPP
{
    class ACNet;
}

class Anime4KCPP::ACNet : public CNNProcessor
{
public:
    ACNet() = default;
    virtual ~ACNet() = default;
    virtual void process(const cv::Mat& src, cv::Mat& dst) override;
private:
    const static double kernelsL1[9 * 8];
    const static double kernels[8][9 * 8 * 8];
    const static double kernelsL10[4 * 8];
    const static double biasL1[8];
    const static double biases[8][8];
};
