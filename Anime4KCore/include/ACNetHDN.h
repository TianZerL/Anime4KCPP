#pragma once
#include "CNNProcessor.h"

namespace Anime4KCPP
{
    class ACNetHDN;
}

class Anime4KCPP::ACNetHDN : public CNNProcessor
{
public:
    ACNetHDN() = default;
    virtual ~ACNetHDN() = default;
    virtual void process(const cv::Mat& src, cv::Mat& dst) override;
private:
    const static double kernelsL1[9 * 8];
    const static double kernels[8][9 * 8 * 8];
    const static double kernelsL10[4 * 8];
    const static double biasL1[8];
    const static double biases[8][8];
};
