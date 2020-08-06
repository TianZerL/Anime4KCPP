#pragma once
#include "CNNProcessor.h"

namespace Anime4KCPP
{
    class ACNetHDNL1;
    class ACNetHDNL2;
    class ACNetHDNL3;
}

class Anime4KCPP::ACNetHDNL1 : public CNNProcessor
{
public:
    ACNetHDNL1() = default;
    virtual ~ACNetHDNL1() = default;
    virtual void process(const cv::Mat& src, cv::Mat& dst) override;
private:
    const static double kernelsL1[9 * 8];
    const static double kernels[8][9 * 8 * 8];
    const static double kernelsL10[4 * 8];
    const static double biasL1[8];
    const static double biases[8][8];
};

class Anime4KCPP::ACNetHDNL2 : public CNNProcessor
{
public:
    ACNetHDNL2() = default;
    virtual ~ACNetHDNL2() = default;
    virtual void process(const cv::Mat& src, cv::Mat& dst) override;
private:
    const static double kernelsL1[9 * 8];
    const static double kernels[8][9 * 8 * 8];
    const static double kernelsL10[4 * 8];
    const static double biasL1[8];
    const static double biases[8][8];
};

class Anime4KCPP::ACNetHDNL3 : public CNNProcessor
{
public:
    ACNetHDNL3() = default;
    virtual ~ACNetHDNL3() = default;
    virtual void process(const cv::Mat& src, cv::Mat& dst) override;
private:
    const static double kernelsL1[9 * 8];
    const static double kernels[8][9 * 8 * 8];
    const static double kernelsL10[4 * 8];
    const static double biasL1[8];
    const static double biases[8][8];
};