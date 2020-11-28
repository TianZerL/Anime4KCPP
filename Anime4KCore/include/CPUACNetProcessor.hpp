#pragma once

#include"CPUCNNProcessor.hpp"
#include"CNN.hpp"

namespace Anime4KCPP
{
    namespace CPU
    {
        constexpr int L2 = 0, L3 = 1, L4 = 2, L5 = 3, L6 = 4, L7 = 5, L8 = 6, L9 = 7;

        class ACNetProcessor;
        class ACNetHDNL0;
        class ACNetHDNL1;
        class ACNetHDNL2;
        class ACNetHDNL3;

        ACNetProcessor* createACNetProcessor(const CNNType type);
        void releaseACNetProcessor(ACNetProcessor* processor) noexcept;
    }
}

class Anime4KCPP::CPU::ACNetProcessor : public CNNProcessor
{
public:
    ACNetProcessor() = default;;
    virtual ~ACNetProcessor() = default;

    virtual void processB(const cv::Mat & src, cv::Mat & dst) = 0;
    virtual void processW(const cv::Mat & src, cv::Mat & dst) = 0;
    virtual void processF(const cv::Mat& src, cv::Mat& dst) = 0;
};

class Anime4KCPP::CPU::ACNetHDNL0 : public ACNetProcessor
{
public:
    ACNetHDNL0() = default;
    virtual ~ACNetHDNL0() = default;
    virtual void processB(const cv::Mat & src, cv::Mat & dst) override;
    virtual void processW(const cv::Mat & src, cv::Mat & dst) override;
    virtual void processF(const cv::Mat & src, cv::Mat & dst) override;
private:
    const static double kernelsL1[9 * 8];
    const static double kernels[8][9 * 8 * 8];
    const static double kernelsL10[4 * 8];
    const static double biasL1[8];
    const static double biases[8][8];
};

class Anime4KCPP::CPU::ACNetHDNL1 : public ACNetProcessor
{
public:
    ACNetHDNL1() = default;
    virtual ~ACNetHDNL1() = default;
    virtual void processB(const cv::Mat& src, cv::Mat& dst) override;
    virtual void processW(const cv::Mat& src, cv::Mat& dst) override;
    virtual void processF(const cv::Mat& src, cv::Mat& dst) override;
private:
    const static double kernelsL1[9 * 8];
    const static double kernels[8][9 * 8 * 8];
    const static double kernelsL10[4 * 8];
    const static double biasL1[8];
    const static double biases[8][8];
};

class Anime4KCPP::CPU::ACNetHDNL2 : public ACNetProcessor
{
public:
    ACNetHDNL2() = default;
    virtual ~ACNetHDNL2() = default;
    virtual void processB(const cv::Mat& src, cv::Mat& dst) override;
    virtual void processW(const cv::Mat& src, cv::Mat& dst) override;
    virtual void processF(const cv::Mat& src, cv::Mat& dst) override;
private:
    const static double kernelsL1[9 * 8];
    const static double kernels[8][9 * 8 * 8];
    const static double kernelsL10[4 * 8];
    const static double biasL1[8];
    const static double biases[8][8];
};

class Anime4KCPP::CPU::ACNetHDNL3 : public ACNetProcessor
{
public:
    ACNetHDNL3() = default;
    virtual ~ACNetHDNL3() = default;
    virtual void processB(const cv::Mat& src, cv::Mat& dst) override;
    virtual void processW(const cv::Mat& src, cv::Mat& dst) override;
    virtual void processF(const cv::Mat& src, cv::Mat& dst) override;
private:
    const static double kernelsL1[9 * 8];
    const static double kernels[8][9 * 8 * 8];
    const static double kernelsL10[4 * 8];
    const static double biasL1[8];
    const static double biases[8][8];
};
