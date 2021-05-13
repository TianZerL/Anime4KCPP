#pragma once

#include"AC.hpp"
#include"CPUCNNProcessor.hpp"
#include"CNN.hpp"

namespace Anime4KCPP
{
    namespace CPU
    {
        class ACNetProcessor;
#ifdef ENABLE_OPENCV_DNN
        class ACNetHDN;
#else
        class ACNetHDNL0;
        class ACNetHDNL1;
        class ACNetHDNL2;
        class ACNetHDNL3;
#endif
        ACNetProcessor* createACNetProcessor(const CNNType type);
        void releaseACNetProcessor(ACNetProcessor* processor) noexcept;
    }
}

class Anime4KCPP::CPU::ACNetProcessor 
{
public:
    ACNetProcessor() = default;
    virtual ~ACNetProcessor() = default;

    virtual void processB(const cv::Mat& src, cv::Mat& dst, int scaleTimes) = 0;
    virtual void processW(const cv::Mat& src, cv::Mat& dst, int scaleTimes) = 0;
    virtual void processF(const cv::Mat& src, cv::Mat& dst, int scaleTimes) = 0;
};

#ifdef ENABLE_OPENCV_DNN
class Anime4KCPP::CPU::ACNetHDN : public ACNetProcessor
{
public:
    ACNetHDN(std::string modelPath);
    virtual ~ACNetHDN() = default;
    virtual void processB(const cv::Mat& src, cv::Mat& dst, int scaleTimes) override;
    virtual void processW(const cv::Mat& src, cv::Mat& dst, int scaleTimes) override;
    virtual void processF(const cv::Mat& src, cv::Mat& dst, int scaleTimes) override;
private:
    cv::dnn::Net net;
};

#else

class Anime4KCPP::CPU::ACNetHDNL0 : public ACNetProcessor, public CNNProcessor
{
public:
    ACNetHDNL0() = default;
    virtual ~ACNetHDNL0() = default;
    virtual void processB(const cv::Mat& src, cv::Mat& dst, int scaleTimes) override;
    virtual void processW(const cv::Mat& src, cv::Mat& dst, int scaleTimes) override;
    virtual void processF(const cv::Mat& src, cv::Mat& dst, int scaleTimes) override;
private:
    alignas(32) const static float kernelsL1[9 * 8];
    alignas(32) const static float kernels[8][9 * 8 * 8];
    alignas(32) const static float kernelsL10[4 * 8];
    alignas(32) const static float biasL1[8];
    alignas(32) const static float biases[8][8];
};

class Anime4KCPP::CPU::ACNetHDNL1 : public ACNetProcessor, public CNNProcessor
{
public:
    ACNetHDNL1() = default;
    virtual ~ACNetHDNL1() = default;
    virtual void processB(const cv::Mat& src, cv::Mat& dst, int scaleTimes) override;
    virtual void processW(const cv::Mat& src, cv::Mat& dst, int scaleTimes) override;
    virtual void processF(const cv::Mat& src, cv::Mat& dst, int scaleTimes) override;
private:
    alignas(32) const static float kernelsL1[9 * 8];
    alignas(32) const static float kernels[8][9 * 8 * 8];
    alignas(32) const static float kernelsL10[4 * 8];
    alignas(32) const static float biasL1[8];
    alignas(32) const static float biases[8][8];
};

class Anime4KCPP::CPU::ACNetHDNL2 : public ACNetProcessor, public CNNProcessor
{
public:
    ACNetHDNL2() = default;
    virtual ~ACNetHDNL2() = default;
    virtual void processB(const cv::Mat& src, cv::Mat& dst, int scaleTimes) override;
    virtual void processW(const cv::Mat& src, cv::Mat& dst, int scaleTimes) override;
    virtual void processF(const cv::Mat& src, cv::Mat& dst, int scaleTimes) override;
private:
    alignas(32) const static float kernelsL1[9 * 8];
    alignas(32) const static float kernels[8][9 * 8 * 8];
    alignas(32) const static float kernelsL10[4 * 8];
    alignas(32) const static float biasL1[8];
    alignas(32) const static float biases[8][8];
};

class Anime4KCPP::CPU::ACNetHDNL3 : public ACNetProcessor, public CNNProcessor
{
public:
    ACNetHDNL3() = default;
    virtual ~ACNetHDNL3() = default;
    virtual void processB(const cv::Mat& src, cv::Mat& dst, int scaleTimes) override;
    virtual void processW(const cv::Mat& src, cv::Mat& dst, int scaleTimes) override;
    virtual void processF(const cv::Mat& src, cv::Mat& dst, int scaleTimes) override;
private:
    alignas(32) const static float kernelsL1[9 * 8];
    alignas(32) const static float kernels[8][9 * 8 * 8];
    alignas(32) const static float kernelsL10[4 * 8];
    alignas(32) const static float biasL1[8];
    alignas(32) const static float biases[8][8];
};
#endif // ENABLE_OPENCV_DNN
