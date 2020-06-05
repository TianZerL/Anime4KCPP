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
    class DLL Anime4KCPUCNN;
}

class Anime4KCPP::Anime4KCPUCNN :public Anime4K
{
public:
    Anime4KCPUCNN(const Parameters& parameters = Parameters());
    virtual ~Anime4KCPUCNN() = default;
    virtual void process() override;

    void conv1To8(cv::InputArray img, const std::vector<cv::Mat>& kernels, const cv::Mat& biases, std::pair<cv::Mat, cv::Mat>& tmpMats);
    void conv8To8(const std::vector<std::vector<cv::Mat>>& kernels, const cv::Mat& biases, std::pair<cv::Mat, cv::Mat>& tmpMats);
    void convTranspose8To1(cv::Mat& img, const std::vector<cv::Mat>& kernels, std::pair<cv::Mat, cv::Mat>& tmpMats);

private:
    void changEachPixel1To8(cv::InputArray _src, const std::function<void(int, int, Chan, Chan, LineC)>&& callBack, std::pair<cv::Mat, cv::Mat>& tmpMats);
    void changEachPixel8To8(const std::function<void(int, int, Chan, Chan, LineF, LineF)>&& callBack, std::pair<cv::Mat, cv::Mat>& tmpMats);
    void changEachPixel8To1(cv::Mat& img, const std::function<void(int, int, PIXEL, LineF, LineF)>&& callBack, std::pair<cv::Mat, cv::Mat>& tmpMats);

private:
    const static std::vector<cv::Mat> kernelsL1;
    const static std::vector<std::vector<cv::Mat>> kernelsL2;
    const static std::vector<std::vector<cv::Mat>> kernelsL3;
    const static std::vector<std::vector<cv::Mat>> kernelsL4;
    const static std::vector<std::vector<cv::Mat>> kernelsL5;
    const static std::vector<std::vector<cv::Mat>> kernelsL6;
    const static std::vector<std::vector<cv::Mat>> kernelsL7;
    const static std::vector<std::vector<cv::Mat>> kernelsL8;
    const static std::vector<std::vector<cv::Mat>> kernelsL9;
    const static std::vector<cv::Mat> kernelsL10;
    const static cv::Mat biasesL1;
    const static cv::Mat biasesL2;
    const static cv::Mat biasesL3;
    const static cv::Mat biasesL4;
    const static cv::Mat biasesL5;
    const static cv::Mat biasesL6;
    const static cv::Mat biasesL7;
    const static cv::Mat biasesL8;
    const static cv::Mat biasesL9;
};

