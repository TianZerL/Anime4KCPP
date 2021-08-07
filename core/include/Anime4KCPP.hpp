#pragma once

#ifdef ENABLE_VIDEO
#include"VideoProcessor.hpp"
#include"VideoCodec.hpp"
#endif

#include"ACCreator.hpp"

#define ANIME4KCPP_CORE_VERSION "2.6.0"

namespace Anime4KCPP
{
    template<typename T, int W, int H, typename ...Types>
    double benchmark(Types&&... args);
}

template<typename T, int W, int H, typename ...Types>
inline double Anime4KCPP::benchmark(Types && ...args)
{
    Anime4KCPP::ACCreator creator;

    creator.pushManager<typename Processor::GetManager<T>::Manager>(std::forward<Types>(args)...);
    try
    {
        creator.init();
    }
    catch (const std::exception&)
    {
        return 0.0;
    }

    cv::Mat testImg = cv::Mat::zeros(cv::Size(W, H), CV_8UC1);
    cv::randu(testImg, cv::Scalar::all(0.0f), cv::Scalar::all(1.0f));

    double avg = 0.0;
    std::chrono::steady_clock::time_point s;
    std::chrono::steady_clock::time_point e;

    T ac{};
    ac.loadImage(testImg, testImg, testImg); // YUV
    ac.process();

    for (int i = 0; i < 3; i++)
    {
        ac.loadImage(testImg, testImg, testImg);
        s = std::chrono::steady_clock::now();
        ac.process();
        e = std::chrono::steady_clock::now();
        avg += 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
    }

    return avg / 3.0;
}
