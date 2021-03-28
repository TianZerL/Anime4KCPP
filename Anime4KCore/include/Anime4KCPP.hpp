#pragma once

#include"ACCPU.hpp"
#include"ACOpenCL.hpp"

#ifdef ENABLE_CUDA
#include"ACCuda.hpp"
#endif

#ifdef ENABLE_NCNN
#include"ACNCNN.hpp"
#endif

#include"ACManager.hpp"

#define ANIME4KCPP_CORE_VERSION "2.6.0"

namespace Anime4KCPP
{
    class DLL ACCreator;

    template<typename T, typename ...Types>
    double benchmark(Types&&... args);
}

//Anime4KCPP Processor Factory
class Anime4KCPP::ACCreator
{
public:
    using ManagerSP = std::shared_ptr<Processor::Manager>;
    using ManagerSPList = std::initializer_list<std::shared_ptr<Processor::Manager>>;
    using ManagerSPVector = std::vector<std::shared_ptr<Processor::Manager>>;

    ACCreator() = default;
    ACCreator(const ManagerSP& manager, const bool initNow = true);
    ACCreator(ManagerSPList&& managerList, const bool initNow = true);
    ACCreator(ManagerSPVector managerList, const bool initNow = true);
    ~ACCreator();
    static std::unique_ptr<AC> createUP(const Parameters& parameters, const Processor::Type type);
    static AC* create(const Parameters& parameters, const Processor::Type type);
    static void release(AC* ac) noexcept;

    template<typename Manager, typename... Types>
    void pushManager(Types&&... args);

    void init();
    void deinit(bool clearManager = false);
private:
    ManagerSPVector managers;
};

template<typename Manager, typename... Types>
inline void Anime4KCPP::ACCreator::pushManager(Types&&... args)
{
    managers.emplace_back(std::make_shared<Manager>(std::forward<Types>(args)...));
}

template<typename T, typename ...Types>
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

    cv::Mat testImg = cv::Mat::zeros(cv::Size(1920, 1080), CV_32FC1);
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
        avg += 10000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
    }

    return avg / 3.0;
}
