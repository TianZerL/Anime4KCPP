#pragma once

#ifdef ENABLE_NCNN

#include"NCNNACNet.hpp"
#include"ACManager.hpp"

namespace Anime4KCPP
{
    namespace NCNN
    {
        class DLL Manager;

        struct DLL GPUList;

        //return platforms, devices of each platform, all devices information
        DLL GPUList listGPUs();
    }

    namespace Processor
    {
        template<>
        struct GetManager<NCNN::ACNet> {
            using Manager = NCNN::Manager;
        };
    }
}

class Anime4KCPP::NCNN::Manager : public Anime4KCPP::Processor::Manager
{
public:
    Manager(std::string modelPath, std::string paramPath, int dID = -1, CNNType type = CNNType::Default, int threads = std::thread::hardware_concurrency());
    Manager(int dID = -1, CNNType type = CNNType::Default, int threads = std::thread::hardware_concurrency());
    void init() override;
    void release() override;
    bool isInitialized() override;
    bool isSupport() override;
private:
    bool testFlag;
    int dID, threads;
    int currACNetType;
    std::string modelPath, paramPath;
};

struct Anime4KCPP::NCNN::GPUList
{
    int devices;
    std::string message;

    GPUList(int devices, std::string message);
    std::string& operator()() noexcept;
};

#endif
