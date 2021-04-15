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
    Manager(std::string modelPath, std::string paramPath, const int dID = -1, const CNNType type = CNNType::Default, const int threads = std::thread::hardware_concurrency());
    Manager(const int dID = -1, const CNNType type = CNNType::Default, const int threads = std::thread::hardware_concurrency());
    virtual void init() override;
    virtual void release() override;
    virtual bool isInitialized() override;
    virtual bool isSupport() override;
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

    GPUList(const int devices, std::string message);
    std::string& operator()() noexcept;
};

#endif
