#pragma once

#ifdef ENABLE_NCNN

#include"NCNNACNet.hpp"
#include"ACManager.hpp"

namespace Anime4KCPP
{
    namespace NCNN
    {
        class Manager;

        struct GPUList;

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

struct Anime4KCPP::NCNN::GPUList
{
    int devices;
    std::string message;

    GPUList(const int devices, std::string message);
    std::string& operator()() noexcept;
};

inline Anime4KCPP::NCNN::GPUList::GPUList(const int devices, std::string message)
    : devices(devices), message(std::move(message)) {}

inline std::string& Anime4KCPP::NCNN::GPUList::operator()() noexcept
{
    return message;
}

class Anime4KCPP::NCNN::Manager : public Anime4KCPP::Processor::Manager
{
public:
    Manager(std::string modelPath, std::string paramPath, const CNNType type = CNNType::Default, const int dID = -1, const int threads = std::thread::hardware_concurrency());
    Manager(const CNNType type = CNNType::Default, const int dID = -1, const int threads = std::thread::hardware_concurrency());
    virtual void init() override;
    virtual void release() override;
    virtual bool isInitialized() override;
    virtual bool isSupport() override;
private:
    bool testFlag;
    int dID, threads;
    ACNetType currACNetType;
    std::string modelPath, paramPath;
};

inline Anime4KCPP::NCNN::Manager::Manager(std::string modelPath, std::string paramPath, const CNNType type, const int dID, const int threads)
    : modelPath(std::move(modelPath)), paramPath(std::move(paramPath)), dID(dID), threads(threads), testFlag(true)
{
    switch (type)
    {
    case CNNType::ACNetHDNL0:
        currACNetType = ACNetType::HDNL0;
        break;
    case CNNType::ACNetHDNL1:
        currACNetType = ACNetType::HDNL1;
        break;
    case CNNType::ACNetHDNL2:      
        currACNetType = ACNetType::HDNL2;
        break;
    case CNNType::ACNetHDNL3:     
        currACNetType = ACNetType::HDNL3;
        break;
    case CNNType::Default:
    default:
        currACNetType = ACNetType::TotalTypeCount;
        break;
    }
}

inline Anime4KCPP::NCNN::Manager::Manager(const CNNType type, const int dID, const int threads)
    : Manager(std::move(std::string{}), std::move(std::string{}), type, dID, threads)
{
    testFlag = false;
}

inline void Anime4KCPP::NCNN::Manager::init()
{
    if (!Anime4KCPP::NCNN::ACNet::isInitialized())
    {
        if (!testFlag)
        {
            if (currACNetType == ACNetType::TotalTypeCount)
                Anime4KCPP::NCNN::ACNet::init(dID, threads);
            else
                Anime4KCPP::NCNN::ACNet::init(currACNetType, dID, threads);
        }
        else
            Anime4KCPP::NCNN::ACNet::init(modelPath, paramPath, currACNetType, dID, threads);
    }

}

inline void Anime4KCPP::NCNN::Manager::release()
{
    if (Anime4KCPP::NCNN::ACNet::isInitialized())
        Anime4KCPP::NCNN::ACNet::release();
}

inline bool Anime4KCPP::NCNN::Manager::isInitialized()
{
    return Anime4KCPP::NCNN::ACNet::isInitialized();
}

inline bool Anime4KCPP::NCNN::Manager::isSupport()
{
    return true;
}

#endif
