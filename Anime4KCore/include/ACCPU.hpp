#pragma once

#include"CPUAnime4K09.hpp"
#include"CPUACNet.hpp"
#include"ACManager.hpp"

namespace Anime4KCPP
{
    namespace CPU
    {
        class Manager;
    }

    namespace Processor
    {
        template<>
        struct GetManager<CPU::ACNet> {
            using Manager = CPU::Manager;
        };
        template<>
        struct GetManager<CPU::Anime4K09> {
            using Manager = CPU::Manager;
        };
    }
}

class Anime4KCPP::CPU::Manager : public Anime4KCPP::Processor::Manager
{
public:
    Manager();
    virtual void init() override;
    virtual void release() override;
    virtual bool isInitialized() override;
    virtual bool isSupport() override;
};

inline Anime4KCPP::CPU::Manager::Manager() {}

inline void Anime4KCPP::CPU::Manager::init() {}

inline void Anime4KCPP::CPU::Manager::release() {}

inline bool Anime4KCPP::CPU::Manager::isInitialized()
{
    return true;
}

inline bool Anime4KCPP::CPU::Manager::isSupport()
{
    return true;
}
