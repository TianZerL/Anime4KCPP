#pragma once

namespace Anime4KCPP
{
    namespace Processor
    {
        class Manager;

        template<typename T>
        struct GetManager {};
    }
}

class Anime4KCPP::Processor::Manager
{
public:
    virtual void init() = 0;
    virtual void release() = 0;
    virtual bool isInitialized() = 0;
    virtual bool isSupport() = 0;
};
