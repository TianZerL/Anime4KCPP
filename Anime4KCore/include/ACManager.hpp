#pragma once

namespace Anime4KCPP
{
    namespace Processor
    {
        class Manager;

        template<typename T>
        struct GetManager {
            using Manager = std::nullptr_t;
        };
    }
}

class Anime4KCPP::Processor::Manager
{
public:
    virtual void init() = 0;
    virtual void release() = 0;
    virtual bool isInitialized() = 0;
};
