#pragma once

namespace Anime4KCPP
{
    namespace Processor
    {
        class Manager;
    }
}

class Anime4KCPP::Processor::Manager
{
public:
    virtual void init() = 0;
    virtual void release() = 0;
};
