#include"ACInitializer.hpp"

Anime4KCPP::ACInitializer::~ACInitializer()
{
    release(true);
}

void Anime4KCPP::ACInitializer::init()
{
    for (auto& manager : managers)
    {
        if (!manager->isSupport())
            throw ACException<ExceptionType::RunTimeError>("Try initializing unsupported processor");
        if (!manager->isInitialized())
            manager->init();
    }
}

void Anime4KCPP::ACInitializer::release(bool clearManagerList)
{
    for (auto& manager : managers)
    {
        if (manager->isInitialized())
            manager->release();
    }
    if (clearManagerList)
        managers.clear();
}
