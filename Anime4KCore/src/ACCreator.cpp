#define DLL

#include"ACCreator.hpp"

Anime4KCPP::ACCreator::ACCreator(const ManagerSP& manager, const bool initNow)
{
    managers.emplace_back(manager);
    if (initNow)
        init();
}

Anime4KCPP::ACCreator::ACCreator(ManagerSPList&& managerList, const bool initNow)
    : managers(managerList)
{
    if (initNow)
        init();
}

Anime4KCPP::ACCreator::ACCreator(ManagerSPVector managerList, const bool initNow)
    : managers(std::move(managerList))
{
    if (initNow)
        init();
}

Anime4KCPP::ACCreator::~ACCreator()
{
    deinit(true);
}

#define AC_CASE_UP_ITEM
#include"ACRegister.hpp"
#undef AC_CASE_UP_ITEM
std::unique_ptr<Anime4KCPP::AC> Anime4KCPP::ACCreator::createUP(
    const Parameters& parameters, const Processor::Type type
)
{
    switch (type)
    {
        PROCESSOR_CASE_UP
    }
    return nullptr;
}

#define AC_CASE_ITEM
#include"ACRegister.hpp"
#undef AC_CASE_ITEM
Anime4KCPP::AC* Anime4KCPP::ACCreator::create(
    const Parameters& parameters, const Processor::Type type
)
{
    switch (type)
    {
        PROCESSOR_CASE
    }
    return nullptr;
}

void Anime4KCPP::ACCreator::release(AC* ac) noexcept
{
    if (ac != nullptr)
    {
        delete ac;
        ac = nullptr;
    }
}

void Anime4KCPP::ACCreator::init()
{
    for (auto& manager : managers)
    {
        if (!manager->isSupport())
            throw ACException<ExceptionType::RunTimeError>("Try initializing unsupported processor");
        if (!manager->isInitialized())
            manager->init();
    }
}

void Anime4KCPP::ACCreator::deinit(bool clearManager)
{
    for (auto& manager : managers)
    {
        if (manager->isInitialized())
            manager->release();
    }
    if (clearManager)
        managers.clear();
}
