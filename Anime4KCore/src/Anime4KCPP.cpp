#define DLL

#include "Anime4KCPP.hpp"

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
    deinit();
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

std::pair<double, double> Anime4KCPP::benchmark(const unsigned int pID, const unsigned int dID)
{
    std::pair<double, double> ret;

    OpenCL::GPUInfo checkGPUResult = OpenCL::checkGPUSupport(pID, dID);

    Anime4KCPP::Parameters parameters;
    Anime4KCPP::ACCreator creator;
    if (checkGPUResult)
    {
        creator.pushManager<OpenCL::Manager<OpenCL::ACNet>>(pID, dID, Anime4KCPP::CNNType::ACNetHDNL0);
        creator.init();
    }
    cv::Mat testImg = cv::Mat::zeros(cv::Size(1920, 1080), CV_8UC1);
    cv::randu(testImg, cv::Scalar::all(0), cv::Scalar::all(255));

    std::unique_ptr<Anime4KCPP::AC> acCPU = creator.createUP(parameters, Anime4KCPP::Processor::Type::CPU_ACNet);

    acCPU->loadImage(testImg, testImg, testImg); // YUV
    std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();
    acCPU->process();
    std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
    ret.first = 10000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();

    if (checkGPUResult)
    {
        std::unique_ptr<Anime4KCPP::AC> acGPU = creator.createUP(parameters, Anime4KCPP::Processor::Type::OpenCL_ACNet);

        acGPU->loadImage(testImg, testImg, testImg);
        s = std::chrono::steady_clock::now();
        acGPU->process();
        e = std::chrono::steady_clock::now();
        ret.second = 10000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
    }
    else
        ret.second = 0.0;

    return ret;
}
