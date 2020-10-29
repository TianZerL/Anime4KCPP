#pragma once

#include"ACCPU.hpp"
#include"ACOpenCL.hpp"
#include"ACManager.hpp"

#define ANIME4KCPP_CORE_VERSION "2.4.1"

namespace Anime4KCPP
{
    class DLL ACCreator;
    DLL std::pair<double, double> benchmark(const unsigned int pID = 0, const unsigned int dID = 0);
}

//Anime4KCPP Processor Factory
class Anime4KCPP::ACCreator
{
public:
    ACCreator() = default;
    ACCreator(std::initializer_list<std::shared_ptr<Processor::Manager>> managerList, bool initNow = false);
    ACCreator(std::shared_ptr<Processor::Manager> manager, bool initNow = false);
    ACCreator(bool initGPU, bool initCNN, unsigned int platformID = 0, unsigned int deviceID = 0, const CNNType type = CNNType::Default);
    ~ACCreator();
    std::shared_ptr<AC> createSP(const Parameters& parameters, const Processor::Type type);
    AC* create(const Parameters& parameters, const Processor::Type type);
    void release(AC*& ac) noexcept;
    void init();
private:
    std::vector<std::shared_ptr<Processor::Manager>> managers;
};
