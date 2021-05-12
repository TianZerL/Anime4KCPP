#pragma once

#include"ACCPU.hpp"

#ifdef ENABLE_OPENCL
#include"ACOpenCL.hpp"
#endif

#ifdef ENABLE_CUDA
#include"ACCuda.hpp"
#endif

#ifdef ENABLE_NCNN
#include"ACNCNN.hpp"
#endif

#include"ACManager.hpp"

namespace Anime4KCPP
{
    class DLL ACCreator;
}

//Anime4KCPP Processor Factory
class Anime4KCPP::ACCreator
{
public:
    using ManagerSP = std::shared_ptr<Processor::Manager>;
    using ManagerSPList = std::initializer_list<std::shared_ptr<Processor::Manager>>;
    using ManagerSPVector = std::vector<std::shared_ptr<Processor::Manager>>;

    ACCreator() = default;
    ACCreator(const ManagerSP& manager, const bool initNow = true);
    ACCreator(ManagerSPList&& managerList, const bool initNow = true);
    ACCreator(ManagerSPVector managerList, const bool initNow = true);
    ~ACCreator();
    static std::unique_ptr<AC> createUP(const Parameters& parameters, const Processor::Type type);
    static AC* create(const Parameters& parameters, const Processor::Type type);
    static void release(AC* ac) noexcept;

    template<typename Manager, typename... Types>
    void pushManager(Types&&... args);

    void init();
    void deinit(bool clearManager = false);
private:
    ManagerSPVector managers;
};

template<typename Manager, typename... Types>
inline void Anime4KCPP::ACCreator::pushManager(Types&&... args)
{
    managers.emplace_back(std::make_shared<Manager>(std::forward<Types>(args)...));
}
