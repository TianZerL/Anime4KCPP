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
    class AC_EXPORT ACInitializer;
}

class Anime4KCPP::ACInitializer
{
public:
    ACInitializer(const ACInitializer&) = delete;
    ACInitializer(ACInitializer&&) = delete;
    ~ACInitializer();

    template<typename Manager, typename... Types>
    void pushManager(Types&&... args);

    void init();
    void release(bool clearManagerList = false);
private:

    std::vector<std::unique_ptr<Processor::Manager>> managers;
};

template<typename Manager, typename... Types>
inline void Anime4KCPP::ACInitializer::pushManager(Types&&... args)
{
    managers.emplace_back(std::make_unique<Manager>(std::forward<Types>(args)...));
}
