#define RUAPU_IMPLEMENTATION
#include "ruapu.h"

#include "AC/Core/Dispatch.hpp"

#define gISA (ac::core::cpu::dispatch::detail::ISA::instance())

namespace ac::core::cpu::dispatch::detail
{
    struct ISA
    {
    public:
        bool sse;
        bool avx;
        bool fma;
        bool neon;
    public:
        static const ISA& instance() noexcept
        {
            static const ISA isa{};
            return isa;
        }
    private:
        ISA() noexcept
        {
            ruapu_init();
            sse = ruapu_supports("sse3");
            avx = ruapu_supports("avx");
            fma = ruapu_supports("fma");
            neon = ruapu_supports("neon");
        }
    };
}

bool ac::core::cpu::dispatch::supportSSE() noexcept
{
    return gISA.sse;
}
bool ac::core::cpu::dispatch::supportAVX() noexcept
{
    return gISA.avx;
}
bool ac::core::cpu::dispatch::supportFMA() noexcept
{
    return gISA.fma;
}
bool ac::core::cpu::dispatch::supportNEON() noexcept
{
    return gISA.neon;
}
