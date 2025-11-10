#define RUAPU_IMPLEMENTATION
#include "ruapu.h"

#include "AC/Core/SIMD.hpp"

#define gISA (ac::core::simd::detail::ISA::instance())

namespace ac::core::simd::detail
{
    struct ISA
    {
    public:
        bool sse;
        bool sse2;
        bool avx;
        bool avx2;
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
            sse = ruapu_supports("sse");
            sse2 = ruapu_supports("sse2");
            avx = ruapu_supports("avx");
            avx2 = ruapu_supports("avx2");
            fma = ruapu_supports("fma");
            neon = ruapu_supports("neon");
        }
    };
}

bool ac::core::simd::supportSSE() noexcept
{
    return gISA.sse;
}
bool ac::core::simd::supportSSE2() noexcept
{
    return gISA.sse2;
}
bool ac::core::simd::supportAVX() noexcept
{
    return gISA.avx;
}
bool ac::core::simd::supportAVX2() noexcept
{
    return gISA.avx2;
}
bool ac::core::simd::supportFMA() noexcept
{
    return gISA.fma;
}
bool ac::core::simd::supportNEON() noexcept
{
    return gISA.neon;
}
