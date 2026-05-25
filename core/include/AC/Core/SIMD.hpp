#ifndef AC_CORE_DISPATCH_HPP
#define AC_CORE_DISPATCH_HPP

namespace ac::core::simd
{
    // x86
    bool supportSSE() noexcept;
    bool supportSSE2() noexcept;
    bool supportAVX() noexcept;
    bool supportAVX2() noexcept;
    bool supportAVX512() noexcept;
    bool supportFMA() noexcept;
    // arm
    bool supportNEON() noexcept;
    // risc-v
    bool supportRVV() noexcept;
}

#endif
