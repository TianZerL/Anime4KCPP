#ifndef AC_CORE_LIB_DISPATCH_HPP
#define AC_CORE_LIB_DISPATCH_HPP

namespace ac::core::cpu::dispatch
{
    // x86
    bool supportSSE() noexcept;
    bool supportAVX() noexcept;
    bool supportFMA() noexcept;
    // arm
    bool supportNEON() noexcept;
}

#endif
