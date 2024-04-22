#ifndef AC_CORE_LIB_DISPATCH_HPP
#define AC_CORE_LIB_DISPATCH_HPP

namespace ac::core::cpu::dispatch
{
    bool supportSSE() noexcept;
    bool supportAVX() noexcept;
    bool supportFMA() noexcept;
}

#endif
