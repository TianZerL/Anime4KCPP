#ifndef AC_CORE_INTERNAL_DATA_TYPE_HPP
#define AC_CORE_INTERNAL_DATA_TYPE_HPP

#include <cstdint>

#ifdef AC_CORE_WITH_HALF
#   include <half.hpp>
#endif

namespace ac::core
{
    struct DataType
    {
        using UInt8 = std::uint8_t;
        using UInt16 = std::uint16_t;
        using Float32 = float;
#   if defined(AC_CORE_HAVE_GCC_FLOAT16)
        using Float16 = _Float16;
#   elif defined(AC_CORE_HAVE_ARM_FP16)
        using Float16 = __fp16;
#   else
        using Float16 = half_float::half;
#   endif
    };
}

#endif
