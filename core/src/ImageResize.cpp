#include <cassert>
#include <cmath>

#define STB_IMAGE_RESIZE2_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include <stb_image_resize2.h>

#include "AC/Core/SIMD.hpp"
#include "AC/Core/Image.hpp"

#ifdef AC_CORE_WITH_SSE2
extern const decltype(&stbir_resize_extended) stbir_resize_extended_sse2;
#endif
#ifdef AC_CORE_WITH_AVX
extern const decltype(&stbir_resize_extended) stbir_resize_extended_avx;
#endif
#ifdef AC_CORE_WITH_AVX2
extern const decltype(&stbir_resize_extended) stbir_resize_extended_avx2;
#endif
#ifdef AC_CORE_WITH_NEON
extern const decltype(&stbir_resize_extended) stbir_resize_extended_neon;
#endif
#ifdef AC_CORE_WITH_WASM_SIMD128
extern const decltype(&stbir_resize_extended) stbir_resize_extended_wasm;
#endif

namespace ac::core::detail
{
   const auto stbir_resize_extended_auto = []() {
        // x86
#   ifdef AC_CORE_WITH_AVX2
        if (ac::core::simd::supportAVX2()) return stbir_resize_extended_avx2;
#   endif
#   ifdef AC_CORE_WITH_AVX
        if (ac::core::simd::supportAVX()) return stbir_resize_extended_avx;
#   endif
#   ifdef AC_CORE_WITH_SSE2
        if (ac::core::simd::supportSSE2()) return stbir_resize_extended_sse2;
#   endif
        // arm
#   ifdef AC_CORE_WITH_NEON
        if (ac::core::simd::supportNEON()) return stbir_resize_extended_neon;
#   endif
        // wasm
#   ifdef AC_CORE_WITH_WASM_SIMD128
        return stbir_resize_extended_wasm;
#   endif
        // generic
        return stbir_resize_extended;
    }();

    static inline float sinc(const float x) noexcept
    {
        constexpr float pi = 3.14159265358979f;
        return x == 0.0f ? 1.0f : std::sin(x * pi) / (x * pi);
    }
    static inline float poly3(const float x, const float c0, const float c1, const float c2, const float c3) noexcept
    {
        return c0 + x * (c1 + x * (c2 + x * c3));
    }

    template<int bn, int bd, int cn, int cd>
    static inline float bicubic(const float v) noexcept
    {
        constexpr float b = static_cast<float>(bn) / static_cast<float>(bd);
        constexpr float c = static_cast<float>(cn) / static_cast<float>(cd);
        constexpr float p0 = (6.0f - 2.0f * b) / 6.0f;
        constexpr float p1 = 0.0f;
        constexpr float p2 = (-18.0f + 12.0f * b + 6.0f * c) / 6.0f;
        constexpr float p3 = (12.0f - 9.0f * b - 6.0f * c) / 6.0f;
        constexpr float q0 = (8.0f * b + 24.0f * c) / 6.0f;
        constexpr float q1 = (-12.0f * b - 48.0f * c) / 6.0f;
        constexpr float q2 = (6.0f * b + 30.0f * c) / 6.0f;
        constexpr float q3 = (-b - 6.0f * c) / 6.0f;

        auto x = std::abs(v);

        if (x < 1.0f)
            return poly3(x, p0, p1, p2, p3);
        else if (x < 2.0f)
            return poly3(x, q0, q1, q2, q3);
        else
            return 0.0f;
    }
    template<int taps>
    static inline float lanczos(const float v) noexcept
    {
        auto x = std::abs(v);

        return x < taps ? sinc(x) * sinc(x / static_cast<float>(taps)) : 0.0f;
    }
    static inline float spine16(const float v) noexcept
    {
        auto x = std::abs(v);

        if (x < 1.0f)
            return poly3(x, 1.0f, -1.0f / 5.0f, -9.0f / 5.0f, 1.0f);
        else if (x < 2.0f)
            return poly3(x - 1.0f, 0.0f, -7.0f / 15.0f, 4.0f / 5.0f, -1.0f / 3.0f);
        else
            return 0.0f;
    }
    static inline float spine36(const float v) noexcept
    {
        auto x = std::abs(v);

        if (x < 1.0f)
            return poly3(x, 1.0f, -3.0f / 209.0f, -453.0f / 209.0f, 13.0f / 11.0f);
        else if (x < 2.0f)
            return poly3(x - 1.0f, 0.0f, -156.0f / 209.0f, 270.0f / 209.0f, -6.0f / 11.0f);
        else if (x < 3.0f)
            return poly3(x - 2.0f, 0.0f, 26.0f / 209.0f, -45.0f / 209.0f, 1.0f / 11.0f);
        else
            return 0.0f;
    }
    static inline float spine64(const float v) noexcept
    {
        auto x = std::abs(v);

        if (x < 1.0f)
            return poly3(x, 1.0f, -3.0f / 2911.0f, -6387.0f / 2911.0f, 49.0f / 41.0f);
        else if (x < 2.0f)
            return poly3(x - 1.0f, 0.0f, -2328.0f / 2911.0f, 4032.0f / 2911.0f, -24.0f / 41.0f);
        else if (x < 3.0f)
            return poly3(x - 2.0f, 0.0f, 582.0f / 2911.0f, -1008.0f / 2911.0f, 6.0f / 41.0f);
        else if (x < 4.0f)
            return poly3(x - 3.0f, 0.0f, -97.0f / 2911.0f, 168.0f / 2911.0f, -1.0f / 41.0f);
        else
            return 0.0f;
    }
    static inline float bilinear(const float v) noexcept
    {
        auto x = std::abs(v);

        if (x <= 1.0f)
            return 1.0f - x;
        else
            return 0.0f;
    }

    static inline void resize(const Image& src, Image& dst, const double fx, const double fy, const int mode) noexcept
    {
        if (src.empty()) return;

        if (fx > 0.0 && fy > 0.0)
        {
            if (fx == 1.0 && fy == 1.0)
            {
                dst = src;
                return;
            }

            auto dstW = static_cast<int>(src.width() * fx);
            auto dstH = static_cast<int>(src.height() * fy);

            if ((dst.width() != dstW) || (dst.height() != dstH) || (dst.channels() != src.channels()) || (dst.type() != src.type()))
                dst.create(dstW, dstH, src.channels(), src.type());
        }
        else
        {
            if (dst.empty()) return;
            if (dst.width() == src.width() && dst.height() == src.height())
            {
                dst = src;
                return;
            }

            if ((dst.channels() != src.channels()) || (dst.type() != src.type()))
                dst.create(dst.width(), dst.height(), src.channels(), src.type());
        }

        stbir__kernel_callback* filter{};
        stbir__support_callback* support{};
        STBIR_RESIZE ctx{};

        stbir_resize_init(&ctx,
            src.ptr(), src.width(), src.height(), src.stride(),
            dst.ptr(), dst.width(), dst.height(), dst.stride(),
            [&]() -> stbir_pixel_layout {
                switch (src.channels())
                {
                case 1: return STBIR_1CHANNEL;
                case 2: return STBIR_2CHANNEL;
                case 3: return STBIR_RGB;
                case 4: return STBIR_4CHANNEL;
                default: return assert(src.channels() == 1 || src.channels() == 2 || src.channels() == 3 || src.channels() == 4), STBIR_1CHANNEL;
                }
            }(),
            [&]() -> stbir_datatype {
                switch (src.type())
                {
                case Image::UInt8: return STBIR_TYPE_UINT8;
                case Image::UInt16: return STBIR_TYPE_UINT16;
                case Image::Float32: return STBIR_TYPE_FLOAT;
                default: return assert(src.type() == Image::UInt8 || src.type() == Image::UInt16 || src.type() == Image::Float32), STBIR_TYPE_UINT8;
                }
            }()
           );

        switch (mode)
        {
        case RESIZE_POINT:
            filter = [](float x, float, void*) -> float { return 1.0f; };
            support = [](float, void*) -> float { return 0.5f; };
            break;
        case RESIZE_CATMULL_ROM: // b = 0, c = 1/2
            filter = [](float x, float, void*) -> float { return bicubic<0, 1, 1, 2>(x); };
            support = [](float, void*) -> float { return 2.0f; };
            break;
        case RESIZE_MITCHELL_NETRAVALI: // b = 1/3, c = 1/3
            filter = [](float x, float, void*) -> float { return bicubic<1, 3, 1, 3>(x); };
            support = [](float, void*) -> float { return 2.0f; };
            break;
        case RESIZE_BICUBIC_0_60: // b = 0, c = 3/5
            filter = [](float x, float, void*) -> float { return bicubic<0, 1, 3, 5>(x); };
            support = [](float, void*) -> float { return 2.0f; };
            break;
        case RESIZE_BICUBIC_0_75: // b = 0, c = 3/4
            filter = [](float x, float, void*) -> float { return bicubic<0, 1, 3, 4>(x); };
            support = [](float, void*) -> float { return 2.0f; };
            break;
        case RESIZE_BICUBIC_0_100: // b = 0, c = 1
            filter = [](float x, float, void*) -> float { return bicubic<0, 1, 1, 1>(x); };
            support = [](float, void*) -> float { return 2.0f; };
            break;
        case RESIZE_BICUBIC_20_50: // b = 0.2, c = 0.5
            filter = [](float x, float, void*) -> float { return bicubic<1, 5, 1, 2>(x); };
            support = [](float, void*) -> float { return 2.0f; };
            break;
        case RESIZE_SOFTCUBIC50: // b = 1/2, c = 1/2
            filter = [](float x, float, void*) -> float { return bicubic<1, 2, 1, 2>(x); };
            support = [](float, void*) -> float { return 2.0f; };
            break;
        case RESIZE_SOFTCUBIC75: // b = 3/4, c = 1/4
            filter = [](float x, float, void*) -> float { return bicubic<3, 4, 1, 4>(x); };
            support = [](float, void*) -> float { return 2.0f; };
            break;
        case RESIZE_SOFTCUBIC100: // b = 1, c = 0
            filter = [](float x, float, void*) -> float { return bicubic<1, 1, 0, 1>(x); };
            support = [](float, void*) -> float { return 2.0f; };
            break;
        case RESIZE_LANCZOS2:
            filter = [](float x, float, void*) -> float { return lanczos<2>(x); };
            support = [](float, void*) -> float { return 2.0f; };
            break;
        case RESIZE_LANCZOS3:
            filter = [](float x, float, void*) -> float { return lanczos<3>(x); };
            support = [](float, void*) -> float { return 3.0f; };
            break;
        case RESIZE_LANCZOS4:
            filter = [](float x, float, void*) -> float { return lanczos<4>(x); };
            support = [](float, void*) -> float { return 4.0f; };
            break;
        case RESIZE_SPLINE16:
            filter = [](float x, float, void*) -> float { return spine16(x); };
            support = [](float, void*) -> float { return 2.0f; };
            break;
        case RESIZE_SPLINE36:
            filter = [](float x, float, void*) -> float { return spine36(x); };
            support = [](float, void*) -> float { return 3.0f; };
            break;
        case RESIZE_SPLINE64:
            filter = [](float x, float, void*) -> float { return spine64(x); };
            support = [](float, void*) -> float { return 4.0f; };
            break;
        case RESIZE_BILINEAR:
        default:
            filter = [](float x, float, void*) -> float { return bilinear(x); };
            support = [](float, void*) -> float { return 1.0f; };
            break;
        }

        stbir_set_edgemodes(&ctx, STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP);
        stbir_set_filter_callbacks(&ctx, filter, support, filter, support);

        stbir_resize_extended_auto(&ctx);
    }
}

void ac::core::resize(const ac::core::Image& src, ac::core::Image& dst, const double fx, const double fy, const int mode) noexcept
{
    if (src == dst)
    {
        Image tmp{};
        detail::resize(src, tmp, fx, fy, mode);
        if (!tmp.empty()) dst = tmp;
    }
    else detail::resize(src, dst, fx, fy, mode);
}
ac::core::Image ac::core::resize(const ac::core::Image& src, const double fx, const double fy, const int mode) noexcept
{
    if (fx <= 0.0 || fy <= 0.0) return src;

    Image dst{};
    detail::resize(src, dst, fx, fy, mode);
    return dst;
}
