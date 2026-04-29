#include <immintrin.h>

#include "AC/Core/SIMD.hpp"
#include "AC/Core/Image.hpp"
#include "AC/Core/Util.hpp"

#include "AC/Core/Internal/Processor/CPU/Common.hpp"

namespace ac::core::cpu
{
    template <bool fma>
    struct OpImplAVX
    {
    private:
        static inline float hsum(const __m256& v) noexcept
        {
            __m128 v128 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 0x01));
            __m128 v64 = _mm_add_ps(v128, _mm_movehl_ps(v128, v128));
            __m128 v32 = _mm_add_ss(v64, _mm_movehdup_ps(v64));
            return _mm_cvtss_f32(v32);
        }

        template <int cin, int scount, int cpos>
        static void convKernel(const int sgroupIdx, const float** const rptr, __m256* const s, float* const out, const float* const kernels) noexcept
        {
            constexpr int vstep = 8;
            constexpr int count = cin / vstep;
            constexpr int remain = cin % vstep;

            for (int n = 0; n < scount; n++) s[n] = _mm256_setzero_ps();
            for (int p = 0; p < cpos; p++)
            {
                for (int idx = 0; idx < count; idx++)
                {
                    __m256 r = _mm256_loadu_ps(rptr[p] + idx * vstep);
                    for (int n = 0; n < scount; n++)
                    {
                        __m256 k = _mm256_loadu_ps(kernels + (sgroupIdx * scount + n) * cin * cpos + cin * p + idx * vstep);
#                   ifdef AC_CORE_WITH_FMA
                        if constexpr (fma)
                            s[n] = _mm256_fmadd_ps(r, k, s[n]);
                        else
#                   endif
                            s[n] = _mm256_add_ps(_mm256_mul_ps(r, k), s[n]);
                    }
                }
                if constexpr (remain)
                    for (int c = count * vstep; c < cin; c++)
                        for (int n = 0; n < scount; n++)
                            out[sgroupIdx * scount + n] += rptr[p][c] * kernels[(sgroupIdx * scount + n) * cin * cpos + cin * p + c];
            }
            for (int n = 0; n < scount; n++) out[sgroupIdx * scount + n] += hsum(s[n]);
        }

    public:
        template <int vsize>
        static float dot(const float* const v1, const float* const v2) noexcept
        {
            constexpr int vstep = 8;
            constexpr int count = vsize / vstep;
            constexpr int remain = vsize % vstep;

            float sum = 0.0f;

            __m256 s = _mm256_setzero_ps();
            for (int idx = 0; idx < count; idx++)
            {
                __m256 r1 = _mm256_loadu_ps(v1 + idx * vstep);
                __m256 r2 = _mm256_loadu_ps(v2 + idx * vstep);

#           ifdef AC_CORE_WITH_FMA
                if constexpr (fma)
                    s = _mm256_fmadd_ps(r1, r2, s);
                else
#           endif
                    s = _mm256_add_ps(_mm256_mul_ps(r1, r2), s);
            }
            sum += hsum(s);

            if constexpr (remain)
                for (int i = count * vstep; i < vsize; i++)
                    sum += v1[i] * v2[i];

            return sum;
        }

        template <int cout, int cpos>
        static void conv_cin1(const float* const rptr, float* const out, const float* const kernels, const float* const biases) noexcept
        {
            constexpr int vstep = 8;
            constexpr int count = cpos / vstep;
            constexpr int remain = cpos % vstep;

            __m256 r[count];
            for (int idx = 0; idx < count; idx++) r[idx] = _mm256_loadu_ps(rptr + idx * vstep);

            for (int n = 0; n < cout; n++)
            {
                __m256 s = _mm256_setzero_ps();
                auto kptr = kernels + n * cpos;
                for (int idx = 0; idx < count; idx++)
                {
                    __m256 k = _mm256_loadu_ps(kptr + idx * vstep);

#               ifdef AC_CORE_WITH_FMA
                    if constexpr (fma)
                        s = _mm256_fmadd_ps(r[idx], k, s);
                    else
#               endif
                        s = _mm256_add_ps(_mm256_mul_ps(r[idx], k), s);
                }
                auto sum = hsum(s);

                for (int i = 0; i < remain; i++) sum += rptr[count * vstep + i] * kptr[count * vstep + i];

                out[n] = sum + biases[n];
            }
        }

        template <int cin, int cout, int cpos>
        static void conv(const float** const rptr, float* const out, const float* const kernels, const float* const biases) noexcept
        {
            constexpr int scount = 8;
            constexpr int sgroup = cout / scount;
            constexpr int sremian = cout % scount;

            std::memcpy(out, biases, sizeof(float) * cout);

            __m256 s[sgroup > 0 ? scount : sremian];

            if constexpr (sgroup)
                for (int i = 0; i < sgroup; i++)
                    convKernel<cin, scount, cpos>(i, rptr, s, out, kernels);
            if constexpr (sremian)
                convKernel<cin, sremian, cpos>(sgroup, rptr, s, out, kernels);
        }
    };

    void conv3x3_1to8_relu_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplAVX<false>, std::uint8_t, 8>(src, dst, kernels, biases, ReLU{});
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplAVX<false>, std::uint16_t, 8>(src, dst, kernels, biases, ReLU{});
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplAVX<false>, float, 8>(src, dst, kernels, biases, ReLU{});
            break;
        }
    }
    void conv3x3_8to8_relu_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_float<OpImplAVX<true>, 8, 8>(src, dst, kernels, biases, ReLU{});
        else
#   endif
            conv3x3_float<OpImplAVX<false>, 8, 8>(src, dst, kernels, biases, ReLU{});
    }
    void deconv2x2_8to1_avx(const Image& src, Image& dst, const float* kernels)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            deconv2x2<OpImplAVX<false>, std::uint8_t, 8, 1>(src, dst, kernels);
            break;
        case Image::UInt16:
            deconv2x2<OpImplAVX<false>, std::uint16_t, 8, 1>(src, dst, kernels);
            break;
        case Image::Float32:
            deconv2x2<OpImplAVX<false>, float, 8, 1>(src, dst, kernels);
            break;
        }
    }

    void conv3x3_1to8_prelu_avx(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplAVX<false>, std::uint8_t, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplAVX<false>, std::uint16_t, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplAVX<false>, float, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        }
    }

    void conv3x3_1to8_identity_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplAVX<false>, std::uint8_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplAVX<false>, std::uint16_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplAVX<false>, float, 8>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_8to8_prelu_avx(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_float<OpImplAVX<true>, 8, 8>(src, dst, kernels, biases, PReLU{ alphas });
        else
#   endif
            conv3x3_float<OpImplAVX<false>, 8, 8>(src, dst, kernels, biases, PReLU{ alphas });
    }
    void conv3x3_8to8_identity_residual_avx(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_float<OpImplAVX<true>, 8, 8>(src, dst, kernels, biases, Identity{}, ResidualArg{ id, scale });
        else
#   endif
            conv3x3_float<OpImplAVX<false>, 8, 8>(src, dst, kernels, biases, Identity{}, ResidualArg{ id, scale });
    }
    void conv3x3_8to8_identity_residual_conv1x1_8to8_prelu_add_avx(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1,
        const Image& id, const float scale,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_conv1x1_float<OpImplAVX<true>, 8, 8, 8, false, false>(
                src, dst,
                kernels1, biases1, Identity{}, ResidualArg{ id, scale },
                kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
            );
        else
#   endif
            conv3x3_conv1x1_float<OpImplAVX<false>, 8, 8, 8, false, false>(
                src, dst,
                kernels1, biases1, Identity{}, ResidualArg{ id, scale },
                kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
            );
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_add_avx(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
        {
            switch (dst.type())
            {
            case Image::UInt8:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<true>, std::uint8_t, 8, 2>(src, dst, kernels, biases, ResidualArg{ id, 1.0f });
                break;
            case Image::UInt16:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<true>, std::uint16_t, 8, 2>(src, dst, kernels, biases, ResidualArg{ id, 1.0f });
                break;
            case Image::Float32:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<true>, float, 8, 2>(src, dst, kernels, biases, ResidualArg{ id, 1.0f });
                break;
            }
        }
        else
#   endif
        {
            switch (dst.type())
            {
            case Image::UInt8:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<false>, std::uint8_t, 8, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::UInt16:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<false>, std::uint16_t, 8, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::Float32:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<false>, float, 8, 2>(src, dst, kernels, biases, nullptr);
                break;
            }
        }
    }

    void conv3x3_1to16_identity_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplAVX<false>, std::uint8_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplAVX<false>, std::uint16_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplAVX<false>, float, 16>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_16to16_relu_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_float<OpImplAVX<true>, 16, 16>(src, dst, kernels, biases, ReLU{});
        else
#   endif
            conv3x3_float<OpImplAVX<false>, 16, 16>(src, dst, kernels, biases, ReLU{});
    }
    void conv3x3_16to16_identity_add_avx(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_float<OpImplAVX<true>, 16, 16>(src, dst, kernels, biases, Identity{}, ResidualArg{ feat, 1.0f });
        else
#   endif
            conv3x3_float<OpImplAVX<false>, 16, 16>(src, dst, kernels, biases, Identity{}, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_16to4_identity_pixelshuffle_4to1_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
        {
            switch (dst.type())
            {
            case Image::UInt8:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<true>, std::uint8_t, 16, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::UInt16:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<true>, std::uint16_t, 16, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::Float32:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<true>, float, 16, 2>(src, dst, kernels, biases, nullptr);
                break;
            }
        }
        else
#   endif
        {
            switch (dst.type())
            {
            case Image::UInt8:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<false>, std::uint8_t, 16, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::UInt16:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<false>, std::uint16_t, 16, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::Float32:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<false>, float, 16, 2>(src, dst, kernels, biases, nullptr);
                break;
            }
        }
    }

    void conv3x3_1to32_identity_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplAVX<false>, std::uint8_t, 32>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplAVX<false>, std::uint16_t, 32>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplAVX<false>, float, 32>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_32to32_relu_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_float<OpImplAVX<true>, 32, 32>(src, dst, kernels, biases, ReLU{});
        else
#   endif
            conv3x3_float<OpImplAVX<false>, 32, 32>(src, dst, kernels, biases, ReLU{});
    }
    void conv3x3_32to32_identity_add_avx(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_float<OpImplAVX<true>, 32, 32>(src, dst, kernels, biases, Identity{}, ResidualArg{ feat, 1.0f });
        else
#   endif
            conv3x3_float<OpImplAVX<false>, 32, 32>(src, dst, kernels, biases, Identity{}, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_32to4_identity_pixelshuffle_4to1_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
        {
            switch (dst.type())
            {
            case Image::UInt8:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<true>, std::uint8_t, 32, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::UInt16:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<true>, std::uint16_t, 32, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::Float32:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<true>, float, 32, 2>(src, dst, kernels, biases, nullptr);
                break;
            }
        }
        else
#   endif
        {
            switch (dst.type())
            {
            case Image::UInt8:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<false>, std::uint8_t, 32, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::UInt16:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<false>, std::uint16_t, 32, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::Float32:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<false>, float, 32, 2>(src, dst, kernels, biases, nullptr);
                break;
            }
        }
    }

    void conv5x5_1to8_identity_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
        {
            switch (src.type())
            {
            case Image::UInt8:
                conv5x5_cin1<OpImplAVX<true>, std::uint8_t, 8>(src, dst, kernels, biases, Identity{});
                break;
            case Image::UInt16:
                conv5x5_cin1<OpImplAVX<true>, std::uint16_t, 8>(src, dst, kernels, biases, Identity{});
                break;
            case Image::Float32:
                conv5x5_cin1<OpImplAVX<true>, float, 8>(src, dst, kernels, biases, Identity{});
                break;
            }
        }
        else
#   endif
        {
            switch (src.type())
            {
            case Image::UInt8:
                conv5x5_cin1<OpImplAVX<false>, std::uint8_t, 8>(src, dst, kernels, biases, Identity{});
                break;
            case Image::UInt16:
                conv5x5_cin1<OpImplAVX<false>, std::uint16_t, 8>(src, dst, kernels, biases, Identity{});
                break;
            case Image::Float32:
                conv5x5_cin1<OpImplAVX<false>, float, 8>(src, dst, kernels, biases, Identity{});
                break;
            }
        }
    }
    void conv3x3_8to8_prelu_conv1x1_8to8_add_prelu_avx(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_conv1x1_float<OpImplAVX<true>, 8, 8, 8, false, true>(
                src, dst,
                kernels1, biases1, PReLU{ alphas1 }, nullptr,
                kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
            );
        else
#   endif
            conv3x3_conv1x1_float<OpImplAVX<false>, 8, 8, 8, false, true>(
                src, dst,
                kernels1, biases1, PReLU{ alphas1 }, nullptr,
                kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
            );
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
        {
            switch (dst.type())
            {
            case Image::UInt8:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<true>, std::uint8_t, 8, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::UInt16:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<true>, std::uint16_t, 8, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::Float32:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<true>, float, 8, 2>(src, dst, kernels, biases, nullptr);
                break;
            }
        }
        else
#   endif
        {
            switch (dst.type())
            {
            case Image::UInt8:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<false>, std::uint8_t, 8, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::UInt16:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<false>, std::uint16_t, 8, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::Float32:
                conv3x3_identity_pixelshuffle_float<OpImplAVX<false>, float, 8, 2>(src, dst, kernels, biases, nullptr);
                break;
            }
        }
    }

    void conv5x5_1to16_identity_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
        {
            switch (src.type())
            {
            case Image::UInt8:
                conv5x5_cin1<OpImplAVX<true>, std::uint8_t, 16>(src, dst, kernels, biases, Identity{});
                break;
            case Image::UInt16:
                conv5x5_cin1<OpImplAVX<true>, std::uint16_t, 16>(src, dst, kernels, biases, Identity{});
                break;
            case Image::Float32:
                conv5x5_cin1<OpImplAVX<true>, float, 16>(src, dst, kernels, biases, Identity{});
                break;
            }
        }
        else
#   endif
        {
            switch (src.type())
            {
            case Image::UInt8:
                conv5x5_cin1<OpImplAVX<false>, std::uint8_t, 16>(src, dst, kernels, biases, Identity{});
                break;
            case Image::UInt16:
                conv5x5_cin1<OpImplAVX<false>, std::uint16_t, 16>(src, dst, kernels, biases, Identity{});
                break;
            case Image::Float32:
                conv5x5_cin1<OpImplAVX<false>, float, 16>(src, dst, kernels, biases, Identity{});
                break;
            }
        }
    }
    void conv3x3_16to16_prelu_avx(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_float<OpImplAVX<true>, 16, 16>(src, dst, kernels, biases, PReLU{ alphas });
        else
#   endif
            conv3x3_float<OpImplAVX<false>, 16, 16>(src, dst, kernels, biases, PReLU{ alphas });
    }
    void conv3x3_16to16_prelu_conv1x1_16to16_add_prelu_avx(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_conv1x1_float<OpImplAVX<true>, 16, 16, 16, false, true>(
                src, dst,
                kernels1, biases1, PReLU{ alphas1 }, nullptr,
                kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
            );
        else
#   endif
            conv3x3_conv1x1_float<OpImplAVX<false>, 16, 16, 16, false, true>(
                src, dst,
                kernels1, biases1, PReLU{ alphas1 }, nullptr,
                kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
            );
    }
}
