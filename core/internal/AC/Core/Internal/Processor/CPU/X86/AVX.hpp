#ifndef AC_CORE_INTERNAL_PROCESSOR_CPU_AVX_HPP
#define AC_CORE_INTERNAL_PROCESSOR_CPU_AVX_HPP

#include <cstring>

#include <immintrin.h>

#include "AC/Util/Macro.hpp"

#include "AC/Core/Internal/Processor/CPU/X86/SSE.hpp"

namespace ac::core::cpu
{
    template <bool fma = false>
    struct OpImplX86SIMD256
    {
    private:
        static constexpr int vstep = 8;

    public:
        static constexpr int alignment = vstep * sizeof(float);

    private:
        static AC_FORCE_INLINE float hsum(const __m256& v) noexcept
        {
            __m128 v128 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 0x01));
            __m128 v64 = _mm_add_ps(v128, _mm_movehl_ps(v128, v128));
            __m128 v32 = _mm_add_ss(v64, _mm_movehdup_ps(v64));
            return _mm_cvtss_f32(v32);
        }

        template <int cin, int cpos, int sgroupSize, int scount>
        static AC_FORCE_INLINE void conv_kernel(const int sgroupIdx, const float* const* const rptr, __m256* const s, float* const out, const float* const kernels) noexcept
        {
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
                        __m256 k = _mm256_loadu_ps(kernels + (sgroupIdx * sgroupSize + n) * cin * cpos + cin * p + idx * vstep);
                        if constexpr (fma)
                            s[n] = _mm256_fmadd_ps(r, k, s[n]);
                        else
                            s[n] = _mm256_add_ps(_mm256_mul_ps(r, k), s[n]);
                    }
                }
                if constexpr (remain)
                    for (int c = count * vstep; c < cin; c++)
                        for (int n = 0; n < scount; n++)
                            out[sgroupIdx * sgroupSize + n] += rptr[p][c] * kernels[(sgroupIdx * sgroupSize + n) * cin * cpos + cin * p + c];
            }
            for (int n = 0; n < scount; n++) out[sgroupIdx * sgroupSize + n] += hsum(s[n]);
        }

    public:
        template <int vsize>
        static float AC_FORCE_INLINE dot(const float* const v1, const float* const v2) noexcept
        {
            if constexpr (vsize < vstep) return OpImplX86SIMD128<fma>::template dot<vsize>(v1, v2);
            else
            {
                constexpr int count = vsize / vstep;
                constexpr int remain = vsize % vstep;

                float sum = 0.0f;

                __m256 s = _mm256_setzero_ps();
                for (int idx = 0; idx < count; idx++)
                {
                    __m256 r1 = _mm256_loadu_ps(v1 + idx * vstep);
                    __m256 r2 = _mm256_loadu_ps(v2 + idx * vstep);

                    if constexpr (fma)
                        s = _mm256_fmadd_ps(r1, r2, s);
                    else
                        s = _mm256_add_ps(_mm256_mul_ps(r1, r2), s);
                }
                sum += hsum(s);

                if constexpr (remain)
                    for (int i = count * vstep; i < vsize; i++)
                        sum += v1[i] * v2[i];

                return sum;
            }
        }

        template <int cout, int cpos>
        static void AC_FORCE_INLINE conv_cin1(const float* const rptr, float* const out, const float* const kernels, const float* const biases) noexcept
        {
            if constexpr (cpos < vstep) OpImplX86SIMD128<fma>::template conv_cin1<cout, cpos>(rptr, out, kernels, biases);
            else
            {
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

                        if constexpr (fma)
                            s = _mm256_fmadd_ps(r[idx], k, s);
                        else
                            s = _mm256_add_ps(_mm256_mul_ps(r[idx], k), s);
                    }
                    auto sum = hsum(s);

                    for (int i = 0; i < remain; i++) sum += rptr[count * vstep + i] * kptr[count * vstep + i];

                    out[n] = sum + biases[n];
                }
            }
        }

        template <int cin, int cout, int cpos>
        static void AC_FORCE_INLINE conv(const float* const* const rptr, float* const out, const float* const kernels, const float* const biases) noexcept
        {
            if constexpr (cin < vstep) OpImplX86SIMD128<fma>::template conv<cin, cout, cpos>(rptr, out, kernels, biases);
            else
            {
                constexpr int scount = 8;
                constexpr int sgroup = cout / scount;
                constexpr int sremian = cout % scount;

                std::memcpy(out, biases, sizeof(float) * cout);

                __m256 s[sgroup > 0 ? scount : sremian];

                if constexpr (sgroup)
                    for (int i = 0; i < sgroup; i++)
                        conv_kernel<cin, cpos, scount, scount>(i, rptr, s, out, kernels);
                if constexpr (sremian)
                    conv_kernel<cin, cpos, scount, sremian>(sgroup, rptr, s, out, kernels);
            }
        }
    };

    using OpImplAVX = OpImplX86SIMD256<false>;
    using OpImplFMA = OpImplX86SIMD256<true>;
}

#endif
