#ifndef AC_CORE_INTERNAL_PROCESSOR_CPU_AVX512_HPP
#define AC_CORE_INTERNAL_PROCESSOR_CPU_AVX512_HPP

#include <cstring>

#include <immintrin.h>

#include "AC/Util/Macro.hpp"

#include "AC/Core/Internal/Processor/CPU/X86/AVX.hpp"

namespace ac::core::cpu
{
    struct OpImplAVX512
    {
    private:
        static AC_FORCE_INLINE float hsum(const __m512& v) noexcept
        {
            return _mm512_reduce_add_ps(v);
        }

        template <int cin, int cpos, int sgroupSize, int scount>
        static AC_FORCE_INLINE void conv_kernel(const int sgroupIdx, const float* const* const rptr, __m512* const s, float* const out, const float* const kernels) noexcept
        {
            constexpr int vstep = 16;
            constexpr int count = cin / vstep;
            constexpr int remain = cin % vstep;

            for (int n = 0; n < scount; n++) s[n] = _mm512_setzero_ps();
            for (int p = 0; p < cpos; p++)
            {
                for (int idx = 0; idx < count; idx++)
                {
                    __m512 r = _mm512_loadu_ps(rptr[p] + idx * vstep);
                    for (int n = 0; n < scount; n++)
                    {
                        __m512 k = _mm512_loadu_ps(kernels + (sgroupIdx * sgroupSize + n) * cin * cpos + cin * p + idx * vstep);
                        s[n] = _mm512_fmadd_ps(r, k, s[n]);
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
        static AC_FORCE_INLINE float dot(const float* const v1, const float* const v2) noexcept
        {
            constexpr int vstep = 16;
            if constexpr (vsize < vstep) return OpImplAVX<true>::template dot<vsize>(v1, v2);
            else
            {
                constexpr int count = vsize / vstep;
                constexpr int remain = vsize % vstep;

                float sum = 0.0f;

                __m512 s = _mm512_setzero_ps();
                for (int idx = 0; idx < count; idx++)
                {
                    __m512 r1 = _mm512_loadu_ps(v1 + idx * vstep);
                    __m512 r2 = _mm512_loadu_ps(v2 + idx * vstep);
                    s = _mm512_fmadd_ps(r1, r2, s);
                }
                sum += hsum(s);

                if constexpr (remain)
                    for (int i = count * vstep; i < vsize; i++)
                        sum += v1[i] * v2[i];

                return sum;
            }
        }

        template <int cout, int cpos>
        static AC_FORCE_INLINE void conv_cin1(const float* const rptr, float* const out, const float* const kernels, const float* const biases) noexcept
        {
            constexpr int vstep = 16;
            if constexpr (cpos < vstep) OpImplAVX<true>::template conv_cin1<cout, cpos>(rptr, out, kernels, biases);
            else
            {
                constexpr int count = cpos / vstep;
                constexpr int remain = cpos % vstep;

                __m512 r[count];
                for (int idx = 0; idx < count; idx++) r[idx] = _mm512_loadu_ps(rptr + idx * vstep);

                for (int n = 0; n < cout; n++)
                {
                    __m512 s = _mm512_setzero_ps();
                    auto kptr = kernels + n * cpos;
                    for (int idx = 0; idx < count; idx++)
                    {
                        __m512 k = _mm512_loadu_ps(kptr + idx * vstep);
                        s = _mm512_fmadd_ps(r[idx], k, s);
                    }
                    auto sum = hsum(s);

                    for (int i = 0; i < remain; i++) sum += rptr[count * vstep + i] * kptr[count * vstep + i];

                    out[n] = sum + biases[n];
                }
            }
        }

        template <int cin, int cout, int cpos>
        static AC_FORCE_INLINE void conv(const float* const* const rptr, float* const out, const float* const kernels, const float* const biases) noexcept
        {
            constexpr int vstep = 16;
            if constexpr (cin < vstep) OpImplAVX<true>::template conv<cin, cout, cpos>(rptr, out, kernels, biases);
            else
            {
#           if defined(_MSC_VER) && !defined(__clang__)
                constexpr int scount = 8; // MSVC seems prefer 8.
#           else
                constexpr int scount = 16;
#           endif
                constexpr int sgroup = cout / scount;
                constexpr int sremian = cout % scount;

                std::memcpy(out, biases, sizeof(float) * cout);

                __m512 s[sgroup > 0 ? scount : sremian];

                if constexpr (sgroup)
                    for (int i = 0; i < sgroup; i++)
                        conv_kernel<cin, cpos, scount, scount>(i, rptr, s, out, kernels);
                if constexpr (sremian)
                    conv_kernel<cin, cpos, scount, sremian>(sgroup, rptr, s, out, kernels);
            }
        }
    };
}

#endif
