#ifndef AC_CORE_INTERNAL_PROCESSOR_CPU_LOONGARCH_LASX_HPP
#define AC_CORE_INTERNAL_PROCESSOR_CPU_LOONGARCH_LASX_HPP

#include <cstring>

#include <lasxintrin.h>

#include "AC/Util/Macro.hpp"

#include "AC/Core/Internal/Processor/CPU/LoongArch/LSX.hpp"

namespace ac::core::cpu
{
    struct OpImplLASX
    {
    private:
        static constexpr int vstep = 8;

    public:
        static constexpr int alignment = vstep * sizeof(float);

    private:
        static AC_FORCE_INLINE float hsum(const __m256& v) noexcept
        {
            __m256 v64x2 = __lasx_xvfadd_s(v, (__m256)__lasx_xvbsrl_v((__m256i)v, 8));
            __m256 v32x2 = __lasx_xvfadd_s(v64x2, (__m256)__lasx_xvbsrl_v((__m256i)v64x2, 4));
            int i32lo = __lasx_xvpickve2gr_w((__m256i)v32x2, 0);
            float f32lo;
            std::memcpy(&f32lo, &i32lo, sizeof(float));
            int i32hi = __lasx_xvpickve2gr_w((__m256i)v32x2, 4);
            float f32hi;
            std::memcpy(&f32hi, &i32hi, sizeof(float));
            return f32lo + f32hi;
        }

        template <int cin, int cpos, int sgroupSize, int scount>
        static AC_FORCE_INLINE void conv_kernel(const int sgroupIdx, const float* const* const rptr, __m256* const s, float* const out, const float* const kernels) noexcept
        {
            constexpr int count = cin / vstep;
            constexpr int remain = cin % vstep;

            for (int n = 0; n < scount; n++) s[n] = (__m256)__lasx_xvrepli_w(0);
            for (int p = 0; p < cpos; p++)
            {
                for (int idx = 0; idx < count; idx++)
                {
                    __m256 r = (__m256)__lasx_xvld(rptr[p] + idx * vstep, 0);
                    for (int n = 0; n < scount; n++)
                    {
                        __m256 k = (__m256)__lasx_xvld(kernels + (sgroupIdx * sgroupSize + n) * cin * cpos + cin * p + idx * vstep, 0);
                        s[n] = __lasx_xvfmadd_s(r, k, s[n]);
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
            if constexpr (vsize < vstep) return OpImplLSX::template dot<vsize>(v1, v2);
            else
            {
                constexpr int count = vsize / vstep;
                constexpr int remain = vsize % vstep;

                float sum = 0.0f;

                __m256 s = (__m256)__lasx_xvrepli_w(0);
                for (int idx = 0; idx < count; idx++)
                {
                    __m256 r1 = (__m256)__lasx_xvld(v1 + idx * vstep, 0);
                    __m256 r2 = (__m256)__lasx_xvld(v2 + idx * vstep, 0);
                    s = __lasx_xvfmadd_s(r1, r2, s);
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
            if constexpr (cpos < vstep) OpImplLSX::template conv_cin1<cout, cpos>(rptr, out, kernels, biases);
            else
            {
                constexpr int count = cpos / vstep;
                constexpr int remain = cpos % vstep;

                __m256 r[count];
                for (int idx = 0; idx < count; idx++) r[idx] = (__m256)__lasx_xvld(rptr + idx * vstep, 0);

                for (int n = 0; n < cout; n++)
                {
                    __m256 s = (__m256)__lasx_xvrepli_w(0);
                    auto kptr = kernels + n * cpos;
                    for (int idx = 0; idx < count; idx++)
                    {
                        __m256 k = (__m256)__lasx_xvld(kptr + idx * vstep, 0);
                        s = __lasx_xvfmadd_s(r[idx], k, s);
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
            if constexpr (cin % vstep) OpImplLSX::template conv<cin, cout, cpos>(rptr, out, kernels, biases);
            else
            {
                constexpr int scount = 16;
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
}

#endif
