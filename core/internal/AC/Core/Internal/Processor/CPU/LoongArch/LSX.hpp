#ifndef AC_CORE_INTERNAL_PROCESSOR_CPU_LOONGARCH_LSX_HPP
#define AC_CORE_INTERNAL_PROCESSOR_CPU_LOONGARCH_LSX_HPP

#include <cstring>

#include <lsxintrin.h>

#include "AC/Util/Macro.hpp"

#ifdef AC_CORE_WITH_EIGEN3
#   include "AC/Core/Internal/Processor/CPU/Eigen3.hpp"
#else
#   include "AC/Core/Internal/Processor/CPU/Generic.hpp"
#endif

namespace ac::core::cpu
{
    struct OpImplLSX
    {
    private:
        static constexpr int vstep = 4;

    public:
        static constexpr int alignment = vstep * sizeof(float);

    private:
        static AC_FORCE_INLINE float hsum(const __m128& v) noexcept
        {
            __m128 v64 = __lsx_vfadd_s(v, (__m128)__lsx_vbsrl_v((__m128i)v, 8));
            __m128 v32 = __lsx_vfadd_s(v64, (__m128)__lsx_vbsrl_v((__m128i)v64, 4));
            int i32 = __lsx_vpickve2gr_w((__m128i)v32, 0);
            float f32;
            std::memcpy(&f32, &i32, sizeof(float));
            return f32;
        }

        template <int cin, int cpos, int sgroupSize, int scount>
        static AC_FORCE_INLINE void conv_kernel(const int sgroupIdx, const float* const* const rptr, __m128* const s, float* const out, const float* const kernels) noexcept
        {
            constexpr int count = cin / vstep;
            constexpr int remain = cin % vstep;

            for (int n = 0; n < scount; n++) s[n] = (__m128)__lsx_vrepli_w(0);
            for (int p = 0; p < cpos; p++)
            {
                for (int idx = 0; idx < count; idx++)
                {
                    __m128 r = (__m128)__lsx_vld(rptr[p] + idx * vstep, 0);
                    for (int n = 0; n < scount; n++)
                    {
                        __m128 k = (__m128)__lsx_vld(kernels + (sgroupIdx * sgroupSize + n) * cin * cpos + cin * p + idx * vstep, 0);
                        s[n] = __lsx_vfmadd_s(r, k, s[n]);
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
            if constexpr (vsize < vstep)
            {
#           ifdef AC_CORE_WITH_EIGEN3
                return OpImplEigen3::template dot<vsize>(v1, v2);
#           else
                return OpImplGeneric::template dot<vsize>(v1, v2);
#           endif
            }
            else
            {
                constexpr int count = vsize / vstep;
                constexpr int remain = vsize % vstep;

                float sum = 0.0f;

                __m128 s = (__m128)__lsx_vrepli_w(0);
                for (int idx = 0; idx < count; idx++)
                {
                    __m128 r1 = (__m128)__lsx_vld(v1 + idx * vstep, 0);
                    __m128 r2 = (__m128)__lsx_vld(v2 + idx * vstep, 0);
                    s = __lsx_vfmadd_s(r1, r2, s);
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
            if constexpr (cpos % vstep)
            {
#           ifdef AC_CORE_WITH_EIGEN3
                OpImplEigen3::template conv_cin1<cout, cpos>(rptr, out, kernels, biases);
#           else
                OpImplGeneric::template conv_cin1<cout, cpos>(rptr, out, kernels, biases);
#           endif
            }
            else
            {
                constexpr int count = cpos / vstep;
                constexpr int remain = cpos % vstep;

                __m128 r[count];
                for (int idx = 0; idx < count; idx++) r[idx] = (__m128)__lsx_vld(rptr + idx * vstep, 0);

                for (int n = 0; n < cout; n++)
                {
                    __m128 s = (__m128)__lsx_vrepli_w(0);
                    auto kptr = kernels + n * cpos;
                    for (int idx = 0; idx < count; idx++)
                    {
                        __m128 k = (__m128)__lsx_vld(kptr + idx * vstep, 0);
                        s = __lsx_vfmadd_s(r[idx], k, s);
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
            if constexpr (cin % vstep)
            {
#           ifdef AC_CORE_WITH_EIGEN3
                OpImplEigen3::template conv<cin, cout, cpos>(rptr, out, kernels, biases);
#           else
                OpImplGeneric::template conv<cin, cout, cpos>(rptr, out, kernels, biases);
#           endif
            }
            else
            {
                constexpr int scount = 16;
                constexpr int sgroup = cout / scount;
                constexpr int sremian = cout % scount;

                std::memcpy(out, biases, sizeof(float) * cout);

                __m128 s[sgroup > 0 ? scount : sremian];

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
