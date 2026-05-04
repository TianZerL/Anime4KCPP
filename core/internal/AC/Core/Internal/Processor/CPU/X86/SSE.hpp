#ifndef AC_CORE_INTERNAL_PROCESSOR_CPU_SSE_HPP
#define AC_CORE_INTERNAL_PROCESSOR_CPU_SSE_HPP

#include <cstring>

#include <xmmintrin.h>

#include "AC/Util/Macro.hpp"

#ifdef AC_CORE_WITH_EIGEN3
#   include "AC/Core/Internal/Processor/CPU/Eigen3.hpp"
#else
#   include "AC/Core/Internal/Processor/CPU/Generic.hpp"
#endif

namespace ac::core::cpu
{
    struct OpImplSSE
    {
    private:
        static AC_FORCE_INLINE float hsum(const __m128& v) noexcept
        {
            __m128 v64 = _mm_add_ps(v, _mm_movehl_ps(v, v));
            __m128 v32 = _mm_add_ss(v64, _mm_shuffle_ps(v64, v64, _MM_SHUFFLE(3, 3, 1, 1)));
            return _mm_cvtss_f32(v32);
        }

        template <int cin, int scount, int cpos>
        static AC_FORCE_INLINE void conv_kernel(const int sgroupIdx, const float* const* const rptr, __m128* const s, float* const out, const float* const kernels) noexcept
        {
            constexpr int vstep = 4;
            constexpr int count = cin / vstep;
            constexpr int remain = cin % vstep;

            for (int n = 0; n < scount; n++) s[n] = _mm_setzero_ps();
            for (int p = 0; p < cpos; p++)
            {
                for (int idx = 0; idx < count; idx++)
                {
                    __m128 r = _mm_loadu_ps(rptr[p] + idx * vstep);
                    for (int n = 0; n < scount; n++)
                    {
                        __m128 k = _mm_loadu_ps(kernels + (sgroupIdx * scount + n) * cin * cpos + cin * p + idx * vstep);
                        s[n] = _mm_add_ps(_mm_mul_ps(r, k), s[n]);
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
        static AC_FORCE_INLINE float dot(const float* const v1, const float* const v2) noexcept
        {
            constexpr int vstep = 4;
            if constexpr (vsize < vstep)
            {
#ifdef AC_CORE_WITH_EIGEN3
                return OpImplEigen3::template dot<vsize>(v1, v2);
#else
                return OpImplGeneric::template dot<vsize>(v1, v2);
#endif
            }
            else
            {
                constexpr int count = vsize / vstep;
                constexpr int remain = vsize % vstep;

                float sum = 0.0f;

                __m128 s = _mm_setzero_ps();
                for (int idx = 0; idx < count; idx++)
                {
                    __m128 r1 = _mm_loadu_ps(v1 + idx * vstep);
                    __m128 r2 = _mm_loadu_ps(v2 + idx * vstep);
                    s = _mm_add_ps(_mm_mul_ps(r1, r2), s);
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
            constexpr int vstep = 4;
            if constexpr (cpos < vstep)
            {
#ifdef AC_CORE_WITH_EIGEN3
                OpImplEigen3::template conv_cin1<cout, cpos>(rptr, out, kernels, biases);
#else
                OpImplGeneric::template conv_cin1<cout, cpos>(rptr, out, kernels, biases);
#endif
            }
            else
            {
                constexpr int count = cpos / vstep;
                constexpr int remain = cpos % vstep;

                __m128 r[count];
                for (int idx = 0; idx < count; idx++) r[idx] = _mm_loadu_ps(rptr + idx * vstep);

                for (int n = 0; n < cout; n++)
                {
                    __m128 s = _mm_setzero_ps();
                    auto kptr = kernels + n * cpos;
                    for (int idx = 0; idx < count; idx++)
                    {
                        __m128 k = _mm_loadu_ps(kptr + idx * vstep);
                        s = _mm_add_ps(_mm_mul_ps(r[idx], k), s);
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
            constexpr int vstep = 4;
            if constexpr (cin < vstep)
            {
#ifdef AC_CORE_WITH_EIGEN3
                OpImplEigen3::template conv<cin, cout, cpos>(rptr, out, kernels, biases);
#else
                OpImplGeneric::template conv<cin, cout, cpos>(rptr, out, kernels, biases);
#endif
            }
            else
            {
                constexpr int scount = 8;
                constexpr int sgroup = cout / scount;
                constexpr int sremian = cout % scount;

                std::memcpy(out, biases, sizeof(float) * cout);

                __m128 s[sgroup > 0 ? scount : sremian];

                if constexpr (sgroup)
                    for (int i = 0; i < sgroup; i++)
                        conv_kernel<cin, scount, cpos>(i, rptr, s, out, kernels);
                if constexpr (sremian)
                    conv_kernel<cin, sremian, cpos>(sgroup, rptr, s, out, kernels);
            }
        }
    };
}

#endif
