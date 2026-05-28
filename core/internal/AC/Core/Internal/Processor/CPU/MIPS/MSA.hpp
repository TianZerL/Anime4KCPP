#ifndef AC_CORE_INTERNAL_PROCESSOR_CPU_MIPS_MSA_HPP
#define AC_CORE_INTERNAL_PROCESSOR_CPU_MIPS_MSA_HPP

#include <cstring>

#include <msa.h>

#include "AC/Util/Macro.hpp"

#ifdef AC_CORE_WITH_EIGEN3
#   include "AC/Core/Internal/Processor/CPU/Eigen3.hpp"
#else
#   include "AC/Core/Internal/Processor/CPU/Generic.hpp"
#endif

namespace ac::core::cpu
{
    struct OpImplMSA
    {
    private:
        static constexpr int vstep = 4;

    public:
        static constexpr int alignment = vstep * sizeof(float);

    private:
        static AC_FORCE_INLINE float hsum(const v4f32& v) noexcept
        {
            v4f32 v64 = __msa_fadd_w(v, (v4f32)__msa_shf_w((v4i32)v, ((1 << 6) | (0 << 4) | (3 << 2) | 2)));
            v4f32 v32 = __msa_fadd_w(v64, (v4f32)__msa_shf_w((v4i32)v64, ((2 << 6) | (3 << 4) | (0 << 2) | 1)));
            int i32 = __msa_copy_s_w((v4i32)v32, 0);
            float f32;
            std::memcpy(&f32, &i32, sizeof(float));
            return f32;
        }

        template <int cin, int cpos, int sgroupSize, int scount>
        static AC_FORCE_INLINE void conv_kernel(const int sgroupIdx, const float* const* const rptr, v4f32* const s, float* const out, const float* const kernels) noexcept
        {
            constexpr int count = cin / vstep;
            constexpr int remain = cin % vstep;

            for (int n = 0; n < scount; n++) s[n] = (v4f32)__msa_fill_w(0);
            for (int p = 0; p < cpos; p++)
            {
                for (int idx = 0; idx < count; idx++)
                {
                    v4f32 r = (v4f32)__msa_ld_w(const_cast<float*>(rptr[p] + idx * vstep), 0);
                    for (int n = 0; n < scount; n++)
                    {
                        v4f32 k = (v4f32)__msa_ld_w(const_cast<float*>(kernels + (sgroupIdx * sgroupSize + n) * cin * cpos + cin * p + idx * vstep), 0);
                        s[n] = __msa_fmadd_w(s[n], r, k);
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

                v4f32 s = (v4f32)__msa_fill_w(0);
                for (int idx = 0; idx < count; idx++)
                {
                    v4f32 r1 = (v4f32)__msa_ld_w(const_cast<float*>(v1 + idx * vstep), 0);
                    v4f32 r2 = (v4f32)__msa_ld_w(const_cast<float*>(v2 + idx * vstep), 0);
                    s = __msa_fmadd_w(s, r1, r2);
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
            if constexpr (cpos < vstep)
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

                v4f32 r[count];
                for (int idx = 0; idx < count; idx++) r[idx] = (v4f32)__msa_ld_w(const_cast<float*>(rptr + idx * vstep), 0);

                for (int n = 0; n < cout; n++)
                {
                    v4f32 s = (v4f32)__msa_fill_w(0);
                    auto kptr = kernels + n * cpos;
                    for (int idx = 0; idx < count; idx++)
                    {
                        v4f32 k = (v4f32)__msa_ld_w(const_cast<float*>(kptr + idx * vstep), 0);
                        s = __msa_fmadd_w(s, r[idx], k);
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

                v4f32 s[sgroup > 0 ? scount : sremian];

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
