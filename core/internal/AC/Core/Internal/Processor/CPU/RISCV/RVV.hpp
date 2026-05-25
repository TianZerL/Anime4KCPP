#ifndef AC_CORE_INTERNAL_PROCESSOR_CPU_RISCV_RVV_HPP
#define AC_CORE_INTERNAL_PROCESSOR_CPU_RISCV_RVV_HPP

#include <cstddef>

#include <riscv_vector.h>

#include "AC/Util/Macro.hpp"

#ifdef AC_CORE_WITH_EIGEN3
#   include "AC/Core/Internal/Processor/CPU/Eigen3.hpp"
#else
#   include "AC/Core/Internal/Processor/CPU/Generic.hpp"
#endif

namespace ac::core::cpu
{
    struct OpImplRVV
    {
    public:
        static constexpr int alignment = 4 * sizeof(float);

    private:
        template <int vsize>
        static AC_FORCE_INLINE auto vector_dot(const float* const v1, const float* const v2) noexcept
        {
            auto vstep = __riscv_vsetvl_e32m8(vsize);
            auto count = vsize / vstep;
            auto remain = vsize % vstep;

            vfloat32m8_t s = __riscv_vfmv_v_f_f32m8(0.0f, vstep);

            for (int idx = 0; idx < count; idx++)
            {
                vfloat32m8_t r1 = __riscv_vle32_v_f32m8(v1 + idx * vstep, vstep);
                vfloat32m8_t r2 = __riscv_vle32_v_f32m8(v2 + idx * vstep, vstep);
                s = __riscv_vfmacc_vv_f32m8(s, r1, r2, vstep);
            }
            if (remain)
            {
                vfloat32m8_t r1 = __riscv_vle32_v_f32m8(v1 + count * vstep, remain);
                vfloat32m8_t r2 = __riscv_vle32_v_f32m8(v2 + count * vstep, remain);
                s = __riscv_vfmacc_vv_f32m8(s, r1, r2, remain);
            }

            return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m8_f32m1(s, __riscv_vfmv_v_f_f32m1(0.0f, 1), vstep));
        }

    public:
        template <int vsize>
        static AC_FORCE_INLINE float dot(const float* const v1, const float* const v2) noexcept
        {
            if constexpr (vsize < 2)
            {
#           ifdef AC_CORE_WITH_EIGEN3
                return OpImplEigen3::template dot<vsize>(v1, v2);
#           else
                return OpImplGeneric::template dot<vsize>(v1, v2);
#           endif
            }
            else
            {
                return vector_dot<vsize>(v1, v2);
            }
        }

        template <int cout, int cpos>
        static AC_FORCE_INLINE void conv_cin1(const float* const rptr, float* const out, const float* const kernels, const float* const biases) noexcept
        {
            if constexpr (cpos < 2)
            {
#           ifdef AC_CORE_WITH_EIGEN3
                OpImplEigen3::template conv_cin1<cout, cpos>(rptr, out, kernels, biases);
#           else
                OpImplGeneric::template conv_cin1<cout, cpos>(rptr, out, kernels, biases);
#           endif
            }
            else
            {
                for (int n = 0; n < cout; n++)
                {
                    auto kptr = kernels + n * cpos;
                    out[n] = vector_dot<cpos>(rptr, kptr) + biases[n];
                }
            }
        }

        template <int cin, int cout, int cpos>
        static AC_FORCE_INLINE void conv(const float* const* const rptr, float* const out, const float* const kernels, const float* const biases) noexcept
        {
            if constexpr (cin < 2)
            {
#           ifdef AC_CORE_WITH_EIGEN3
                OpImplEigen3::template conv<cin, cout, cpos>(rptr, out, kernels, biases);
#           else
                OpImplGeneric::template conv<cin, cout, cpos>(rptr, out, kernels, biases);
#           endif
            }
            else
            {
                auto vstep = __riscv_vsetvl_e32m4(cin);
                auto count = cin / vstep;
                auto remain = cin % vstep;

                for (int n = 0; n < cout; n++)
                {
                    vfloat32m4_t s = __riscv_vfmv_v_f_f32m4(0.0f, vstep);
                    for (int p = 0; p < cpos; p++)
                    {
                        auto kptr = kernels + n * cin * cpos + cin * p;
                        for (int idx = 0; idx < count; idx++)
                        {
                            vfloat32m4_t r = __riscv_vle32_v_f32m4(rptr[p] + idx * vstep, vstep);
                            vfloat32m4_t k = __riscv_vle32_v_f32m4(kptr + idx * vstep, vstep);
                            s = __riscv_vfmacc_vv_f32m4(s, r, k, vstep);
                        }
                        if (remain)
                        {
                            vfloat32m4_t r = __riscv_vle32_v_f32m4(rptr[p] + count * vstep, remain);
                            vfloat32m4_t k = __riscv_vle32_v_f32m4(kptr + count * vstep, remain);
                            s = __riscv_vfmacc_vv_f32m4(s, r, k, remain);
                        }
                    }
                    out[n] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(s, __riscv_vfmv_v_f_f32m1(0.0f, 1), vstep)) + biases[n];
                }
            }
        }
    };
}

#endif
