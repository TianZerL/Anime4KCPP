#ifndef AC_CORE_INTERNAL_PROCESSOR_CPU_WASM_SIMD128_HPP
#define AC_CORE_INTERNAL_PROCESSOR_CPU_WASM_SIMD128_HPP

#include <cstring>

#include <wasm_simd128.h>

#include "AC/Util/Macro.hpp"

#ifdef AC_CORE_WITH_EIGEN3
#   include "AC/Core/Internal/Processor/CPU/Eigen3.hpp"
#else
#   include "AC/Core/Internal/Processor/CPU/Generic.hpp"
#endif

namespace ac::core::cpu
{
    struct OpImplWASM
    {
    private:
        static constexpr int vstep = 4;

    public:
        static constexpr int alignment = vstep * sizeof(float);

    private:
        static AC_FORCE_INLINE float hsum(const v128_t& v) noexcept
        {
            v128_t v64 = wasm_f32x4_add(v, wasm_i32x4_shuffle(v, v, 2, 3, 0, 0));
            v128_t v32 = wasm_f32x4_add(v64, wasm_i32x4_shuffle(v64, v64, 1, 0, 0, 0));
            return wasm_f32x4_extract_lane(v32, 0);
        }

        template <int cin, int cpos, int sgroupSize, int scount>
        static AC_FORCE_INLINE void conv_kernel(const int sgroupIdx, const float* const* const rptr, v128_t* const s, float* const out, const float* const kernels) noexcept
        {
            constexpr int count = cin / vstep;
            constexpr int remain = cin % vstep;

            for (int n = 0; n < scount; n++) s[n] = wasm_f32x4_splat(0.0f);
            for (int p = 0; p < cpos; p++)
            {
                for (int idx = 0; idx < count; idx++)
                {
                    v128_t r = wasm_v128_load(rptr[p] + idx * vstep);
                    for (int n = 0; n < scount; n++)
                    {
                        v128_t k = wasm_v128_load(kernels + (sgroupIdx * sgroupSize + n) * cin * cpos + cin * p + idx * vstep);
                        s[n] = wasm_f32x4_add(wasm_f32x4_mul(r, k), s[n]);
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

                v128_t s = wasm_f32x4_splat(0.0f);
                for (int idx = 0; idx < count; idx++)
                {
                    v128_t r1 = wasm_v128_load(v1 + idx * vstep);
                    v128_t r2 = wasm_v128_load(v2 + idx * vstep);
                    s = wasm_f32x4_add(wasm_f32x4_mul(r1, r2), s);
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

                v128_t r[count];
                for (int idx = 0; idx < count; idx++) r[idx] = wasm_v128_load(rptr + idx * vstep);

                for (int n = 0; n < cout; n++)
                {
                    v128_t s = wasm_f32x4_splat(0.0f);
                    auto kptr = kernels + n * cpos;
                    for (int idx = 0; idx < count; idx++)
                    {
                        v128_t k = wasm_v128_load(kptr + idx * vstep);
                        s = wasm_f32x4_add(wasm_f32x4_mul(r[idx], k), s);
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
            if constexpr (cin < vstep)
            {
#           ifdef AC_CORE_WITH_EIGEN3
                OpImplEigen3::template conv<cin, cout, cpos>(rptr, out, kernels, biases);
#           else
                OpImplGeneric::template conv<cin, cout, cpos>(rptr, out, kernels, biases);
#           endif
            }
            else
            {
                constexpr int scount = 8;
                constexpr int sgroup = cout / scount;
                constexpr int sremian = cout % scount;

                std::memcpy(out, biases, sizeof(float) * cout);

                v128_t s[sgroup > 0 ? scount : sremian];

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
