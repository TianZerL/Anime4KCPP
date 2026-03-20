#include <array>

#include <wasm_simd128.h>

#include "AC/Core/Image.hpp"
#include "AC/Core/Util.hpp"

namespace ac::core::cpu
{
    static inline float wasm_simd128_f32x4_hsum(const v128_t& v) noexcept
    {
        v128_t v64 = wasm_f32x4_add(v, wasm_i32x4_shuffle(v, v, 2, 3, 0, 0));
        v128_t v32 = wasm_f32x4_add(v64, wasm_i32x4_shuffle(v64, v64, 1, 0, 0, 0));
        return wasm_f32x4_extract_lane(v32, 0);
    }

    template <int cin, int cout>
    inline void conv1x1_wasm_simd128_float_impl(const float* rptr[], float* const out, const float* const kernels, const float* const biases) noexcept
    {
        constexpr int vstep = 4;
        constexpr int count = cin / vstep;
        constexpr int remain = cin % vstep;

        std::memcpy(out, biases, sizeof(float) * cout);

        for (int idx = 0; idx < count; idx++)
        {
            v128_t r = wasm_v128_load(rptr[0] + idx * vstep);

            for (int n = 0; n < cout; n++)
            {
                auto kptr = kernels + n * cin;

                v128_t k = wasm_v128_load(kptr + idx * vstep);

                out[n] += wasm_simd128_f32x4_hsum(wasm_f32x4_mul(r, k));
            }
        }
        if constexpr (remain)
        {
            v128_t r = wasm_f32x4_make((rptr[0] + count * vstep)[0], remain > 1 ? (rptr[0] + count * vstep)[1] : 0.0f, remain > 2 ? (rptr[0] + count * vstep)[2] : 0.0f, 0.0f);

            for (int n = 0; n < cout; n++)
            {
                auto kptr = kernels + n * cin;

                v128_t k = wasm_f32x4_make((kptr + count * vstep)[0], remain > 1 ? (kptr + count * vstep)[1] : 0.0f, remain > 2 ? (kptr + count * vstep)[2] : 0.0f, 0.0f);

                out[n] += wasm_simd128_f32x4_hsum(wasm_f32x4_mul(r, k));
            }
        }
    }
    template <int cin, int cout>
    inline void conv3x3_wasm_simd128_float_impl(const float* rptr[], float* const out, const float* const kernels, const float* const biases) noexcept
    {
        constexpr int vstep = 4;
        constexpr int count = cin / vstep;
        constexpr int remain = cin % vstep;

        std::memcpy(out, biases, sizeof(float) * cout);

        for (int idx = 0; idx < count; idx++)
        {
            v128_t r0 = wasm_v128_load(rptr[0] + idx * vstep);
            v128_t r1 = wasm_v128_load(rptr[1] + idx * vstep);
            v128_t r2 = wasm_v128_load(rptr[2] + idx * vstep);
            v128_t r3 = wasm_v128_load(rptr[3] + idx * vstep);
            v128_t r4 = wasm_v128_load(rptr[4] + idx * vstep);
            v128_t r5 = wasm_v128_load(rptr[5] + idx * vstep);
            v128_t r6 = wasm_v128_load(rptr[6] + idx * vstep);
            v128_t r7 = wasm_v128_load(rptr[7] + idx * vstep);
            v128_t r8 = wasm_v128_load(rptr[8] + idx * vstep);

            for (int n = 0; n < cout; n++)
            {
                const float* kptr[] = {
                    kernels + n * cin * 9 + cin * 0,
                    kernels + n * cin * 9 + cin * 1,
                    kernels + n * cin * 9 + cin * 2,
                    kernels + n * cin * 9 + cin * 3,
                    kernels + n * cin * 9 + cin * 4,
                    kernels + n * cin * 9 + cin * 5,
                    kernels + n * cin * 9 + cin * 6,
                    kernels + n * cin * 9 + cin * 7,
                    kernels + n * cin * 9 + cin * 8
                };

                v128_t s0 = wasm_f32x4_splat(0.0f);
                v128_t s1 = wasm_f32x4_splat(0.0f);
                v128_t s2 = wasm_f32x4_splat(0.0f);

                v128_t k0 = wasm_v128_load(kptr[0] + idx * vstep);
                v128_t k1 = wasm_v128_load(kptr[1] + idx * vstep);
                v128_t k2 = wasm_v128_load(kptr[2] + idx * vstep);
                v128_t k3 = wasm_v128_load(kptr[3] + idx * vstep);
                v128_t k4 = wasm_v128_load(kptr[4] + idx * vstep);
                v128_t k5 = wasm_v128_load(kptr[5] + idx * vstep);
                v128_t k6 = wasm_v128_load(kptr[6] + idx * vstep);
                v128_t k7 = wasm_v128_load(kptr[7] + idx * vstep);
                v128_t k8 = wasm_v128_load(kptr[8] + idx * vstep);

                s0 = wasm_f32x4_add(wasm_f32x4_mul(r0, k0), s0);
                s1 = wasm_f32x4_add(wasm_f32x4_mul(r1, k1), s1);
                s2 = wasm_f32x4_add(wasm_f32x4_mul(r2, k2), s2);
                s0 = wasm_f32x4_add(wasm_f32x4_mul(r3, k3), s0);
                s1 = wasm_f32x4_add(wasm_f32x4_mul(r4, k4), s1);
                s2 = wasm_f32x4_add(wasm_f32x4_mul(r5, k5), s2);
                s0 = wasm_f32x4_add(wasm_f32x4_mul(r6, k6), s0);
                s1 = wasm_f32x4_add(wasm_f32x4_mul(r7, k7), s1);
                s2 = wasm_f32x4_add(wasm_f32x4_mul(r8, k8), s2);

                out[n] += wasm_simd128_f32x4_hsum(wasm_f32x4_add(s0, wasm_f32x4_add(s1, s2)));
            }
        }
        if constexpr (remain)
        {
            v128_t r0 = wasm_f32x4_make((rptr[0] + count * vstep)[0], remain > 1 ? (rptr[0] + count * vstep)[1] : 0.0f, remain > 2 ? (rptr[0] + count * vstep)[2] : 0.0f, 0.0f);
            v128_t r1 = wasm_f32x4_make((rptr[1] + count * vstep)[0], remain > 1 ? (rptr[1] + count * vstep)[1] : 0.0f, remain > 2 ? (rptr[1] + count * vstep)[2] : 0.0f, 0.0f);
            v128_t r2 = wasm_f32x4_make((rptr[2] + count * vstep)[0], remain > 1 ? (rptr[2] + count * vstep)[1] : 0.0f, remain > 2 ? (rptr[2] + count * vstep)[2] : 0.0f, 0.0f);
            v128_t r3 = wasm_f32x4_make((rptr[3] + count * vstep)[0], remain > 1 ? (rptr[3] + count * vstep)[1] : 0.0f, remain > 2 ? (rptr[3] + count * vstep)[2] : 0.0f, 0.0f);
            v128_t r4 = wasm_f32x4_make((rptr[4] + count * vstep)[0], remain > 1 ? (rptr[4] + count * vstep)[1] : 0.0f, remain > 2 ? (rptr[4] + count * vstep)[2] : 0.0f, 0.0f);
            v128_t r5 = wasm_f32x4_make((rptr[5] + count * vstep)[0], remain > 1 ? (rptr[5] + count * vstep)[1] : 0.0f, remain > 2 ? (rptr[5] + count * vstep)[2] : 0.0f, 0.0f);
            v128_t r6 = wasm_f32x4_make((rptr[6] + count * vstep)[0], remain > 1 ? (rptr[6] + count * vstep)[1] : 0.0f, remain > 2 ? (rptr[6] + count * vstep)[2] : 0.0f, 0.0f);
            v128_t r7 = wasm_f32x4_make((rptr[7] + count * vstep)[0], remain > 1 ? (rptr[7] + count * vstep)[1] : 0.0f, remain > 2 ? (rptr[7] + count * vstep)[2] : 0.0f, 0.0f);
            v128_t r8 = wasm_f32x4_make((rptr[8] + count * vstep)[0], remain > 1 ? (rptr[8] + count * vstep)[1] : 0.0f, remain > 2 ? (rptr[8] + count * vstep)[2] : 0.0f, 0.0f);

            for (int n = 0; n < cout; n++)
            {
                const float* kptr[] = {
                    kernels + n * cin * 9 + cin * 0,
                    kernels + n * cin * 9 + cin * 1,
                    kernels + n * cin * 9 + cin * 2,
                    kernels + n * cin * 9 + cin * 3,
                    kernels + n * cin * 9 + cin * 4,
                    kernels + n * cin * 9 + cin * 5,
                    kernels + n * cin * 9 + cin * 6,
                    kernels + n * cin * 9 + cin * 7,
                    kernels + n * cin * 9 + cin * 8
                };

                v128_t s0 = wasm_f32x4_splat(0.0f);
                v128_t s1 = wasm_f32x4_splat(0.0f);
                v128_t s2 = wasm_f32x4_splat(0.0f);

                v128_t k0 = wasm_f32x4_make((kptr[0] + count * vstep)[0], remain > 1 ? (kptr[0] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[0] + count * vstep)[2] : 0.0f, 0.0f);
                v128_t k1 = wasm_f32x4_make((kptr[1] + count * vstep)[0], remain > 1 ? (kptr[1] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[1] + count * vstep)[2] : 0.0f, 0.0f);
                v128_t k2 = wasm_f32x4_make((kptr[2] + count * vstep)[0], remain > 1 ? (kptr[2] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[2] + count * vstep)[2] : 0.0f, 0.0f);
                v128_t k3 = wasm_f32x4_make((kptr[3] + count * vstep)[0], remain > 1 ? (kptr[3] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[3] + count * vstep)[2] : 0.0f, 0.0f);
                v128_t k4 = wasm_f32x4_make((kptr[4] + count * vstep)[0], remain > 1 ? (kptr[4] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[4] + count * vstep)[2] : 0.0f, 0.0f);
                v128_t k5 = wasm_f32x4_make((kptr[5] + count * vstep)[0], remain > 1 ? (kptr[5] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[5] + count * vstep)[2] : 0.0f, 0.0f);
                v128_t k6 = wasm_f32x4_make((kptr[6] + count * vstep)[0], remain > 1 ? (kptr[6] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[6] + count * vstep)[2] : 0.0f, 0.0f);
                v128_t k7 = wasm_f32x4_make((kptr[7] + count * vstep)[0], remain > 1 ? (kptr[7] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[7] + count * vstep)[2] : 0.0f, 0.0f);
                v128_t k8 = wasm_f32x4_make((kptr[8] + count * vstep)[0], remain > 1 ? (kptr[8] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[8] + count * vstep)[2] : 0.0f, 0.0f);

                s0 = wasm_f32x4_add(wasm_f32x4_mul(r0, k0), s0);
                s1 = wasm_f32x4_add(wasm_f32x4_mul(r1, k1), s1);
                s2 = wasm_f32x4_add(wasm_f32x4_mul(r2, k2), s2);
                s0 = wasm_f32x4_add(wasm_f32x4_mul(r3, k3), s0);
                s1 = wasm_f32x4_add(wasm_f32x4_mul(r4, k4), s1);
                s2 = wasm_f32x4_add(wasm_f32x4_mul(r5, k5), s2);
                s0 = wasm_f32x4_add(wasm_f32x4_mul(r6, k6), s0);
                s1 = wasm_f32x4_add(wasm_f32x4_mul(r7, k7), s1);
                s2 = wasm_f32x4_add(wasm_f32x4_mul(r8, k8), s2);

                out[n] += wasm_simd128_f32x4_hsum(wasm_f32x4_add(s0, wasm_f32x4_add(s1, s2)));
            }
        }
    }

    template <int cin, int cout, bool postactive = false, typename ActiveFunc, typename... ResidualArgs>
    inline void conv1x1_wasm_simd128_float(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc, ResidualArgs&& ...residualArg)
    {
        [[maybe_unused]] const std::array<float, sizeof...(ResidualArgs)> scales{ residualArg.scale... };

        util::parallelFor(0, src.height(), [&](const int i) {
            auto tp = i > 0 ? 1 : 0;
            auto bp = i < src.height() - 1 ? 1 : 0;

            for (int j = 0; j < src.width(); j++)
            {
                [[maybe_unused]] const std::array<const float*, sizeof...(ResidualArgs)> iptrs{ static_cast<const float*>(residualArg.image.ptr(j, i))... };

                auto out = static_cast<float*>(dst.ptr(j, i));

                auto lp = j > 0 ? 1 : 0;
                auto rp = j < src.width() - 1 ? 1 : 0;

                const float* rptr[] = { static_cast<const float*>(src.ptr(j, i)) };

                float sum[cout]{};

                conv1x1_wasm_simd128_float_impl<cin, cout>(rptr, sum, kernels, biases);

                for (int n = 0; n < cout; n++)
                {
                    if constexpr (!postactive) sum[n] = activeFunc(sum[n], n);

                    if constexpr (sizeof...(ResidualArgs))
                        for (int idx = 0; idx < sizeof...(ResidualArgs); idx++)
                            sum[n] = sum[n] * scales[idx] + iptrs[idx][n];

                    if constexpr (postactive) sum[n] = activeFunc(sum[n], n);

                    out[n] = sum[n];
                }
            }
        });
    }
    template <int cin, int cout, bool postactive = false, typename ActiveFunc, typename... ResidualArgs>
    inline void conv3x3_wasm_simd128_float(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc, ResidualArgs&& ...residualArg)
    {
        [[maybe_unused]] const std::array<float, sizeof...(ResidualArgs)> scales{ residualArg.scale... };

        util::parallelFor(0, src.height(), [&](const int i) {
            auto tp = i > 0 ? 1 : 0;
            auto bp = i < src.height() - 1 ? 1 : 0;

            for (int j = 0; j < src.width(); j++)
            {
                [[maybe_unused]] const std::array<const float*, sizeof...(ResidualArgs)> iptrs{ static_cast<const float*>(residualArg.image.ptr(j, i))... };

                auto out = static_cast<float*>(dst.ptr(j, i));

                auto lp = j > 0 ? 1 : 0;
                auto rp = j < src.width() - 1 ? 1 : 0;

                const float* rptr[] = {
                    static_cast<const float*>(src.ptr(j - lp, i - tp)),
                    static_cast<const float*>(src.ptr(j     , i - tp)),
                    static_cast<const float*>(src.ptr(j + rp, i - tp)),
                    static_cast<const float*>(src.ptr(j - lp, i     )),
                    static_cast<const float*>(src.ptr(j     , i     )),
                    static_cast<const float*>(src.ptr(j + rp, i     )),
                    static_cast<const float*>(src.ptr(j - lp, i + bp)),
                    static_cast<const float*>(src.ptr(j     , i + bp)),
                    static_cast<const float*>(src.ptr(j + rp, i + bp)),
                };

                float sum[cout]{};

                conv3x3_wasm_simd128_float_impl<cin, cout>(rptr, sum, kernels, biases);

                for (int n = 0; n < cout; n++)
                {
                    if constexpr (!postactive) sum[n] = activeFunc(sum[n], n);

                    if constexpr (sizeof...(ResidualArgs))
                        for (int idx = 0; idx < sizeof...(ResidualArgs); idx++)
                            sum[n] = sum[n] * scales[idx] + iptrs[idx][n];

                    if constexpr (postactive) sum[n] = activeFunc(sum[n], n);

                    out[n] = sum[n];
                }
            }
        });
    }
    template <typename IN, int cout, typename ActiveFunc>
    inline void conv3x3_wasm_simd128_cin1(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc)
    {
        util::parallelFor(0, src.height(), [&](const int i) {
            auto tp = i > 0 ? 1 : 0;
            auto bp = i < src.height() - 1 ? 1 : 0;

            for (int j = 0; j < src.width(); j++)
            {
                auto out = static_cast<float*>(dst.ptr(j, i));

                auto lp = j > 0 ? 1 : 0;
                auto rp = j < src.width() - 1 ? 1 : 0;

                v128_t r0 = wasm_f32x4_make(
                    toFloat(*static_cast<const IN*>(src.ptr(j - lp, i - tp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j     , i - tp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + rp, i - tp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j - lp, i     )))
                );
                v128_t r4 = wasm_f32x4_make(
                    toFloat(*static_cast<const IN*>(src.ptr(j     , i     ))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + rp, i     ))),
                    toFloat(*static_cast<const IN*>(src.ptr(j - lp, i + bp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j     , i + bp)))
                );
                auto r8 = toFloat(*static_cast<const IN*>(src.ptr(j + rp, i + bp)));

                for (int n = 0; n < cout; n++)
                {
                    v128_t k0 = wasm_v128_load(kernels + n * 9 + 0);
                    v128_t k4 = wasm_v128_load(kernels + n * 9 + 4);
                    auto sum = wasm_simd128_f32x4_hsum(wasm_f32x4_add(wasm_f32x4_mul(r0, k0), wasm_f32x4_mul(r4, k4)));
                    auto k8 = *(kernels + n * 9 + 8);
                    out[n] = activeFunc(sum + r8 * k8 + biases[n], n);
                }
            }
        });
    }
    template <typename IN, int cout, typename ActiveFunc>
    inline void conv5x5_wasm_simd128_cin1(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc)
    {
        util::parallelFor(0, src.height(), [&](const int i) {
            int ioffsets[5] = { i > 1 ? -2 : (i > 0 ? -1 : 0) , i > 0 ? -1 : 0 , 0, i < src.height() - 1 ? 1 : 0, i < src.height() - 2 ? 2 : (i < src.height() - 1 ? 1 : 0)};

            for (int j = 0; j < src.width(); j++)
            {
                auto out = static_cast<float*>(dst.ptr(j, i));

                int joffsets[5] = { j > 1 ? -2 : (j > 0 ? -1 : 0), j > 0 ? -1 : 0, 0, j < src.width() - 1 ? 1 : 0 ,j < src.width() - 2 ? 2 : (j < src.width() - 1 ? 1 : 0)};

                v128_t r0 = wasm_f32x4_make(
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[0], i + ioffsets[0]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[1], i + ioffsets[0]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[2], i + ioffsets[0]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[3], i + ioffsets[0])))
                );
                v128_t r4 = wasm_f32x4_make(
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[4], i + ioffsets[0]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[0], i + ioffsets[1]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[1], i + ioffsets[1]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[2], i + ioffsets[1])))
                );
                v128_t r8 = wasm_f32x4_make(
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[3], i + ioffsets[1]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[4], i + ioffsets[1]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[0], i + ioffsets[2]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[1], i + ioffsets[2])))
                );
                v128_t r12 = wasm_f32x4_make(
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[2], i + ioffsets[2]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[3], i + ioffsets[2]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[4], i + ioffsets[2]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[0], i + ioffsets[3])))
                );
                v128_t r16 = wasm_f32x4_make(
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[1], i + ioffsets[3]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[2], i + ioffsets[3]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[3], i + ioffsets[3]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[4], i + ioffsets[3])))
                );
                v128_t r20 = wasm_f32x4_make(
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[0], i + ioffsets[4]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[1], i + ioffsets[4]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[2], i + ioffsets[4]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[3], i + ioffsets[4])))
                );
                auto r24 = toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[4], i + ioffsets[4])));

                for (int n = 0; n < cout; n++)
                {
                    v128_t k0 = wasm_v128_load(kernels + n * 25 + 0);
                    v128_t k4 = wasm_v128_load(kernels + n * 25 + 4);
                    v128_t k8 = wasm_v128_load(kernels + n * 25 + 8);
                    v128_t k12 = wasm_v128_load(kernels + n * 25 + 12);
                    v128_t k16 = wasm_v128_load(kernels + n * 25 + 16);
                    v128_t k20 = wasm_v128_load(kernels + n * 25 + 20);

                    v128_t s0 = wasm_f32x4_splat(0.0f);
                    v128_t s1 = wasm_f32x4_splat(0.0f);
                    v128_t s2 = wasm_f32x4_splat(0.0f);

                    s0 = wasm_f32x4_add(wasm_f32x4_mul(r0, k0), s0);
                    s1 = wasm_f32x4_add(wasm_f32x4_mul(r4, k4), s1);
                    s2 = wasm_f32x4_add(wasm_f32x4_mul(r8, k8), s2);
                    s0 = wasm_f32x4_add(wasm_f32x4_mul(r12, k12), s0);
                    s1 = wasm_f32x4_add(wasm_f32x4_mul(r16, k16), s1);
                    s2 = wasm_f32x4_add(wasm_f32x4_mul(r20, k20), s2);

                    auto sum = wasm_simd128_f32x4_hsum(wasm_f32x4_add(s0, wasm_f32x4_add(s1, s2)));

                    auto k24 = *(kernels + n * 25 + 24);
                    out[n] = activeFunc(sum + r24 * k24 + biases[n], n);
                }
            }
        });
    }
    template <typename OUT, int cin, int cout>
    inline void deconv2x2_wasm_simd128_float(const Image& src, Image& dst, const float* const kernels)
    {
        filter([=](const int i, const int j, const void* const sptr, void* const dptr) {
            auto in = static_cast<const float*>(sptr);
            auto out = static_cast<OUT*>(dptr);

            auto index = ((i & 1) << 1) + (j & 1);

            constexpr int vstep = 4;
            constexpr int count = cin / vstep;
            constexpr int remain = cin % vstep;

            v128_t r[count + (remain ? 1 : 0)]{};
            for (int idx = 0; idx < count; idx++) r[idx] = wasm_v128_load(in + idx * vstep);
            if constexpr (remain) r[count] = wasm_f32x4_make((in + count * vstep)[0], remain > 1 ? (in + count * vstep)[1] : 0.0f, remain > 2 ? (in + count * vstep)[2] : 0.0f, 0.0f);

            for (int n = 0; n < cout; n++)
            {
                auto kptr = kernels + n * cin * 4 + cin * index;
                float sum = 0.0f;
                v128_t k[count + (remain ? 1 : 0)]{};
                for (int idx = 0; idx < count; idx++)
                {
                    k[idx] = wasm_v128_load(kptr + idx * vstep);
                    sum += wasm_simd128_f32x4_hsum(wasm_f32x4_mul(r[idx], k[idx]));
                }
                if constexpr (remain)
                {
                    k[count] = wasm_f32x4_make((kptr + count * vstep)[0], remain > 1 ? (kptr + count * vstep)[1] : 0.0f, remain > 2 ? (kptr + count * vstep)[2] : 0.0f, 0.0f);
                    sum += wasm_simd128_f32x4_hsum(wasm_f32x4_mul(r[count], k[count]));
                }
                out[n] = fromFloat<OUT>(sum);
            }
        }, src, dst);
    }

    template <typename OUT, int cin, int upscale>
    inline void conv3x3_identity_pixelshuffle_wasm_simd128_float(const Image& src, Image& dst, const float* const kernels, const float* const biases) noexcept
    {
        static constexpr int cout = upscale * upscale;

        util::parallelFor(0, src.height(), [&](const int i) {
            auto tp = i > 0 ? 1 : 0;
            auto bp = i < src.height() - 1 ? 1 : 0;

            for (int j = 0; j < src.width(); j++)
            {
                auto dstY = i * upscale;
                auto dstX = j * upscale;

                auto lp = j > 0 ? 1 : 0;
                auto rp = j < src.width() - 1 ? 1 : 0;

                const float* rptr[] = {
                    static_cast<const float*>(src.ptr(j - lp, i - tp)),
                    static_cast<const float*>(src.ptr(j     , i - tp)),
                    static_cast<const float*>(src.ptr(j + rp, i - tp)),
                    static_cast<const float*>(src.ptr(j - lp, i     )),
                    static_cast<const float*>(src.ptr(j     , i     )),
                    static_cast<const float*>(src.ptr(j + rp, i     )),
                    static_cast<const float*>(src.ptr(j - lp, i + bp)),
                    static_cast<const float*>(src.ptr(j     , i + bp)),
                    static_cast<const float*>(src.ptr(j + rp, i + bp)),
                };

                float sum[cout]{};

                conv3x3_wasm_simd128_float_impl<cin, cout>(rptr, sum, kernels, biases);

                for (int n = 0; n < cout; n++) *static_cast<OUT*>(dst.ptr(dstX + (n & 1), dstY + (n >> 1))) = fromFloat<OUT>(sum[n]);
            }
        });
    }

    template <int cin, int ctemp, int cout, bool postactive3x3 = false, bool postactive1x1 = false, typename ActiveFunc3x3, typename ResidualArg3x3, typename ActiveFunc1x1, typename ResidualArg1x1>
    inline void conv3x3_conv1x1_wasm_simd128_float(
        const Image& src, Image& dst,
        const float* const kernels3x3, const float* const biases3x3, ActiveFunc3x3&& activeFunc3x3, ResidualArg3x3&& residualArg3x3,
        const float* const kernels1x1, const float* const biases1x1, ActiveFunc1x1&& activeFunc1x1, ResidualArg1x1&& residualArg1x1)
    {
        util::parallelFor(0, src.height(), [&](const int i) {
            auto tp = i > 0 ? 1 : 0;
            auto bp = i < src.height() - 1 ? 1 : 0;

            for (int j = 0; j < src.width(); j++)
            {
                auto out = static_cast<float*>(dst.ptr(j, i));

                auto lp = j > 0 ? 1 : 0;
                auto rp = j < src.width() - 1 ? 1 : 0;

                const float* rptr[] = {
                    static_cast<const float*>(src.ptr(j - lp, i - tp)),
                    static_cast<const float*>(src.ptr(j     , i - tp)),
                    static_cast<const float*>(src.ptr(j + rp, i - tp)),
                    static_cast<const float*>(src.ptr(j - lp, i     )),
                    static_cast<const float*>(src.ptr(j     , i     )),
                    static_cast<const float*>(src.ptr(j + rp, i     )),
                    static_cast<const float*>(src.ptr(j - lp, i + bp)),
                    static_cast<const float*>(src.ptr(j     , i + bp)),
                    static_cast<const float*>(src.ptr(j + rp, i + bp)),
                };

                float buffer[ctemp]{};

                conv3x3_wasm_simd128_float_impl<cin, ctemp>(rptr, buffer, kernels3x3, biases3x3);

                for (int n = 0; n < ctemp; n++)
                {
                    if constexpr (!postactive3x3) buffer[n] = activeFunc3x3(buffer[n], n);

                    if constexpr (std::is_same_v<ResidualArg3x3, ResidualArg>)
                        buffer[n] = buffer[n] * residualArg3x3.scale + static_cast<const float*>(residualArg3x3.image.ptr(j, i))[n];

                    if constexpr (postactive3x3) buffer[n] = activeFunc3x3(buffer[n], n);
                }

                rptr[0] = buffer;
                float sum[cout]{};
                conv1x1_wasm_simd128_float_impl<ctemp, cout>(rptr, sum, kernels1x1, biases1x1);

                for (int n = 0; n < cout; n++)
                {
                    if constexpr (!postactive1x1) sum[n] = activeFunc1x1(sum[n], n);

                    if constexpr (std::is_same_v<ResidualArg1x1, ResidualArg>)
                        sum[n] = sum[n] * residualArg1x1.scale + static_cast<const float*>(residualArg1x1.image.ptr(j, i))[n];

                    if constexpr (postactive1x1) sum[n] = activeFunc1x1(sum[n], n);

                    out[n] = sum[n];
                }
            }
        });
    }

    void conv3x3_1to8_relu_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_wasm_simd128_cin1<std::uint8_t, 8>(src, dst, kernels, biases, ReLU());
            break;
        case Image::UInt16:
            conv3x3_wasm_simd128_cin1<std::uint16_t, 8>(src, dst, kernels, biases, ReLU());
            break;
        case Image::Float32:
            conv3x3_wasm_simd128_cin1<float, 8>(src, dst, kernels, biases, ReLU());
            break;
        }
    }
    void conv3x3_8to8_relu_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_wasm_simd128_float<8, 8>(src, dst, kernels, biases, ReLU());
    }
    void deconv2x2_8to1_wasm_simd128(const Image& src, Image& dst, const float* kernels)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            deconv2x2_wasm_simd128_float<std::uint8_t, 8, 1>(src, dst, kernels);
            break;
        case Image::UInt16:
            deconv2x2_wasm_simd128_float<std::uint16_t, 8, 1>(src, dst, kernels);
            break;
        case Image::Float32:
            deconv2x2_wasm_simd128_float<float, 8, 1>(src, dst, kernels);
            break;
        }
    }

    void conv3x3_1to8_identity_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_wasm_simd128_cin1<std::uint8_t, 8>(src, dst, kernels, biases, Identity());
            break;
        case Image::UInt16:
            conv3x3_wasm_simd128_cin1<std::uint16_t, 8>(src, dst, kernels, biases, Identity());
            break;
        case Image::Float32:
            conv3x3_wasm_simd128_cin1<float, 8>(src, dst, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_8to8_lrelu_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases, const float negativeSlope)
    {
        conv3x3_wasm_simd128_float<8, 8>(src, dst, kernels, biases, LReLU(negativeSlope));
    }
    void conv3x3_8to8_identity_residual_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale)
    {
        conv3x3_wasm_simd128_float<8, 8>(src, dst, kernels, biases, Identity(), ResidualArg{ id, scale });
    }
    void conv3x3_8to8_identity_residual_add_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale, const Image& feat)
    {
        conv3x3_wasm_simd128_float<8, 8>(src, dst, kernels, biases, Identity(), ResidualArg{ id, scale }, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_wasm_simd128_float<std::uint8_t, 8, 2>(src, dst, kernels, biases);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_wasm_simd128_float<std::uint16_t, 8, 2>(src, dst, kernels, biases);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_wasm_simd128_float<float, 8, 2>(src, dst, kernels, biases);
            break;
        }
    }
    void conv3x3_8to4_identity_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_wasm_simd128_float<8, 4>(src, dst, kernels, biases, Identity());
    }

    void conv3x3_1to16_identity_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_wasm_simd128_cin1<std::uint8_t, 16>(src, dst, kernels, biases, Identity());
            break;
        case Image::UInt16:
            conv3x3_wasm_simd128_cin1<std::uint16_t, 16>(src, dst, kernels, biases, Identity());
            break;
        case Image::Float32:
            conv3x3_wasm_simd128_cin1<float, 16>(src, dst, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_16to16_relu_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_wasm_simd128_float<16, 16>(src, dst, kernels, biases, ReLU());
    }
    void conv3x3_16to16_identity_add_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
        conv3x3_wasm_simd128_float<16, 16>(src, dst, kernels, biases, Identity(), ResidualArg{ feat, 1.0f });
    }
    void conv3x3_16to4_identity_pixelshuffle_4to1_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_wasm_simd128_float<std::uint8_t, 16, 2>(src, dst, kernels, biases);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_wasm_simd128_float<std::uint16_t, 16, 2>(src, dst, kernels, biases);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_wasm_simd128_float<float, 16, 2>(src, dst, kernels, biases);
            break;
        }
    }
    void conv3x3_16to4_identity_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_wasm_simd128_float<16, 4>(src, dst, kernels, biases, Identity());
    }

    void conv3x3_1to32_identity_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_wasm_simd128_cin1<std::uint8_t, 32>(src, dst, kernels, biases, Identity());
            break;
        case Image::UInt16:
            conv3x3_wasm_simd128_cin1<std::uint16_t, 32>(src, dst, kernels, biases, Identity());
            break;
        case Image::Float32:
            conv3x3_wasm_simd128_cin1<float, 32>(src, dst, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_32to32_relu_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_wasm_simd128_float<32, 32>(src, dst, kernels, biases, ReLU());
    }
    void conv3x3_32to32_identity_add_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
        conv3x3_wasm_simd128_float<32, 32>(src, dst, kernels, biases, Identity(), ResidualArg{ feat, 1.0f });
    }
    void conv3x3_32to4_identity_pixelshuffle_4to1_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_wasm_simd128_float<std::uint8_t, 32, 2>(src, dst, kernels, biases);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_wasm_simd128_float<std::uint16_t, 32, 2>(src, dst, kernels, biases);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_wasm_simd128_float<float, 32, 2>(src, dst, kernels, biases);
            break;
        }
    }
    void conv3x3_32to4_identity_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_wasm_simd128_float<32, 4>(src, dst, kernels, biases, Identity());
    }

    void conv5x5_1to8_identity_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv5x5_wasm_simd128_cin1<std::uint8_t, 8>(src, dst, kernels, biases, Identity());
            break;
        case Image::UInt16:
            conv5x5_wasm_simd128_cin1<std::uint16_t, 8>(src, dst, kernels, biases, Identity());
            break;
        case Image::Float32:
            conv5x5_wasm_simd128_cin1<float, 8>(src, dst, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_8to8_prelu_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        conv3x3_wasm_simd128_float<8, 8>(src, dst, kernels, biases, PReLU(alphas));
    }
    void conv3x3_8to8_prelu_conv1x1_8to8_add_prelu_wasm_simd128(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_wasm_simd128_float<8, 8, 8, false, true>(
            src, dst,
            kernels1, biases1, PReLU(alphas1), nullptr,
            kernels2, biases2, PReLU(alphas2), ResidualArg{ feat, 1.0f }
        );
    }

    void conv5x5_1to16_identity_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv5x5_wasm_simd128_cin1<std::uint8_t, 16>(src, dst, kernels, biases, Identity());
            break;
        case Image::UInt16:
            conv5x5_wasm_simd128_cin1<std::uint16_t, 16>(src, dst, kernels, biases, Identity());
            break;
        case Image::Float32:
            conv5x5_wasm_simd128_cin1<float, 16>(src, dst, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_16to16_prelu_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        conv3x3_wasm_simd128_float<16, 16>(src, dst, kernels, biases, PReLU(alphas));
    }
    void conv3x3_16to16_prelu_conv1x1_16to16_add_prelu_wasm_simd128(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_wasm_simd128_float<16, 16, 16, false, true>(
            src, dst,
            kernels1, biases1, PReLU(alphas1), nullptr,
            kernels2, biases2, PReLU(alphas2), ResidualArg{ feat, 1.0f }
        );
    }
}
