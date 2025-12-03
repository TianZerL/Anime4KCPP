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

    template <int cin, int cout, typename ActiveFunc, typename... ResidualArgs>
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

                auto tl = static_cast<const float*>(src.ptr(j - lp, i - tp));
                auto tc = static_cast<const float*>(src.ptr(j     , i - tp));
                auto tr = static_cast<const float*>(src.ptr(j + rp, i - tp));
                auto ml = static_cast<const float*>(src.ptr(j - lp, i     ));
                auto mc = static_cast<const float*>(src.ptr(j     , i     ));
                auto mr = static_cast<const float*>(src.ptr(j + rp, i     ));
                auto bl = static_cast<const float*>(src.ptr(j - lp, i + bp));
                auto bc = static_cast<const float*>(src.ptr(j     , i + bp));
                auto br = static_cast<const float*>(src.ptr(j + rp, i + bp));

                constexpr int vstep = 4;
                constexpr int count = cin / vstep;
                constexpr int remain = cin % vstep;

                v128_t r0[count + (remain ? 1 : 0)]{};
                v128_t r1[count + (remain ? 1 : 0)]{};
                v128_t r2[count + (remain ? 1 : 0)]{};
                v128_t r3[count + (remain ? 1 : 0)]{};
                v128_t r4[count + (remain ? 1 : 0)]{};
                v128_t r5[count + (remain ? 1 : 0)]{};
                v128_t r6[count + (remain ? 1 : 0)]{};
                v128_t r7[count + (remain ? 1 : 0)]{};
                v128_t r8[count + (remain ? 1 : 0)]{};

                for (int idx = 0; idx < count; idx++)
                {
                    r0[idx] = wasm_v128_load(tl + idx * vstep);
                    r1[idx] = wasm_v128_load(tc + idx * vstep);
                    r2[idx] = wasm_v128_load(tr + idx * vstep);
                    r3[idx] = wasm_v128_load(ml + idx * vstep);
                    r4[idx] = wasm_v128_load(mc + idx * vstep);
                    r5[idx] = wasm_v128_load(mr + idx * vstep);
                    r6[idx] = wasm_v128_load(bl + idx * vstep);
                    r7[idx] = wasm_v128_load(bc + idx * vstep);
                    r8[idx] = wasm_v128_load(br + idx * vstep);
                }
                if constexpr (remain)
                {
                    r0[count] = wasm_f32x4_make((tl + count * vstep)[0], remain > 1 ? (tl + count * vstep)[1] : 0.0f, remain > 2 ? (tl + count * vstep)[2] : 0.0f, 0.0f);
                    r1[count] = wasm_f32x4_make((tc + count * vstep)[0], remain > 1 ? (tc + count * vstep)[1] : 0.0f, remain > 2 ? (tc + count * vstep)[2] : 0.0f, 0.0f);
                    r2[count] = wasm_f32x4_make((tr + count * vstep)[0], remain > 1 ? (tr + count * vstep)[1] : 0.0f, remain > 2 ? (tr + count * vstep)[2] : 0.0f, 0.0f);
                    r3[count] = wasm_f32x4_make((ml + count * vstep)[0], remain > 1 ? (ml + count * vstep)[1] : 0.0f, remain > 2 ? (ml + count * vstep)[2] : 0.0f, 0.0f);
                    r4[count] = wasm_f32x4_make((mc + count * vstep)[0], remain > 1 ? (mc + count * vstep)[1] : 0.0f, remain > 2 ? (mc + count * vstep)[2] : 0.0f, 0.0f);
                    r5[count] = wasm_f32x4_make((mr + count * vstep)[0], remain > 1 ? (mr + count * vstep)[1] : 0.0f, remain > 2 ? (mr + count * vstep)[2] : 0.0f, 0.0f);
                    r6[count] = wasm_f32x4_make((bl + count * vstep)[0], remain > 1 ? (bl + count * vstep)[1] : 0.0f, remain > 2 ? (bl + count * vstep)[2] : 0.0f, 0.0f);
                    r7[count] = wasm_f32x4_make((bc + count * vstep)[0], remain > 1 ? (bc + count * vstep)[1] : 0.0f, remain > 2 ? (bc + count * vstep)[2] : 0.0f, 0.0f);
                    r8[count] = wasm_f32x4_make((br + count * vstep)[0], remain > 1 ? (br + count * vstep)[1] : 0.0f, remain > 2 ? (br + count * vstep)[2] : 0.0f, 0.0f);
                }

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
                    for (int idx = 0; idx < count; idx++)
                    {
                        v128_t k0 = wasm_v128_load(kptr[0] + idx * vstep);
                        v128_t k1 = wasm_v128_load(kptr[1] + idx * vstep);
                        v128_t k2 = wasm_v128_load(kptr[2] + idx * vstep);
                        v128_t k3 = wasm_v128_load(kptr[3] + idx * vstep);
                        v128_t k4 = wasm_v128_load(kptr[4] + idx * vstep);
                        v128_t k5 = wasm_v128_load(kptr[5] + idx * vstep);
                        v128_t k6 = wasm_v128_load(kptr[6] + idx * vstep);
                        v128_t k7 = wasm_v128_load(kptr[7] + idx * vstep);
                        v128_t k8 = wasm_v128_load(kptr[8] + idx * vstep);

                        s0 = wasm_f32x4_add(wasm_f32x4_mul(r0[idx], k0), s0);
                        s1 = wasm_f32x4_add(wasm_f32x4_mul(r1[idx], k1), s1);
                        s2 = wasm_f32x4_add(wasm_f32x4_mul(r2[idx], k2), s2);
                        s0 = wasm_f32x4_add(wasm_f32x4_mul(r3[idx], k3), s0);
                        s1 = wasm_f32x4_add(wasm_f32x4_mul(r4[idx], k4), s1);
                        s2 = wasm_f32x4_add(wasm_f32x4_mul(r5[idx], k5), s2);
                        s0 = wasm_f32x4_add(wasm_f32x4_mul(r6[idx], k6), s0);
                        s1 = wasm_f32x4_add(wasm_f32x4_mul(r7[idx], k7), s1);
                        s2 = wasm_f32x4_add(wasm_f32x4_mul(r8[idx], k8), s2);
                    }
                    if constexpr (remain)
                    {
                        v128_t k0 = wasm_f32x4_make((kptr[0] + count * vstep)[0], remain > 1 ? (kptr[0] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[0] + count * vstep)[2] : 0.0f, 0.0f);
                        v128_t k1 = wasm_f32x4_make((kptr[1] + count * vstep)[0], remain > 1 ? (kptr[1] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[1] + count * vstep)[2] : 0.0f, 0.0f);
                        v128_t k2 = wasm_f32x4_make((kptr[2] + count * vstep)[0], remain > 1 ? (kptr[2] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[2] + count * vstep)[2] : 0.0f, 0.0f);
                        v128_t k3 = wasm_f32x4_make((kptr[3] + count * vstep)[0], remain > 1 ? (kptr[3] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[3] + count * vstep)[2] : 0.0f, 0.0f);
                        v128_t k4 = wasm_f32x4_make((kptr[4] + count * vstep)[0], remain > 1 ? (kptr[4] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[4] + count * vstep)[2] : 0.0f, 0.0f);
                        v128_t k5 = wasm_f32x4_make((kptr[5] + count * vstep)[0], remain > 1 ? (kptr[5] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[5] + count * vstep)[2] : 0.0f, 0.0f);
                        v128_t k6 = wasm_f32x4_make((kptr[6] + count * vstep)[0], remain > 1 ? (kptr[6] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[6] + count * vstep)[2] : 0.0f, 0.0f);
                        v128_t k7 = wasm_f32x4_make((kptr[7] + count * vstep)[0], remain > 1 ? (kptr[7] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[7] + count * vstep)[2] : 0.0f, 0.0f);
                        v128_t k8 = wasm_f32x4_make((kptr[8] + count * vstep)[0], remain > 1 ? (kptr[8] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[8] + count * vstep)[2] : 0.0f, 0.0f);

                        s0 = wasm_f32x4_add(wasm_f32x4_mul(r0[count], k0), s0);
                        s1 = wasm_f32x4_add(wasm_f32x4_mul(r1[count], k1), s1);
                        s2 = wasm_f32x4_add(wasm_f32x4_mul(r2[count], k2), s2);
                        s0 = wasm_f32x4_add(wasm_f32x4_mul(r3[count], k3), s0);
                        s1 = wasm_f32x4_add(wasm_f32x4_mul(r4[count], k4), s1);
                        s2 = wasm_f32x4_add(wasm_f32x4_mul(r5[count], k5), s2);
                        s0 = wasm_f32x4_add(wasm_f32x4_mul(r6[count], k6), s0);
                        s1 = wasm_f32x4_add(wasm_f32x4_mul(r7[count], k7), s1);
                        s2 = wasm_f32x4_add(wasm_f32x4_mul(r8[count], k8), s2);
                    }
                    float sum = wasm_simd128_f32x4_hsum(wasm_f32x4_add(s0, wasm_f32x4_add(s1, s2))) + biases[n];

                    if constexpr (sizeof...(ResidualArgs))
                        for (int idx = 0; idx < sizeof...(ResidualArgs); idx++)
                            sum = sum * scales[idx] + iptrs[idx][n];

                    out[n] = activeFunc(sum);
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
                    out[n] = activeFunc(sum + k8 * r8 + biases[n]);
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

    template <typename OUT>
    inline void conv3x3_8to4_identity_pixelshuffle_4to1_wasm_simd128_float(const Image& src, Image& dst, const float* const kernels, const float* const biases) noexcept
    {
        static constexpr int cin = 8;
        static constexpr int upscale = 2;

        util::parallelFor(0, src.height(), [&](const int i) {
            auto tp = i > 0 ? 1 : 0;
            auto bp = i < src.height() - 1 ? 1 : 0;

            for (int j = 0; j < src.width(); j++)
            {
                auto dstY = i * upscale;
                auto dstX = j * upscale;

                auto lp = j > 0 ? 1 : 0;
                auto rp = j < src.width() - 1 ? 1 : 0;

                auto tl = static_cast<const float*>(src.ptr(j - lp, i - tp));
                auto tc = static_cast<const float*>(src.ptr(j     , i - tp));
                auto tr = static_cast<const float*>(src.ptr(j + rp, i - tp));
                auto ml = static_cast<const float*>(src.ptr(j - lp, i     ));
                auto mc = static_cast<const float*>(src.ptr(j     , i     ));
                auto mr = static_cast<const float*>(src.ptr(j + rp, i     ));
                auto bl = static_cast<const float*>(src.ptr(j - lp, i + bp));
                auto bc = static_cast<const float*>(src.ptr(j     , i + bp));
                auto br = static_cast<const float*>(src.ptr(j + rp, i + bp));

                constexpr int vstep = 4;
                constexpr int count = cin / vstep;

                v128_t r0[count]{};
                v128_t r1[count]{};
                v128_t r2[count]{};
                v128_t r3[count]{};
                v128_t r4[count]{};
                v128_t r5[count]{};
                v128_t r6[count]{};
                v128_t r7[count]{};
                v128_t r8[count]{};

                for (int idx = 0; idx < count; idx++)
                {
                    r0[idx] = wasm_v128_load(tl + idx * vstep);
                    r1[idx] = wasm_v128_load(tc + idx * vstep);
                    r2[idx] = wasm_v128_load(tr + idx * vstep);
                    r3[idx] = wasm_v128_load(ml + idx * vstep);
                    r4[idx] = wasm_v128_load(mc + idx * vstep);
                    r5[idx] = wasm_v128_load(mr + idx * vstep);
                    r6[idx] = wasm_v128_load(bl + idx * vstep);
                    r7[idx] = wasm_v128_load(bc + idx * vstep);
                    r8[idx] = wasm_v128_load(br + idx * vstep);
                }

                for (int n = 0; n < 4; n++)
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
                    for (int idx = 0; idx < count; idx++)
                    {
                        v128_t k0 = wasm_v128_load(kptr[0] + idx * vstep);
                        v128_t k1 = wasm_v128_load(kptr[1] + idx * vstep);
                        v128_t k2 = wasm_v128_load(kptr[2] + idx * vstep);
                        v128_t k3 = wasm_v128_load(kptr[3] + idx * vstep);
                        v128_t k4 = wasm_v128_load(kptr[4] + idx * vstep);
                        v128_t k5 = wasm_v128_load(kptr[5] + idx * vstep);
                        v128_t k6 = wasm_v128_load(kptr[6] + idx * vstep);
                        v128_t k7 = wasm_v128_load(kptr[7] + idx * vstep);
                        v128_t k8 = wasm_v128_load(kptr[8] + idx * vstep);

                        s0 = wasm_f32x4_add(wasm_f32x4_mul(r0[idx], k0), s0);
                        s1 = wasm_f32x4_add(wasm_f32x4_mul(r1[idx], k1), s1);
                        s2 = wasm_f32x4_add(wasm_f32x4_mul(r2[idx], k2), s2);
                        s0 = wasm_f32x4_add(wasm_f32x4_mul(r3[idx], k3), s0);
                        s1 = wasm_f32x4_add(wasm_f32x4_mul(r4[idx], k4), s1);
                        s2 = wasm_f32x4_add(wasm_f32x4_mul(r5[idx], k5), s2);
                        s0 = wasm_f32x4_add(wasm_f32x4_mul(r6[idx], k6), s0);
                        s1 = wasm_f32x4_add(wasm_f32x4_mul(r7[idx], k7), s1);
                        s2 = wasm_f32x4_add(wasm_f32x4_mul(r8[idx], k8), s2);
                    }
                    float sum = wasm_simd128_f32x4_hsum(wasm_f32x4_add(s0, wasm_f32x4_add(s1, s2))) + biases[n];

                    *static_cast<OUT*>(dst.ptr(dstX + (n & 1), dstY + (n >> 1))) = fromFloat<OUT>(sum);
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
    void conv3x3_8to8_residual_identity_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale)
    {
        conv3x3_wasm_simd128_float<8, 8>(src, dst, kernels, biases, Identity(), ResidualArg{ id, scale });
    }
    void conv3x3_8to8_residual_add_identity_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale, const Image& feat)
    {
        conv3x3_wasm_simd128_float<8, 8>(src, dst, kernels, biases, Identity(), ResidualArg{ id, scale }, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_8to4_identity_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_wasm_simd128_float<8, 4>(src, dst, kernels, biases, Identity());
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_8to4_identity_pixelshuffle_4to1_wasm_simd128_float<std::uint8_t>(src, dst, kernels, biases);
            break;
        case Image::UInt16:
            conv3x3_8to4_identity_pixelshuffle_4to1_wasm_simd128_float<std::uint16_t>(src, dst, kernels, biases);
            break;
        case Image::Float32:
            conv3x3_8to4_identity_pixelshuffle_4to1_wasm_simd128_float<float>(src, dst, kernels, biases);
            break;
        }
    }
}
