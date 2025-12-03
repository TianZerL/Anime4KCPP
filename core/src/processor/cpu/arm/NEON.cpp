#include <array>

#include <arm_neon.h>

#include "AC/Core/Image.hpp"
#include "AC/Core/Util.hpp"

namespace ac::core::cpu
{
    static inline float neon_hsum_f32(const float32x4_t& v) noexcept
    {
    #if defined(__aarch64__) || defined(_M_ARM64)
        return vaddvq_f32(v);
    #else
        float32x2_t x64 = vadd_f32(vget_low_f32(v), vget_high_f32(v));
        return vget_lane_f32(vpadd_f32(x64, x64), 0);
    #endif
    }

    template <int cin, int cout, typename ActiveFunc, typename... ResidualArgs>
    inline void conv3x3_neon_float(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc, ResidualArgs&& ...residualArg)
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

                float32x4_t r0[count + (remain ? 1 : 0)]{};
                float32x4_t r1[count + (remain ? 1 : 0)]{};
                float32x4_t r2[count + (remain ? 1 : 0)]{};
                float32x4_t r3[count + (remain ? 1 : 0)]{};
                float32x4_t r4[count + (remain ? 1 : 0)]{};
                float32x4_t r5[count + (remain ? 1 : 0)]{};
                float32x4_t r6[count + (remain ? 1 : 0)]{};
                float32x4_t r7[count + (remain ? 1 : 0)]{};
                float32x4_t r8[count + (remain ? 1 : 0)]{};

                for (int idx = 0; idx < count; idx++)
                {
                    r0[idx] = vld1q_f32(tl + idx * vstep);
                    r1[idx] = vld1q_f32(tc + idx * vstep);
                    r2[idx] = vld1q_f32(tr + idx * vstep);
                    r3[idx] = vld1q_f32(ml + idx * vstep);
                    r4[idx] = vld1q_f32(mc + idx * vstep);
                    r5[idx] = vld1q_f32(mr + idx * vstep);
                    r6[idx] = vld1q_f32(bl + idx * vstep);
                    r7[idx] = vld1q_f32(bc + idx * vstep);
                    r8[idx] = vld1q_f32(br + idx * vstep);
                }
                if constexpr (remain)
                {
                    const float d0[vstep] = {(tl + count * vstep)[0], remain > 1 ? (tl + count * vstep)[1] : 0.0f, remain > 2 ? (tl + count * vstep)[2] : 0.0f, 0.0f};
                    const float d1[vstep] = {(tc + count * vstep)[0], remain > 1 ? (tc + count * vstep)[1] : 0.0f, remain > 2 ? (tc + count * vstep)[2] : 0.0f, 0.0f};
                    const float d2[vstep] = {(tr + count * vstep)[0], remain > 1 ? (tr + count * vstep)[1] : 0.0f, remain > 2 ? (tr + count * vstep)[2] : 0.0f, 0.0f};
                    const float d3[vstep] = {(ml + count * vstep)[0], remain > 1 ? (ml + count * vstep)[1] : 0.0f, remain > 2 ? (ml + count * vstep)[2] : 0.0f, 0.0f};
                    const float d4[vstep] = {(mc + count * vstep)[0], remain > 1 ? (mc + count * vstep)[1] : 0.0f, remain > 2 ? (mc + count * vstep)[2] : 0.0f, 0.0f};
                    const float d5[vstep] = {(mr + count * vstep)[0], remain > 1 ? (mr + count * vstep)[1] : 0.0f, remain > 2 ? (mr + count * vstep)[2] : 0.0f, 0.0f};
                    const float d6[vstep] = {(bl + count * vstep)[0], remain > 1 ? (bl + count * vstep)[1] : 0.0f, remain > 2 ? (bl + count * vstep)[2] : 0.0f, 0.0f};
                    const float d7[vstep] = {(bc + count * vstep)[0], remain > 1 ? (bc + count * vstep)[1] : 0.0f, remain > 2 ? (bc + count * vstep)[2] : 0.0f, 0.0f};
                    const float d8[vstep] = {(br + count * vstep)[0], remain > 1 ? (br + count * vstep)[1] : 0.0f, remain > 2 ? (br + count * vstep)[2] : 0.0f, 0.0f};
                    r0[count] = vld1q_f32(d0);
                    r1[count] = vld1q_f32(d1);
                    r2[count] = vld1q_f32(d2);
                    r3[count] = vld1q_f32(d3);
                    r4[count] = vld1q_f32(d4);
                    r5[count] = vld1q_f32(d5);
                    r6[count] = vld1q_f32(d6);
                    r7[count] = vld1q_f32(d7);
                    r8[count] = vld1q_f32(d8);
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

                    float32x4_t s0 = vdupq_n_f32(0.0f);
                    float32x4_t s1 = vdupq_n_f32(0.0f);
                    float32x4_t s2 = vdupq_n_f32(0.0f);
                    for (int idx = 0; idx < count; idx++)
                    {
                        float32x4_t k0 = vld1q_f32(kptr[0] + idx * vstep);
                        float32x4_t k1 = vld1q_f32(kptr[1] + idx * vstep);
                        float32x4_t k2 = vld1q_f32(kptr[2] + idx * vstep);
                        float32x4_t k3 = vld1q_f32(kptr[3] + idx * vstep);
                        float32x4_t k4 = vld1q_f32(kptr[4] + idx * vstep);
                        float32x4_t k5 = vld1q_f32(kptr[5] + idx * vstep);
                        float32x4_t k6 = vld1q_f32(kptr[6] + idx * vstep);
                        float32x4_t k7 = vld1q_f32(kptr[7] + idx * vstep);
                        float32x4_t k8 = vld1q_f32(kptr[8] + idx * vstep);

                        s0 = vmlaq_f32(s0, r0[idx], k0);
                        s1 = vmlaq_f32(s1, r1[idx], k1);
                        s2 = vmlaq_f32(s2, r2[idx], k2);
                        s0 = vmlaq_f32(s0, r3[idx], k3);
                        s1 = vmlaq_f32(s1, r4[idx], k4);
                        s2 = vmlaq_f32(s2, r5[idx], k5);
                        s0 = vmlaq_f32(s0, r6[idx], k6);
                        s1 = vmlaq_f32(s1, r7[idx], k7);
                        s2 = vmlaq_f32(s2, r8[idx], k8);
                    }
                    if constexpr (remain)
                    {
                        const float d0[vstep] = {(kptr[0] + count * vstep)[0], remain > 1 ? (kptr[0] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[0] + count * vstep)[2] : 0.0f, 0.0f};
                        const float d1[vstep] = {(kptr[1] + count * vstep)[0], remain > 1 ? (kptr[1] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[1] + count * vstep)[2] : 0.0f, 0.0f};
                        const float d2[vstep] = {(kptr[2] + count * vstep)[0], remain > 1 ? (kptr[2] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[2] + count * vstep)[2] : 0.0f, 0.0f};
                        const float d3[vstep] = {(kptr[3] + count * vstep)[0], remain > 1 ? (kptr[3] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[3] + count * vstep)[2] : 0.0f, 0.0f};
                        const float d4[vstep] = {(kptr[4] + count * vstep)[0], remain > 1 ? (kptr[4] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[4] + count * vstep)[2] : 0.0f, 0.0f};
                        const float d5[vstep] = {(kptr[5] + count * vstep)[0], remain > 1 ? (kptr[5] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[5] + count * vstep)[2] : 0.0f, 0.0f};
                        const float d6[vstep] = {(kptr[6] + count * vstep)[0], remain > 1 ? (kptr[6] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[6] + count * vstep)[2] : 0.0f, 0.0f};
                        const float d7[vstep] = {(kptr[7] + count * vstep)[0], remain > 1 ? (kptr[7] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[7] + count * vstep)[2] : 0.0f, 0.0f};
                        const float d8[vstep] = {(kptr[8] + count * vstep)[0], remain > 1 ? (kptr[8] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[8] + count * vstep)[2] : 0.0f, 0.0f};
                        float32x4_t k0 = vld1q_f32(d0);
                        float32x4_t k1 = vld1q_f32(d1);
                        float32x4_t k2 = vld1q_f32(d2);
                        float32x4_t k3 = vld1q_f32(d3);
                        float32x4_t k4 = vld1q_f32(d4);
                        float32x4_t k5 = vld1q_f32(d5);
                        float32x4_t k6 = vld1q_f32(d6);
                        float32x4_t k7 = vld1q_f32(d7);
                        float32x4_t k8 = vld1q_f32(d8);


                        s0 = vmlaq_f32(s0, r0[count], k0);
                        s1 = vmlaq_f32(s1, r1[count], k1);
                        s2 = vmlaq_f32(s2, r2[count], k2);
                        s0 = vmlaq_f32(s0, r3[count], k3);
                        s1 = vmlaq_f32(s1, r4[count], k4);
                        s2 = vmlaq_f32(s2, r5[count], k5);
                        s0 = vmlaq_f32(s0, r6[count], k6);
                        s1 = vmlaq_f32(s1, r7[count], k7);
                        s2 = vmlaq_f32(s2, r8[count], k8);
                    }
                    float sum = neon_hsum_f32(vaddq_f32(s0, vaddq_f32(s1, s2))) + biases[n];

                    if constexpr (sizeof...(ResidualArgs))
                        for (int idx = 0; idx < sizeof...(ResidualArgs); idx++)
                            sum = sum * scales[idx] + iptrs[idx][n];

                    out[n] = activeFunc(sum);
                }
            }
        });
    }
    template <typename IN, int cout, typename ActiveFunc>
    inline void conv3x3_neon_cin1(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc)
    {
        util::parallelFor(0, src.height(), [&](const int i) {
            auto tp = i > 0 ? 1 : 0;
            auto bp = i < src.height() - 1 ? 1 : 0;

            for (int j = 0; j < src.width(); j++)
            {
                auto out = static_cast<float*>(dst.ptr(j, i));

                auto lp = j > 0 ? 1 : 0;
                auto rp = j < src.width() - 1 ? 1 : 0;

                const float d[] = {
                    toFloat(*static_cast<const IN*>(src.ptr(j - lp, i - tp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j     , i - tp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + rp, i - tp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j - lp, i     ))),
                    toFloat(*static_cast<const IN*>(src.ptr(j     , i     ))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + rp, i     ))),
                    toFloat(*static_cast<const IN*>(src.ptr(j - lp, i + bp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j     , i + bp)))
                };
                auto r8 = toFloat(*static_cast<const IN*>(src.ptr(j + rp, i + bp)));

                float32x4_t r0 = vld1q_f32(d + 0);
                float32x4_t r4 = vld1q_f32(d + 4);

                for (int n = 0; n < cout; n++)
                {
                    float32x4_t k0 = vld1q_f32(kernels + n * 9 + 0);
                    float32x4_t k4 = vld1q_f32(kernels + n * 9 + 4);
                    auto sum = neon_hsum_f32(vmlaq_f32(vmulq_f32(r0, k0), r4, k4));
                    auto k8 = *(kernels + n * 9 + 8);
                    out[n] = activeFunc(sum + k8 * r8 + biases[n]);
                }
            }
        });
    }
    template <typename OUT, int cin, int cout>
    inline void deconv2x2_neon_float(const Image& src, Image& dst, const float* const kernels)
    {
        filter([=](const int i, const int j, const void* const sptr, void* const dptr) {
            auto in = static_cast<const float*>(sptr);
            auto out = static_cast<OUT*>(dptr);

            auto index = ((i & 1) << 1) + (j & 1);

            constexpr int vstep = 4;
            constexpr int count = cin / vstep;
            constexpr int remain = cin % vstep;

            float32x4_t  r[count + (remain ? 1 : 0)]{};
            for (int idx = 0; idx < count; idx++) r[idx] = vld1q_f32(in + idx * vstep);
            if constexpr (remain)
            {
                const float d[vstep] = {(in + count * vstep)[0], remain > 1 ? (in + count * vstep)[1] : 0.0f, remain > 2 ? (in + count * vstep)[2] : 0.0f, 0.0f};
                r[count] = vld1q_f32(d);
            }

            for (int n = 0; n < cout; n++)
            {
                auto kptr = kernels + n * cin * 4 + cin * index;
                float sum = 0.0f;
                float32x4_t k[count + (remain ? 1 : 0)]{};
                for (int idx = 0; idx < count; idx++)
                {
                    k[idx] = vld1q_f32(kptr + idx * vstep);
                    sum += neon_hsum_f32(vmulq_f32(r[idx], k[idx]));
                }
                if constexpr (remain)
                {
                    const float d[vstep] = {(kptr + count * vstep)[0], remain > 1 ? (kptr + count * vstep)[1] : 0.0f, remain > 2 ? (kptr + count * vstep)[2] : 0.0f, 0.0f};
                    k[count] = vld1q_f32(d);
                    sum += neon_hsum_f32(vmulq_f32(r[count], k[count]));
                }
                out[n] = fromFloat<OUT>(sum);
            }
        }, src, dst);
    }

    template <typename OUT>
    inline void conv3x3_8to4_identity_pixelshuffle_4to1_neon_float(const Image& src, Image& dst, const float* const kernels, const float* const biases) noexcept
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

                float32x4_t r0[count]{};
                float32x4_t r1[count]{};
                float32x4_t r2[count]{};
                float32x4_t r3[count]{};
                float32x4_t r4[count]{};
                float32x4_t r5[count]{};
                float32x4_t r6[count]{};
                float32x4_t r7[count]{};
                float32x4_t r8[count]{};

                for (int idx = 0; idx < count; idx++)
                {
                    r0[idx] = vld1q_f32(tl + idx * vstep);
                    r1[idx] = vld1q_f32(tc + idx * vstep);
                    r2[idx] = vld1q_f32(tr + idx * vstep);
                    r3[idx] = vld1q_f32(ml + idx * vstep);
                    r4[idx] = vld1q_f32(mc + idx * vstep);
                    r5[idx] = vld1q_f32(mr + idx * vstep);
                    r6[idx] = vld1q_f32(bl + idx * vstep);
                    r7[idx] = vld1q_f32(bc + idx * vstep);
                    r8[idx] = vld1q_f32(br + idx * vstep);
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

                    float32x4_t s0 = vdupq_n_f32(0.0f);
                    float32x4_t s1 = vdupq_n_f32(0.0f);
                    float32x4_t s2 = vdupq_n_f32(0.0f);
                    for (int idx = 0; idx < count; idx++)
                    {
                        float32x4_t k0 = vld1q_f32(kptr[0] + idx * vstep);
                        float32x4_t k1 = vld1q_f32(kptr[1] + idx * vstep);
                        float32x4_t k2 = vld1q_f32(kptr[2] + idx * vstep);
                        float32x4_t k3 = vld1q_f32(kptr[3] + idx * vstep);
                        float32x4_t k4 = vld1q_f32(kptr[4] + idx * vstep);
                        float32x4_t k5 = vld1q_f32(kptr[5] + idx * vstep);
                        float32x4_t k6 = vld1q_f32(kptr[6] + idx * vstep);
                        float32x4_t k7 = vld1q_f32(kptr[7] + idx * vstep);
                        float32x4_t k8 = vld1q_f32(kptr[8] + idx * vstep);

                        s0 = vmlaq_f32(s0, r0[idx], k0);
                        s1 = vmlaq_f32(s1, r1[idx], k1);
                        s2 = vmlaq_f32(s2, r2[idx], k2);
                        s0 = vmlaq_f32(s0, r3[idx], k3);
                        s1 = vmlaq_f32(s1, r4[idx], k4);
                        s2 = vmlaq_f32(s2, r5[idx], k5);
                        s0 = vmlaq_f32(s0, r6[idx], k6);
                        s1 = vmlaq_f32(s1, r7[idx], k7);
                        s2 = vmlaq_f32(s2, r8[idx], k8);
                    }
                    float sum = neon_hsum_f32(vaddq_f32(s0, vaddq_f32(s1, s2))) + biases[n];

                    *static_cast<OUT*>(dst.ptr(dstX + (n & 1), dstY + (n >> 1))) = fromFloat<OUT>(sum);
                }
            }
        });
    }

    void conv3x3_1to8_relu_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_neon_cin1<std::uint8_t, 8>(src, dst, kernels, biases, ReLU());
            break;
        case Image::UInt16:
            conv3x3_neon_cin1<std::uint16_t, 8>(src, dst, kernels, biases, ReLU());
            break;
        case Image::Float32:
            conv3x3_neon_cin1<float, 8>(src, dst, kernels, biases, ReLU());
            break;
        }
    }
    void conv3x3_8to8_relu_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_neon_float<8, 8>(src, dst, kernels, biases, ReLU());
    }
    void deconv2x2_8to1_neon(const Image& src, Image& dst, const float* kernels)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            deconv2x2_neon_float<std::uint8_t, 8, 1>(src, dst, kernels);
            break;
        case Image::UInt16:
            deconv2x2_neon_float<std::uint16_t, 8, 1>(src, dst, kernels);
            break;
        case Image::Float32:
            deconv2x2_neon_float<float, 8, 1>(src, dst, kernels);
            break;
        }
    }

    void conv3x3_1to8_identity_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_neon_cin1<std::uint8_t, 8>(src, dst, kernels, biases, Identity());
            break;
        case Image::UInt16:
            conv3x3_neon_cin1<std::uint16_t, 8>(src, dst, kernels, biases, Identity());
            break;
        case Image::Float32:
            conv3x3_neon_cin1<float, 8>(src, dst, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_8to8_lrelu_neon(const Image& src, Image& dst, const float* kernels, const float* biases, const float negativeSlope)
    {
        conv3x3_neon_float<8, 8>(src, dst, kernels, biases, LReLU(negativeSlope));
    }
    void conv3x3_8to8_residual_identity_neon(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale)
    {
        conv3x3_neon_float<8, 8>(src, dst, kernels, biases, Identity(), ResidualArg{ id, scale });
    }
    void conv3x3_8to8_residual_add_identity_neon(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale, const Image& feat)
    {
        conv3x3_neon_float<8, 8>(src, dst, kernels, biases, Identity(), ResidualArg{ id, scale }, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_8to4_identity_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_neon_float<8, 4>(src, dst, kernels, biases, Identity());
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_8to4_identity_pixelshuffle_4to1_neon_float<std::uint8_t>(src, dst, kernels, biases);
            break;
        case Image::UInt16:
            conv3x3_8to4_identity_pixelshuffle_4to1_neon_float<std::uint16_t>(src, dst, kernels, biases);
            break;
        case Image::Float32:
            conv3x3_8to4_identity_pixelshuffle_4to1_neon_float<float>(src, dst, kernels, biases);
            break;
        }
    }
}
