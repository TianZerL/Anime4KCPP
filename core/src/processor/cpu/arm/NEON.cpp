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

    template <int cin, int cout>
    inline void conv1x1_neon_float_impl(const float* rptr[], float* const out, const float* const kernels, const float* const biases) noexcept
    {
        constexpr int vstep = 4;
        constexpr int count = cin / vstep;
        constexpr int remain = cin % vstep;

        std::memcpy(out, biases, sizeof(float) * cout);

        for (int idx = 0; idx < count; idx++)
        {
            float32x4_t r = vld1q_f32(rptr[0] + idx * vstep);

            for (int n = 0; n < cout; n++)
            {
                auto kptr = kernels + n * cin;

                float32x4_t k = vld1q_f32(kptr + idx * vstep);

                out[n] += neon_hsum_f32(vmulq_f32(r, k));
            }
        }
        if constexpr (remain)
        {
            const float rd[vstep] = {(rptr[0] + count * vstep)[0], remain > 1 ? (rptr[0] + count * vstep)[1] : 0.0f, remain > 2 ? (rptr[0] + count * vstep)[2] : 0.0f, 0.0f};
            float32x4_t r = vld1q_f32(rd);

            for (int n = 0; n < cout; n++)
            {
                auto kptr = kernels + n * cin;

                const float kd[vstep] = {(kptr + count * vstep)[0], remain > 1 ? (kptr + count * vstep)[1] : 0.0f, remain > 2 ? (kptr + count * vstep)[2] : 0.0f, 0.0f};
                float32x4_t k = vld1q_f32(kd);

                out[n] += neon_hsum_f32(vmulq_f32(r, k));
            }
        }
    }
    template <int cin, int cout>
    inline void conv3x3_neon_float_impl(const float* rptr[], float* const out, const float* const kernels, const float* const biases) noexcept
    {
        constexpr int vstep = 4;
        constexpr int count = cin / vstep;
        constexpr int remain = cin % vstep;

        std::memcpy(out, biases, sizeof(float) * cout);

        for (int idx = 0; idx < count; idx++)
        {
            float32x4_t r0 = vld1q_f32(rptr[0] + idx * vstep);
            float32x4_t r1 = vld1q_f32(rptr[1] + idx * vstep);
            float32x4_t r2 = vld1q_f32(rptr[2] + idx * vstep);
            float32x4_t r3 = vld1q_f32(rptr[3] + idx * vstep);
            float32x4_t r4 = vld1q_f32(rptr[4] + idx * vstep);
            float32x4_t r5 = vld1q_f32(rptr[5] + idx * vstep);
            float32x4_t r6 = vld1q_f32(rptr[6] + idx * vstep);
            float32x4_t r7 = vld1q_f32(rptr[7] + idx * vstep);
            float32x4_t r8 = vld1q_f32(rptr[8] + idx * vstep);

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

                float32x4_t k0 = vld1q_f32(kptr[0] + idx * vstep);
                float32x4_t k1 = vld1q_f32(kptr[1] + idx * vstep);
                float32x4_t k2 = vld1q_f32(kptr[2] + idx * vstep);
                float32x4_t k3 = vld1q_f32(kptr[3] + idx * vstep);
                float32x4_t k4 = vld1q_f32(kptr[4] + idx * vstep);
                float32x4_t k5 = vld1q_f32(kptr[5] + idx * vstep);
                float32x4_t k6 = vld1q_f32(kptr[6] + idx * vstep);
                float32x4_t k7 = vld1q_f32(kptr[7] + idx * vstep);
                float32x4_t k8 = vld1q_f32(kptr[8] + idx * vstep);

                s0 = vmlaq_f32(s0, r0, k0);
                s1 = vmlaq_f32(s1, r1, k1);
                s2 = vmlaq_f32(s2, r2, k2);
                s0 = vmlaq_f32(s0, r3, k3);
                s1 = vmlaq_f32(s1, r4, k4);
                s2 = vmlaq_f32(s2, r5, k5);
                s0 = vmlaq_f32(s0, r6, k6);
                s1 = vmlaq_f32(s1, r7, k7);
                s2 = vmlaq_f32(s2, r8, k8);

                out[n] += neon_hsum_f32(vaddq_f32(s0, vaddq_f32(s1, s2)));
            }
        }
        if constexpr (remain)
        {
            const float rd0[vstep] = {(rptr[0] + count * vstep)[0], remain > 1 ? (rptr[0] + count * vstep)[1] : 0.0f, remain > 2 ? (rptr[0] + count * vstep)[2] : 0.0f, 0.0f};
            const float rd1[vstep] = {(rptr[1] + count * vstep)[0], remain > 1 ? (rptr[1] + count * vstep)[1] : 0.0f, remain > 2 ? (rptr[1] + count * vstep)[2] : 0.0f, 0.0f};
            const float rd2[vstep] = {(rptr[2] + count * vstep)[0], remain > 1 ? (rptr[2] + count * vstep)[1] : 0.0f, remain > 2 ? (rptr[2] + count * vstep)[2] : 0.0f, 0.0f};
            const float rd3[vstep] = {(rptr[3] + count * vstep)[0], remain > 1 ? (rptr[3] + count * vstep)[1] : 0.0f, remain > 2 ? (rptr[3] + count * vstep)[2] : 0.0f, 0.0f};
            const float rd4[vstep] = {(rptr[4] + count * vstep)[0], remain > 1 ? (rptr[4] + count * vstep)[1] : 0.0f, remain > 2 ? (rptr[4] + count * vstep)[2] : 0.0f, 0.0f};
            const float rd5[vstep] = {(rptr[5] + count * vstep)[0], remain > 1 ? (rptr[5] + count * vstep)[1] : 0.0f, remain > 2 ? (rptr[5] + count * vstep)[2] : 0.0f, 0.0f};
            const float rd6[vstep] = {(rptr[6] + count * vstep)[0], remain > 1 ? (rptr[6] + count * vstep)[1] : 0.0f, remain > 2 ? (rptr[6] + count * vstep)[2] : 0.0f, 0.0f};
            const float rd7[vstep] = {(rptr[7] + count * vstep)[0], remain > 1 ? (rptr[7] + count * vstep)[1] : 0.0f, remain > 2 ? (rptr[7] + count * vstep)[2] : 0.0f, 0.0f};
            const float rd8[vstep] = {(rptr[8] + count * vstep)[0], remain > 1 ? (rptr[8] + count * vstep)[1] : 0.0f, remain > 2 ? (rptr[8] + count * vstep)[2] : 0.0f, 0.0f};
            float32x4_t r0 = vld1q_f32(rd0);
            float32x4_t r1 = vld1q_f32(rd1);
            float32x4_t r2 = vld1q_f32(rd2);
            float32x4_t r3 = vld1q_f32(rd3);
            float32x4_t r4 = vld1q_f32(rd4);
            float32x4_t r5 = vld1q_f32(rd5);
            float32x4_t r6 = vld1q_f32(rd6);
            float32x4_t r7 = vld1q_f32(rd7);
            float32x4_t r8 = vld1q_f32(rd8);

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

                const float kd0[vstep] = {(kptr[0] + count * vstep)[0], remain > 1 ? (kptr[0] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[0] + count * vstep)[2] : 0.0f, 0.0f};
                const float kd1[vstep] = {(kptr[1] + count * vstep)[0], remain > 1 ? (kptr[1] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[1] + count * vstep)[2] : 0.0f, 0.0f};
                const float kd2[vstep] = {(kptr[2] + count * vstep)[0], remain > 1 ? (kptr[2] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[2] + count * vstep)[2] : 0.0f, 0.0f};
                const float kd3[vstep] = {(kptr[3] + count * vstep)[0], remain > 1 ? (kptr[3] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[3] + count * vstep)[2] : 0.0f, 0.0f};
                const float kd4[vstep] = {(kptr[4] + count * vstep)[0], remain > 1 ? (kptr[4] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[4] + count * vstep)[2] : 0.0f, 0.0f};
                const float kd5[vstep] = {(kptr[5] + count * vstep)[0], remain > 1 ? (kptr[5] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[5] + count * vstep)[2] : 0.0f, 0.0f};
                const float kd6[vstep] = {(kptr[6] + count * vstep)[0], remain > 1 ? (kptr[6] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[6] + count * vstep)[2] : 0.0f, 0.0f};
                const float kd7[vstep] = {(kptr[7] + count * vstep)[0], remain > 1 ? (kptr[7] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[7] + count * vstep)[2] : 0.0f, 0.0f};
                const float kd8[vstep] = {(kptr[8] + count * vstep)[0], remain > 1 ? (kptr[8] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[8] + count * vstep)[2] : 0.0f, 0.0f};
                float32x4_t k0 = vld1q_f32(kd0);
                float32x4_t k1 = vld1q_f32(kd1);
                float32x4_t k2 = vld1q_f32(kd2);
                float32x4_t k3 = vld1q_f32(kd3);
                float32x4_t k4 = vld1q_f32(kd4);
                float32x4_t k5 = vld1q_f32(kd5);
                float32x4_t k6 = vld1q_f32(kd6);
                float32x4_t k7 = vld1q_f32(kd7);
                float32x4_t k8 = vld1q_f32(kd8);

                s0 = vmlaq_f32(s0, r0, k0);
                s1 = vmlaq_f32(s1, r1, k1);
                s2 = vmlaq_f32(s2, r2, k2);
                s0 = vmlaq_f32(s0, r3, k3);
                s1 = vmlaq_f32(s1, r4, k4);
                s2 = vmlaq_f32(s2, r5, k5);
                s0 = vmlaq_f32(s0, r6, k6);
                s1 = vmlaq_f32(s1, r7, k7);
                s2 = vmlaq_f32(s2, r8, k8);

                out[n] += neon_hsum_f32(vaddq_f32(s0, vaddq_f32(s1, s2)));
            }
        }
    }

    template <int cin, int cout, bool postactive = false, typename ActiveFunc, typename... ResidualArgs>
    inline void conv1x1_neon_float(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc, ResidualArgs&& ...residualArg)
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

                conv1x1_neon_float_impl<cin, cout>(rptr, sum, kernels, biases);

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

                conv3x3_neon_float_impl<cin, cout>(rptr, sum, kernels, biases);

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
                    out[n] = activeFunc(sum + r8 * k8 + biases[n], n);
                }
            }
        });
    }
    template <typename IN, int cout, typename ActiveFunc>
    inline void conv5x5_neon_cin1(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc)
    {
        util::parallelFor(0, src.height(), [&](const int i) {
            int ioffsets[5] = { i > 1 ? -2 : (i > 0 ? -1 : 0) , i > 0 ? -1 : 0 , 0, i < src.height() - 1 ? 1 : 0, i < src.height() - 2 ? 2 : (i < src.height() - 1 ? 1 : 0)};

            for (int j = 0; j < src.width(); j++)
            {
                auto out = static_cast<float*>(dst.ptr(j, i));

                int joffsets[5] = { j > 1 ? -2 : (j > 0 ? -1 : 0), j > 0 ? -1 : 0, 0, j < src.width() - 1 ? 1 : 0 ,j < src.width() - 2 ? 2 : (j < src.width() - 1 ? 1 : 0)};

                float d[25]{};
                for (int in = 0; in < 5; in++)
                    for (int jn = 0; jn < 5; jn++)
                        d[in * 5 + jn] = toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[jn], i + ioffsets[in])));

                float32x4_t r0 = vld1q_f32(d + 0);
                float32x4_t r4 = vld1q_f32(d + 4);
                float32x4_t r8 = vld1q_f32(d + 8);
                float32x4_t r12 = vld1q_f32(d + 12);
                float32x4_t r16 = vld1q_f32(d + 16);
                float32x4_t r20 = vld1q_f32(d + 20);
                auto r24 = d[24];

                for (int n = 0; n < cout; n++)
                {
                    float32x4_t k0  = vld1q_f32(kernels + n * 25 + 0);
                    float32x4_t k4  = vld1q_f32(kernels + n * 25 + 4);
                    float32x4_t k8  = vld1q_f32(kernels + n * 25 + 8);
                    float32x4_t k12 = vld1q_f32(kernels + n * 25 + 12);
                    float32x4_t k16 = vld1q_f32(kernels + n * 25 + 16);
                    float32x4_t k20 = vld1q_f32(kernels + n * 25 + 20);

                    float32x4_t s0 = vdupq_n_f32(0.0f);
                    float32x4_t s1 = vdupq_n_f32(0.0f);
                    float32x4_t s2 = vdupq_n_f32(0.0f);

                    s0 = vmlaq_f32(s0, r0, k0);
                    s1 = vmlaq_f32(s1, r4, k4);
                    s2 = vmlaq_f32(s2, r8, k8);
                    s0 = vmlaq_f32(s0, r12, k12);
                    s1 = vmlaq_f32(s1, r16, k16);
                    s2 = vmlaq_f32(s2, r20, k20);

                    auto sum = neon_hsum_f32(vaddq_f32(s0, vaddq_f32(s1, s2)));

                    auto k24 = *(kernels + n * 25 + 24);
                    out[n] = activeFunc(sum + r24 * k24 + biases[n], n);
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

    template <typename OUT, int cin, int upscale>
    inline void conv3x3_identity_pixelshuffle_neon_float(const Image& src, Image& dst, const float* const kernels, const float* const biases) noexcept
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

                conv3x3_neon_float_impl<cin, cout>(rptr, sum, kernels, biases);

                for (int n = 0; n < cout; n++) *static_cast<OUT*>(dst.ptr(dstX + (n & 1), dstY + (n >> 1))) = fromFloat<OUT>(sum[n]);
            }
        });
    }

    template <int cin, int ctemp, int cout, bool postactive3x3 = false, bool postactive1x1 = false, typename ActiveFunc3x3, typename ResidualArg3x3, typename ActiveFunc1x1, typename ResidualArg1x1>
    inline void conv3x3_conv1x1_neon_float(
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

                conv3x3_neon_float_impl<cin, ctemp>(rptr, buffer, kernels3x3, biases3x3);

                for (int n = 0; n < ctemp; n++)
                {
                    if constexpr (!postactive3x3) buffer[n] = activeFunc3x3(buffer[n], n);

                    if constexpr (std::is_same_v<ResidualArg3x3, ResidualArg>)
                        buffer[n] = buffer[n] * residualArg3x3.scale + static_cast<const float*>(residualArg3x3.image.ptr(j, i))[n];

                    if constexpr (postactive3x3) buffer[n] = activeFunc3x3(buffer[n], n);
                }

                rptr[0] = buffer;
                float sum[cout]{};
                conv1x1_neon_float_impl<ctemp, cout>(rptr, sum, kernels1x1, biases1x1);

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
    void conv3x3_8to8_identity_residual_neon(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale)
    {
        conv3x3_neon_float<8, 8>(src, dst, kernels, biases, Identity(), ResidualArg{ id, scale });
    }
    void conv3x3_8to8_identity_residual_add_neon(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale, const Image& feat)
    {
        conv3x3_neon_float<8, 8>(src, dst, kernels, biases, Identity(), ResidualArg{ id, scale }, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_neon_float<std::uint8_t, 8, 2>(src, dst, kernels, biases);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_neon_float<std::uint16_t, 8, 2>(src, dst, kernels, biases);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_neon_float<float, 8, 2>(src, dst, kernels, biases);
            break;
        }
    }
    void conv3x3_8to4_identity_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_neon_float<8, 4>(src, dst, kernels, biases, Identity());
    }

    void conv3x3_1to16_identity_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_neon_cin1<std::uint8_t, 16>(src, dst, kernels, biases, Identity());
            break;
        case Image::UInt16:
            conv3x3_neon_cin1<std::uint16_t, 16>(src, dst, kernels, biases, Identity());
            break;
        case Image::Float32:
            conv3x3_neon_cin1<float, 16>(src, dst, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_16to16_relu_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_neon_float<16, 16>(src, dst, kernels, biases, ReLU());
    }
    void conv3x3_16to16_identity_add_neon(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
        conv3x3_neon_float<16, 16>(src, dst, kernels, biases, Identity(), ResidualArg{ feat, 1.0f });
    }
    void conv3x3_16to4_identity_pixelshuffle_4to1_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_neon_float<std::uint8_t, 16, 2>(src, dst, kernels, biases);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_neon_float<std::uint16_t, 16, 2>(src, dst, kernels, biases);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_neon_float<float, 16, 2>(src, dst, kernels, biases);
            break;
        }
    }
    void conv3x3_16to4_identity_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_neon_float<16, 4>(src, dst, kernels, biases, Identity());
    }

    void conv3x3_1to32_identity_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_neon_cin1<std::uint8_t, 32>(src, dst, kernels, biases, Identity());
            break;
        case Image::UInt16:
            conv3x3_neon_cin1<std::uint16_t, 32>(src, dst, kernels, biases, Identity());
            break;
        case Image::Float32:
            conv3x3_neon_cin1<float, 32>(src, dst, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_32to32_relu_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_neon_float<32, 32>(src, dst, kernels, biases, ReLU());
    }
    void conv3x3_32to32_identity_add_neon(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
        conv3x3_neon_float<32, 32>(src, dst, kernels, biases, Identity(), ResidualArg{ feat, 1.0f });
    }
    void conv3x3_32to4_identity_pixelshuffle_4to1_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_neon_float<std::uint8_t, 32, 2>(src, dst, kernels, biases);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_neon_float<std::uint16_t, 32, 2>(src, dst, kernels, biases);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_neon_float<float, 32, 2>(src, dst, kernels, biases);
            break;
        }
    }
    void conv3x3_32to4_identity_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_neon_float<32, 4>(src, dst, kernels, biases, Identity());
    }

    void conv5x5_1to8_identity_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv5x5_neon_cin1<std::uint8_t, 8>(src, dst, kernels, biases, Identity());
            break;
        case Image::UInt16:
            conv5x5_neon_cin1<std::uint16_t, 8>(src, dst, kernels, biases, Identity());
            break;
        case Image::Float32:
            conv5x5_neon_cin1<float, 8>(src, dst, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_8to8_prelu_neon(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        conv3x3_neon_float<8, 8>(src, dst, kernels, biases, PReLU(alphas));
    }
    void conv3x3_8to8_prelu_conv1x1_8to8_add_prelu_neon(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_neon_float<8, 8, 8, false, true>(
            src, dst,
            kernels1, biases1, PReLU(alphas1), nullptr,
            kernels2, biases2, PReLU(alphas2), ResidualArg{ feat, 1.0f }
        );
    }

    void conv5x5_1to16_identity_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv5x5_neon_cin1<std::uint8_t, 16>(src, dst, kernels, biases, Identity());
            break;
        case Image::UInt16:
            conv5x5_neon_cin1<std::uint16_t, 16>(src, dst, kernels, biases, Identity());
            break;
        case Image::Float32:
            conv5x5_neon_cin1<float, 16>(src, dst, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_16to16_prelu_neon(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        conv3x3_neon_float<16, 16>(src, dst, kernels, biases, PReLU(alphas));
    }
    void conv3x3_16to16_prelu_conv1x1_16to16_add_prelu_neon(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_neon_float<16, 16, 16, false, true>(
            src, dst,
            kernels1, biases1, PReLU(alphas1), nullptr,
            kernels2, biases2, PReLU(alphas2), ResidualArg{ feat, 1.0f }
        );
    }
}
