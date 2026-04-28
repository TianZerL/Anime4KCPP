#include <arm_neon.h>

#include "AC/Core/Image.hpp"
#include "AC/Core/Util.hpp"

#include "AC/Core/Internal/Processor/CPU/Common.hpp"

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

    struct ConvImplNEON
    {
        template <int cout, int cpos>
        static void conv_cin1(const float rptr[], float* const out, const float* const kernels, const float* const biases) noexcept
        {
            constexpr int vstep = 4;
            constexpr int count = cpos / vstep;
            constexpr int remain = cpos % vstep;

            float32x4_t r[count];
            for (int idx = 0; idx < count; idx++) r[idx] = vld1q_f32(rptr + idx * vstep);

            for (int n = 0; n < cout; n++)
            {
                float32x4_t s = vdupq_n_f32(0.0f);
                auto kptr = kernels + n * cpos;
                for (int idx = 0; idx < count; idx++)
                {
                    float32x4_t k = vld1q_f32(kptr + idx * vstep);
                    s = vmlaq_f32(s, r[idx], k);
                }
                auto sum = neon_hsum_f32(s);

                for (int i = 0; i < remain; i++) sum += rptr[count * vstep + i] * kptr[count * vstep + i];

                out[n] = sum + biases[n];
            }
        }

        template <int cin, int cout, int cpos>
        static void conv(const float* rptr[], float* const out, const float* const kernels, const float* const biases) noexcept
        {
            constexpr int vstep = 4;
            constexpr int count = cin / vstep;
            constexpr int remain = cin % vstep;

            std::memcpy(out, biases, sizeof(float) * cout);

            float32x4_t s[cout];
            for (int n = 0; n < cout; n++) s[n] = vdupq_n_f32(0.0f);
            for (int p = 0; p < cpos; p++)
            {
                for (int idx = 0; idx < count; idx++)
                {
                    float32x4_t r = vld1q_f32(rptr[p] + idx * vstep);
                    for (int n = 0; n < cout; n++)
                    {
                        float32x4_t k = vld1q_f32(kernels + n * cin * cpos + cin * p + idx * vstep);
                        s[n] = vmlaq_f32(s[n], r, k);
                    }
                }
                if constexpr (remain)
                    for (int c = count * vstep; c < cin; c++)
                        for (int n = 0; n < cout; n++) out[n] += rptr[p][c] * kernels[n * cin * cpos + cin * p + c];
            }
            for (int n = 0; n < cout; n++) out[n] += neon_hsum_f32(s[n]);
        }
    };

    void conv3x3_1to8_relu_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<ConvImplNEON, std::uint8_t, 8>(src, dst, kernels, biases, ReLU{});
            break;
        case Image::UInt16:
            conv3x3_cin1<ConvImplNEON, std::uint16_t, 8>(src, dst, kernels, biases, ReLU{});
            break;
        case Image::Float32:
            conv3x3_cin1<ConvImplNEON, float, 8>(src, dst, kernels, biases, ReLU{});
            break;
        }
    }
    void conv3x3_8to8_relu_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_float<ConvImplNEON, 8, 8>(src, dst, kernels, biases, ReLU{});
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

    void conv3x3_1to8_prelu_neon(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<ConvImplNEON, std::uint8_t, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        case Image::UInt16:
            conv3x3_cin1<ConvImplNEON, std::uint16_t, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        case Image::Float32:
            conv3x3_cin1<ConvImplNEON, float, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        }
    }

    void conv3x3_1to8_identity_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<ConvImplNEON, std::uint8_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1<ConvImplNEON, std::uint16_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1<ConvImplNEON, float, 8>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_8to8_prelu_neon(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        conv3x3_float<ConvImplNEON, 8, 8>(src, dst, kernels, biases, PReLU{ alphas });
    }
    void conv3x3_8to8_identity_residual_neon(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale)
    {
        conv3x3_float<ConvImplNEON, 8, 8>(src, dst, kernels, biases, Identity{}, ResidualArg{ id, scale });
    }
    void conv3x3_8to8_identity_residual_conv1x1_8to8_prelu_add_neon(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1,
        const Image& id, const float scale,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_float<ConvImplNEON, 8, 8, 8, false, false>(
            src, dst,
            kernels1, biases1, Identity{}, ResidualArg{ id, scale },
            kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
        );
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_add_neon(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<ConvImplNEON, std::uint8_t, 8, 2>(src, dst, kernels, biases, ResidualArg{ id, 1.0f });
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<ConvImplNEON, std::uint16_t, 8, 2>(src, dst, kernels, biases, ResidualArg{ id, 1.0f });
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<ConvImplNEON, float, 8, 2>(src, dst, kernels, biases, ResidualArg{ id, 1.0f });
            break;
        }
    }

    void conv3x3_1to16_identity_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<ConvImplNEON, std::uint8_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1<ConvImplNEON, std::uint16_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1<ConvImplNEON, float, 16>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_16to16_relu_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_float<ConvImplNEON, 16, 16>(src, dst, kernels, biases, ReLU{});
    }
    void conv3x3_16to16_identity_add_neon(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
        conv3x3_float<ConvImplNEON, 16, 16>(src, dst, kernels, biases, Identity{}, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_16to4_identity_pixelshuffle_4to1_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<ConvImplNEON, std::uint8_t, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<ConvImplNEON, std::uint16_t, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<ConvImplNEON, float, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        }
    }

    void conv3x3_1to32_identity_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<ConvImplNEON, std::uint8_t, 32>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1<ConvImplNEON, std::uint16_t, 32>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1<ConvImplNEON, float, 32>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_32to32_relu_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_float<ConvImplNEON, 32, 32>(src, dst, kernels, biases, ReLU{});
    }
    void conv3x3_32to32_identity_add_neon(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
        conv3x3_float<ConvImplNEON, 32, 32>(src, dst, kernels, biases, Identity{}, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_32to4_identity_pixelshuffle_4to1_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<ConvImplNEON, std::uint8_t, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<ConvImplNEON, std::uint16_t, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<ConvImplNEON, float, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        }
    }

    void conv5x5_1to8_identity_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv5x5_cin1<ConvImplNEON, std::uint8_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv5x5_cin1<ConvImplNEON, std::uint16_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv5x5_cin1<ConvImplNEON, float, 8>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_8to8_prelu_conv1x1_8to8_add_prelu_neon(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_float<ConvImplNEON, 8, 8, 8, false, true>(
            src, dst,
            kernels1, biases1, PReLU{ alphas1 }, nullptr,
            kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
        );
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<ConvImplNEON, std::uint8_t, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<ConvImplNEON, std::uint16_t, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<ConvImplNEON, float, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        }
    }

    void conv5x5_1to16_identity_neon(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv5x5_cin1<ConvImplNEON, std::uint8_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv5x5_cin1<ConvImplNEON, std::uint16_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv5x5_cin1<ConvImplNEON, float, 16>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_16to16_prelu_neon(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        conv3x3_float<ConvImplNEON, 16, 16>(src, dst, kernels, biases, PReLU{ alphas });
    }
    void conv3x3_16to16_prelu_conv1x1_16to16_add_prelu_neon(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_float<ConvImplNEON, 16, 16, 16, false, true>(
            src, dst,
            kernels1, biases1, PReLU{ alphas1 }, nullptr,
            kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
        );
    }
}
