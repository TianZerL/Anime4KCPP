#include <array>
#include <type_traits>

#include "AC/Core/Image.hpp"
#include "AC/Core/Util.hpp"

#include "AC/Core/Internal/Processor/CPU/Common.hpp"

namespace ac::core::cpu
{
    template <typename IN, int cout, typename ActiveFunc>
    inline void conv3x3_cin1_generic(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc)
    {
        util::parallelFor(0, src.height(), [&](const int i) {
            auto tp = i > 0 ? 1 : 0;
            auto bp = i < src.height() - 1 ? 1 : 0;

            for (int j = 0; j < src.width(); j++)
            {
                auto out = static_cast<float*>(dst.ptr(j, i));

                auto lp = j > 0 ? 1 : 0;
                auto rp = j < src.width() - 1 ? 1 : 0;

                const float r[] = {
                    toFloat(*static_cast<const IN*>(src.ptr(j - lp, i - tp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j     , i - tp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + rp, i - tp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j - lp, i     ))),
                    toFloat(*static_cast<const IN*>(src.ptr(j     , i     ))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + rp, i     ))),
                    toFloat(*static_cast<const IN*>(src.ptr(j - lp, i + bp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j     , i + bp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + rp, i + bp)))
                };

                for (int n = 0; n < cout; n++)
                {
                    auto k = kernels + n * 9;
                    float sum = biases[n];
                    for (int idx = 0; idx < 9; idx++) sum += r[idx] * k[idx];
                    out[n] = activeFunc(sum, n);
                }
            }
        });
    }
    template <typename IN, int cout, typename ActiveFunc>
    inline void conv5x5_cin1_generic(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc)
    {
        util::parallelFor(0, src.height(), [&](const int i) {
            int ioffsets[5] = { i > 1 ? -2 : (i > 0 ? -1 : 0) , i > 0 ? -1 : 0 , 0, i < src.height() - 1 ? 1 : 0, i < src.height() - 2 ? 2 : (i < src.height() - 1 ? 1 : 0) };

            for (int j = 0; j < src.width(); j++)
            {
                auto out = static_cast<float*>(dst.ptr(j, i));

                int joffsets[5] = { j > 1 ? -2 : (j > 0 ? -1 : 0), j > 0 ? -1 : 0, 0, j < src.width() - 1 ? 1 : 0 ,j < src.width() - 2 ? 2 : (j < src.width() - 1 ? 1 : 0) };

                float r[25]{};
                for (int in = 0; in < 5; in++)
                    for (int jn = 0; jn < 5; jn++)
                        r[in * 5 + jn] = toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[jn], i + ioffsets[in])));

                for (int n = 0; n < cout; n++)
                {
                    auto k = kernels + n * 25;
                    float sum = biases[n];
                    for (int idx = 0; idx < 25; idx++) sum += r[idx] * k[idx];
                    out[n] = activeFunc(sum, n);
                }
            }
        });
    }

    template <typename IN, typename OUT, int cin, int cout>
    inline void deconv2x2_generic(const Image& src, Image& dst, const float* const kernels)
    {
        filter([=](const int i, const int j, const void* const sptr, void* const dptr) {
            auto in = static_cast<const IN*>(sptr);
            auto out = static_cast<OUT*>(dptr);

            auto index = ((i & 1) << 1) + (j & 1);

            for (int n = 0; n < cout; n++)
            {
                auto k = kernels + n * cin * 4 + cin * index;
                float sum = 0.0f;
                for (int c = 0; c < cin; c++) sum += toFloat(in[c]) * k[c];
                out[n] = fromFloat<OUT>(sum);
            }
        }, src, dst);
    }
    template <typename IN, typename OUT, int cin, int upscale>
    inline void pixelshuffle_generic(const Image& src, Image& dst) noexcept
    {
        constexpr int group = upscale * upscale;
        constexpr int cout = cin / group;
        static_assert(cin % group == 0 && cout > 0);

        for (int i = 0; i < dst.height(); i++)
        {
            for (int j = 0; j < dst.width(); j++)
            {
                auto in = static_cast<const IN*>(src.ptr(j / upscale, i / upscale));
                auto out = static_cast<OUT*>(dst.ptr(j, i));

                auto index = (i % upscale) * upscale + (j % upscale);

                for (int n = 0; n < cout; n++) out[n] = fromFloat<OUT>(toFloat(in[n * group + index]));
            }
        }
    }

    struct ConvImplGeneric
    {
        template <int cin, int cout>
        static void conv1x1(const float* rptr[], float* const out, const float* const kernels, const float* const biases) noexcept
        {
            auto r = rptr[0];
            for (int n = 0; n < cout; n++)
            {
                auto k = kernels + n * cin;

                float sum = biases[n];

                for (int c = 0; c < cin; c++) sum += r[c] * k[c];

                out[n] = sum;
            }
        }

        template <int cin, int cout>
        static void conv3x3(const float* rptr[], float* const out, const float* const kernels, const float* const biases) noexcept
        {
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

                float sum = biases[n];

                for (int c = 0; c < cin; c++)
                {
                    sum +=
                        rptr[0][c] * kptr[0][c] +
                        rptr[1][c] * kptr[1][c] +
                        rptr[2][c] * kptr[2][c] +
                        rptr[3][c] * kptr[3][c] +
                        rptr[4][c] * kptr[4][c] +
                        rptr[5][c] * kptr[5][c] +
                        rptr[6][c] * kptr[6][c] +
                        rptr[7][c] * kptr[7][c] +
                        rptr[8][c] * kptr[8][c];
                }

                out[n] = sum;
            }
        }
    };

    void conv3x3_1to8_relu_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1_generic<std::uint8_t, 8>(src, dst, kernels, biases, ReLU{});
            break;
        case Image::UInt16:
            conv3x3_cin1_generic<std::uint16_t, 8>(src, dst, kernels, biases, ReLU{});
            break;
        case Image::Float32:
            conv3x3_cin1_generic<float, 8>(src, dst, kernels, biases, ReLU{});
            break;
        }
    }
    void conv3x3_8to8_relu_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_float<ConvImplGeneric, 8, 8>(src, dst, kernels, biases, ReLU{});
    }
    void deconv2x2_8to1_generic(const Image& src, Image& dst, const float* kernels)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            deconv2x2_generic<float, std::uint8_t, 8, 1>(src, dst, kernels);
            break;
        case Image::UInt16:
            deconv2x2_generic<float, std::uint16_t, 8, 1>(src, dst, kernels);
            break;
        case Image::Float32:
            deconv2x2_generic<float, float, 8, 1>(src, dst, kernels);
            break;
        }
    }

    void conv3x3_1to8_prelu_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1_generic<std::uint8_t, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        case Image::UInt16:
            conv3x3_cin1_generic<std::uint16_t, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        case Image::Float32:
            conv3x3_cin1_generic<float, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        }
    }

    void conv3x3_1to8_identity_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1_generic<std::uint8_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1_generic<std::uint16_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1_generic<float, 8>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_8to8_prelu_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        conv3x3_float<ConvImplGeneric, 8, 8>(src, dst, kernels, biases, PReLU{ alphas });
    }
    void conv3x3_8to8_identity_residual_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale)
    {
        conv3x3_float<ConvImplGeneric, 8, 8>(src, dst, kernels, biases, Identity{}, ResidualArg{ id, scale });
    }
    void conv3x3_8to8_identity_residual_conv1x1_8to8_prelu_add_generic(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1,
        const Image& id, const float scale,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_float<ConvImplGeneric, 8, 8, 8, false, false>(
            src, dst,
            kernels1, biases1, Identity{}, ResidualArg{ id, scale },
            kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
        );
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_add_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<ConvImplGeneric, std::uint8_t, 8, 2>(src, dst, kernels, biases, ResidualArg{id, 1.0f});
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<ConvImplGeneric, std::uint16_t, 8, 2>(src, dst, kernels, biases, ResidualArg{id, 1.0f});
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<ConvImplGeneric, float, 8, 2>(src, dst, kernels, biases, ResidualArg{id, 1.0f});
            break;
        }
    }

    void conv3x3_1to16_identity_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1_generic<std::uint8_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1_generic<std::uint16_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1_generic<float, 16>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_16to16_relu_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_float<ConvImplGeneric, 16, 16>(src, dst, kernels, biases, ReLU{});
    }
    void conv3x3_16to16_identity_add_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
        conv3x3_float<ConvImplGeneric, 16, 16>(src, dst, kernels, biases, Identity{}, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_16to4_identity_pixelshuffle_4to1_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<ConvImplGeneric, std::uint8_t, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<ConvImplGeneric, std::uint16_t, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<ConvImplGeneric, float, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        }
    }

    void conv3x3_1to32_identity_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1_generic<std::uint8_t, 32>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1_generic<std::uint16_t, 32>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1_generic<float, 32>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_32to32_relu_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_float<ConvImplGeneric, 32, 32>(src, dst, kernels, biases, ReLU{});
    }
    void conv3x3_32to32_identity_add_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
        conv3x3_float<ConvImplGeneric, 32, 32>(src, dst, kernels, biases, Identity{}, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_32to4_identity_pixelshuffle_4to1_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<ConvImplGeneric, std::uint8_t, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<ConvImplGeneric, std::uint16_t, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<ConvImplGeneric, float, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        }
    }

    void conv5x5_1to8_identity_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv5x5_cin1_generic<std::uint8_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv5x5_cin1_generic<std::uint16_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv5x5_cin1_generic<float, 8>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_8to8_prelu_conv1x1_8to8_add_prelu_generic(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_float<ConvImplGeneric, 8, 8, 8, false, true>(
            src, dst,
            kernels1, biases1, PReLU{ alphas1 }, nullptr,
            kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
        );
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<ConvImplGeneric, std::uint8_t, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<ConvImplGeneric, std::uint16_t, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<ConvImplGeneric, float, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        }
    }

    void conv5x5_1to16_identity_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv5x5_cin1_generic<std::uint8_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv5x5_cin1_generic<std::uint16_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv5x5_cin1_generic<float, 16>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_16to16_prelu_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        conv3x3_float<ConvImplGeneric, 16, 16>(src, dst, kernels, biases, PReLU{ alphas });
    }
    void conv3x3_16to16_prelu_conv1x1_16to16_add_prelu_generic(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_float<ConvImplGeneric, 16, 16, 16, false, true>(
            src, dst,
            kernels1, biases1, PReLU{ alphas1 }, nullptr,
            kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
        );
    }

    void pixelshuffle_4to1_generic(const Image& src, Image& dst)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            pixelshuffle_generic<float, std::uint8_t, 4, 2>(src, dst);
            break;
        case Image::UInt16:
            pixelshuffle_generic<float, std::uint16_t, 4, 2>(src, dst);
            break;
        case Image::Float32:
            pixelshuffle_generic<float, float, 4, 2>(src, dst);
            break;
        }
    }
}
