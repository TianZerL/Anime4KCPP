#include <array>
#include <type_traits>

#include "AC/Core/Image.hpp"
#include "AC/Core/Util.hpp"

namespace ac::core::cpu
{
    template <typename IN, int cin, int cout, bool postactive = false, typename ActiveFunc, typename... ResidualArgs>
    inline void conv1x1_generic(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc, ResidualArgs&& ...residualArg)
    {
        [[maybe_unused]] const std::array<float, sizeof...(ResidualArgs)> scales{ residualArg.scale... };

        util::parallelFor(0, src.height(), [&](const int i) {
            for (int j = 0; j < src.width(); j++)
            {
                [[maybe_unused]] const std::array<const float*, sizeof...(ResidualArgs)> iptrs{ static_cast<const float*>(residualArg.image.ptr(j, i))... };

                auto out = static_cast<float*>(dst.ptr(j, i));

                auto r = static_cast<const IN*>(src.ptr(j, i));

                for (int n = 0; n < cout; n++)
                {
                    auto k = kernels + n * cin;

                    float sum = biases[n];

                    for (int c = 0; c < cin; c++) sum += toFloat(r[c]) * k[c];

                    if constexpr (!postactive) sum = activeFunc(sum, n);

                    if constexpr (sizeof...(ResidualArgs))
                        for (int idx = 0; idx < sizeof...(ResidualArgs); idx++)
                            sum = sum * scales[idx] + iptrs[idx][n];

                    if constexpr (postactive) sum = activeFunc(sum, n);

                    out[n] = sum;
                }
            }
        });
    }
    template <typename IN, int cin, int cout, bool postactive = false, typename ActiveFunc, typename... ResidualArgs>
    inline void conv3x3_generic(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc, ResidualArgs&& ...residualArg)
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

                auto r0 = static_cast<const IN*>(src.ptr(j - lp, i - tp));
                auto r1 = static_cast<const IN*>(src.ptr(j     , i - tp));
                auto r2 = static_cast<const IN*>(src.ptr(j + rp, i - tp));
                auto r3 = static_cast<const IN*>(src.ptr(j - lp, i     ));
                auto r4 = static_cast<const IN*>(src.ptr(j     , i     ));
                auto r5 = static_cast<const IN*>(src.ptr(j + rp, i     ));
                auto r6 = static_cast<const IN*>(src.ptr(j - lp, i + bp));
                auto r7 = static_cast<const IN*>(src.ptr(j     , i + bp));
                auto r8 = static_cast<const IN*>(src.ptr(j + rp, i + bp));

                for (int n = 0; n < cout; n++)
                {
                    auto k0 = kernels + n * cin * 9 + cin * 0;
                    auto k1 = kernels + n * cin * 9 + cin * 1;
                    auto k2 = kernels + n * cin * 9 + cin * 2;
                    auto k3 = kernels + n * cin * 9 + cin * 3;
                    auto k4 = kernels + n * cin * 9 + cin * 4;
                    auto k5 = kernels + n * cin * 9 + cin * 5;
                    auto k6 = kernels + n * cin * 9 + cin * 6;
                    auto k7 = kernels + n * cin * 9 + cin * 7;
                    auto k8 = kernels + n * cin * 9 + cin * 8;

                    float sum = biases[n];

                    for (int c = 0; c < cin; c++)
                    {
                        sum +=
                            toFloat(r0[c]) * k0[c] +
                            toFloat(r1[c]) * k1[c] +
                            toFloat(r2[c]) * k2[c] +
                            toFloat(r3[c]) * k3[c] +
                            toFloat(r4[c]) * k4[c] +
                            toFloat(r5[c]) * k5[c] +
                            toFloat(r6[c]) * k6[c] +
                            toFloat(r7[c]) * k7[c] +
                            toFloat(r8[c]) * k8[c];
                    }

                    if constexpr (!postactive) sum = activeFunc(sum, n);

                    if constexpr (sizeof...(ResidualArgs))
                        for (int idx = 0; idx < sizeof...(ResidualArgs); idx++)
                            sum = sum * scales[idx] + iptrs[idx][n];

                    if constexpr (postactive) sum = activeFunc(sum, n);

                    out[n] = sum;
                }
            }
        });
    }
    template <typename IN, int cin, int cout, bool postactive = false, typename ActiveFunc, typename... ResidualArgs>
    inline void conv5x5_generic(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc, ResidualArgs&& ...residualArg)
    {
        [[maybe_unused]] const std::array<float, sizeof...(ResidualArgs)> scales{ residualArg.scale... };

        util::parallelFor(0, src.height(), [&](const int i) {
            int ioffsets[5] = { i > 1 ? -2 : (i > 0 ? -1 : 0) , i > 0 ? -1 : 0 , 0, i < src.height() - 1 ? 1 : 0, i < src.height() - 2 ? 2 : (i < src.height() - 1 ? 1 : 0)};

            for (int j = 0; j < src.width(); j++)
            {
                [[maybe_unused]] const std::array<const float*, sizeof...(ResidualArgs)> iptrs{ static_cast<const float*>(residualArg.image.ptr(j, i))... };

                auto out = static_cast<float*>(dst.ptr(j, i));

                int joffsets[5] = { j > 1 ? -2 : (j > 0 ? -1 : 0), j > 0 ? -1 : 0, 0, j < src.width() - 1 ? 1 : 0 ,j < src.width() - 2 ? 2 : (j < src.width() - 1 ? 1 : 0)};

                for (int n = 0; n < cout; n++)
                {
                    float sum = biases[n];

                    for (int in = 0; in < 5; in++)
                    {
                        for (int jn = 0; jn < 5; jn++)
                        {
                            auto r = static_cast<const IN*>(src.ptr(j + joffsets[jn], i + ioffsets[in]));
                            auto k = kernels + n * cin * 25 + cin * (in * 5 + jn);
                            for (int c = 0; c < cin; c++) sum += toFloat(r[c]) * k[c];
                        }
                    }

                    if constexpr (!postactive) sum = activeFunc(sum, n);

                    if constexpr (sizeof...(ResidualArgs))
                        for (int idx = 0; idx < sizeof...(ResidualArgs); idx++)
                            sum = sum * scales[idx] + iptrs[idx][n];

                    if constexpr (postactive) sum = activeFunc(sum, n);

                    out[n] = sum;
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

    template <typename IN, typename OUT, int cin, int upscale>
    inline void conv3x3_identity_pixelshuffle_generic(const Image& src, Image& dst, const float* const kernels, const float* const biases) noexcept
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

                auto r0 = static_cast<const IN*>(src.ptr(j - lp, i - tp));
                auto r1 = static_cast<const IN*>(src.ptr(j     , i - tp));
                auto r2 = static_cast<const IN*>(src.ptr(j + rp, i - tp));
                auto r3 = static_cast<const IN*>(src.ptr(j - lp, i     ));
                auto r4 = static_cast<const IN*>(src.ptr(j     , i     ));
                auto r5 = static_cast<const IN*>(src.ptr(j + rp, i     ));
                auto r6 = static_cast<const IN*>(src.ptr(j - lp, i + bp));
                auto r7 = static_cast<const IN*>(src.ptr(j     , i + bp));
                auto r8 = static_cast<const IN*>(src.ptr(j + rp, i + bp));

                for (int n = 0; n < cout; n++)
                {
                    auto k0 = kernels + n * cin * 9 + cin * 0;
                    auto k1 = kernels + n * cin * 9 + cin * 1;
                    auto k2 = kernels + n * cin * 9 + cin * 2;
                    auto k3 = kernels + n * cin * 9 + cin * 3;
                    auto k4 = kernels + n * cin * 9 + cin * 4;
                    auto k5 = kernels + n * cin * 9 + cin * 5;
                    auto k6 = kernels + n * cin * 9 + cin * 6;
                    auto k7 = kernels + n * cin * 9 + cin * 7;
                    auto k8 = kernels + n * cin * 9 + cin * 8;

                    float sum = biases[n];

                    for (int c = 0; c < cin; c++)
                    {
                        sum +=
                            toFloat(r0[c]) * k0[c] +
                            toFloat(r1[c]) * k1[c] +
                            toFloat(r2[c]) * k2[c] +
                            toFloat(r3[c]) * k3[c] +
                            toFloat(r4[c]) * k4[c] +
                            toFloat(r5[c]) * k5[c] +
                            toFloat(r6[c]) * k6[c] +
                            toFloat(r7[c]) * k7[c] +
                            toFloat(r8[c]) * k8[c];
                    }

                    *static_cast<OUT*>(dst.ptr(dstX + (n & 1), dstY + (n >> 1))) = fromFloat<OUT>(sum);
                }
            }
        });
    }

    template <typename IN, int cin, int ctemp, int cout, bool postactive3x3 = false, bool postactive1x1 = false, typename ActiveFunc3x3, typename ResidualArg3x3, typename ActiveFunc1x1, typename ResidualArg1x1>
    inline void conv3x3_conv1x1_generic(
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

                float buffer[ctemp]{};

                auto lp = j > 0 ? 1 : 0;
                auto rp = j < src.width() - 1 ? 1 : 0;

                auto r0 = static_cast<const IN*>(src.ptr(j - lp, i - tp));
                auto r1 = static_cast<const IN*>(src.ptr(j     , i - tp));
                auto r2 = static_cast<const IN*>(src.ptr(j + rp, i - tp));
                auto r3 = static_cast<const IN*>(src.ptr(j - lp, i     ));
                auto r4 = static_cast<const IN*>(src.ptr(j     , i     ));
                auto r5 = static_cast<const IN*>(src.ptr(j + rp, i     ));
                auto r6 = static_cast<const IN*>(src.ptr(j - lp, i + bp));
                auto r7 = static_cast<const IN*>(src.ptr(j     , i + bp));
                auto r8 = static_cast<const IN*>(src.ptr(j + rp, i + bp));

                for (int n = 0; n < ctemp; n++)
                {
                    auto k0 = kernels3x3 + n * cin * 9 + cin * 0;
                    auto k1 = kernels3x3 + n * cin * 9 + cin * 1;
                    auto k2 = kernels3x3 + n * cin * 9 + cin * 2;
                    auto k3 = kernels3x3 + n * cin * 9 + cin * 3;
                    auto k4 = kernels3x3 + n * cin * 9 + cin * 4;
                    auto k5 = kernels3x3 + n * cin * 9 + cin * 5;
                    auto k6 = kernels3x3 + n * cin * 9 + cin * 6;
                    auto k7 = kernels3x3 + n * cin * 9 + cin * 7;
                    auto k8 = kernels3x3 + n * cin * 9 + cin * 8;

                    float sum = biases3x3[n];

                    for (int c = 0; c < cin; c++)
                    {
                        sum +=
                            toFloat(r0[c]) * k0[c] +
                            toFloat(r1[c]) * k1[c] +
                            toFloat(r2[c]) * k2[c] +
                            toFloat(r3[c]) * k3[c] +
                            toFloat(r4[c]) * k4[c] +
                            toFloat(r5[c]) * k5[c] +
                            toFloat(r6[c]) * k6[c] +
                            toFloat(r7[c]) * k7[c] +
                            toFloat(r8[c]) * k8[c];
                    }

                    if constexpr (!postactive3x3) sum = activeFunc3x3(sum, n);

                    if constexpr (std::is_same_v<ResidualArg3x3, ResidualArg>)
                        sum = sum * residualArg3x3.scale + static_cast<const float*>(residualArg3x3.image.ptr(j, i))[n];

                    if constexpr (postactive3x3) sum = activeFunc3x3(sum, n);

                    buffer[n] = sum;
                }

                for (int n = 0; n < cout; n++)
                {
                    auto k = kernels1x1 + n * ctemp;

                    float sum = biases1x1[n];

                    for (int c = 0; c < ctemp; c++) sum += buffer[c] * k[c];

                    if constexpr (!postactive1x1) sum = activeFunc1x1(sum, n);

                    if constexpr (std::is_same_v<ResidualArg1x1, ResidualArg>)
                        sum = sum * residualArg1x1.scale + static_cast<const float*>(residualArg1x1.image.ptr(j, i))[n];

                    if constexpr (postactive1x1) sum = activeFunc1x1(sum, n);

                    out[n] = sum;
                }
            }
        });
    }

    void conv3x3_1to8_relu_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_generic<std::uint8_t, 1, 8>(src, dst, kernels, biases, ReLU());
            break;
        case Image::UInt16:
            conv3x3_generic<std::uint16_t, 1, 8>(src, dst, kernels, biases, ReLU());
            break;
        case Image::Float32:
            conv3x3_generic<float, 1, 8>(src, dst, kernels, biases, ReLU());
            break;
        }
    }
    void conv3x3_8to8_relu_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_generic<float, 8, 8>(src, dst, kernels, biases, ReLU());
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

    void conv3x3_1to8_identity_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_generic<std::uint8_t, 1, 8>(src, dst, kernels, biases, Identity());
            break;
        case Image::UInt16:
            conv3x3_generic<std::uint16_t, 1, 8>(src, dst, kernels, biases, Identity());
            break;
        case Image::Float32:
            conv3x3_generic<float, 1, 8>(src, dst, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_8to8_lrelu_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const float negativeSlope)
    {
        conv3x3_generic<float, 8, 8>(src, dst, kernels, biases, LReLU(negativeSlope));
    }
    void conv3x3_8to8_identity_residual_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale)
    {
        conv3x3_generic<float, 8, 8>(src, dst, kernels, biases, Identity(), ResidualArg{ id, scale });
    }
    void conv3x3_8to8_identity_residual_add_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale, const Image& feat)
    {
        conv3x3_generic<float, 8, 8>(src, dst, kernels, biases, Identity(), ResidualArg{ id, scale }, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_generic<float, std::uint8_t, 8, 2>(src, dst, kernels, biases);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_generic<float, std::uint16_t, 8, 2>(src, dst, kernels, biases);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_generic<float, float, 8, 2>(src, dst, kernels, biases);
            break;
        }
    }
    void conv3x3_8to4_identity_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_generic<float, 8, 4>(src, dst, kernels, biases, Identity());
    }

    void conv3x3_1to16_identity_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_generic<std::uint8_t, 1, 16>(src, dst, kernels, biases, Identity());
            break;
        case Image::UInt16:
            conv3x3_generic<std::uint16_t, 1, 16>(src, dst, kernels, biases, Identity());
            break;
        case Image::Float32:
            conv3x3_generic<float, 1, 16>(src, dst, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_16to16_relu_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_generic<float, 16, 16>(src, dst, kernels, biases, ReLU());
    }
    void conv3x3_16to16_identity_add_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
        conv3x3_generic<float, 16, 16>(src, dst, kernels, biases, Identity(), ResidualArg{ feat, 1.0f });
    }
    void conv3x3_16to4_identity_pixelshuffle_4to1_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_generic<float, std::uint8_t, 16, 2>(src, dst, kernels, biases);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_generic<float, std::uint16_t, 16, 2>(src, dst, kernels, biases);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_generic<float, float, 16, 2>(src, dst, kernels, biases);
            break;
        }
    }
    void conv3x3_16to4_identity_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_generic<float, 16, 4>(src, dst, kernels, biases, Identity());
    }

    void conv3x3_1to32_identity_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_generic<std::uint8_t, 1, 32>(src, dst, kernels, biases, Identity());
            break;
        case Image::UInt16:
            conv3x3_generic<std::uint16_t, 1, 32>(src, dst, kernels, biases, Identity());
            break;
        case Image::Float32:
            conv3x3_generic<float, 1, 32>(src, dst, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_32to32_relu_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_generic<float, 32, 32>(src, dst, kernels, biases, ReLU());
    }
    void conv3x3_32to32_identity_add_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
        conv3x3_generic<float, 32, 32>(src, dst, kernels, biases, Identity(), ResidualArg{ feat, 1.0f });
    }
    void conv3x3_32to4_identity_pixelshuffle_4to1_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_generic<float, std::uint8_t, 32, 2>(src, dst, kernels, biases);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_generic<float, std::uint16_t, 32, 2>(src, dst, kernels, biases);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_generic<float, float, 32, 2>(src, dst, kernels, biases);
            break;
        }
    }
    void conv3x3_32to4_identity_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_generic<float, 32, 4>(src, dst, kernels, biases, Identity());
    }

    void conv5x5_1to8_identity_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv5x5_generic<std::uint8_t, 1, 8>(src, dst, kernels, biases, Identity());
            break;
        case Image::UInt16:
            conv5x5_generic<std::uint16_t, 1, 8>(src, dst, kernels, biases, Identity());
            break;
        case Image::Float32:
            conv5x5_generic<float, 1, 8>(src, dst, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_8to8_prelu_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        conv3x3_generic<float, 8, 8>(src, dst, kernels, biases, PReLU(alphas));
    }
    void conv3x3_8to8_prelu_conv1x1_8to8_add_prelu_generic(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_generic<float, 8, 8, 8, false, true>(
            src, dst,
            kernels1, biases1, PReLU(alphas1), nullptr,
            kernels2, biases2, PReLU(alphas2), ResidualArg{ feat, 1.0f }
        );
    }

    void conv5x5_1to16_identity_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv5x5_generic<std::uint8_t, 1, 16>(src, dst, kernels, biases, Identity());
            break;
        case Image::UInt16:
            conv5x5_generic<std::uint16_t, 1, 16>(src, dst, kernels, biases, Identity());
            break;
        case Image::Float32:
            conv5x5_generic<float, 1, 16>(src, dst, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_16to16_prelu_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        conv3x3_generic<float, 16, 16>(src, dst, kernels, biases, PReLU(alphas));
    }
    void conv3x3_16to16_prelu_conv1x1_16to16_add_prelu_generic(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_generic<float, 16, 16, 16, false, true>(
            src, dst,
            kernels1, biases1, PReLU(alphas1), nullptr,
            kernels2, biases2, PReLU(alphas2), ResidualArg{ feat, 1.0f }
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
