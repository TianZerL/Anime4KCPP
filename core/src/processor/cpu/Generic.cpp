#include <array>

#include "AC/Core/Image.hpp"
#include "AC/Core/Util.hpp"

namespace ac::core::cpu
{
    template <typename IN, int cin, int cout, typename ActiveFunc, typename... ResidualArgs>
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

                auto tl = static_cast<const IN*>(src.ptr(j - lp, i - tp));
                auto tc = static_cast<const IN*>(src.ptr(j     , i - tp));
                auto tr = static_cast<const IN*>(src.ptr(j + rp, i - tp));
                auto ml = static_cast<const IN*>(src.ptr(j - lp, i     ));
                auto mc = static_cast<const IN*>(src.ptr(j     , i     ));
                auto mr = static_cast<const IN*>(src.ptr(j + rp, i     ));
                auto bl = static_cast<const IN*>(src.ptr(j - lp, i + bp));
                auto bc = static_cast<const IN*>(src.ptr(j     , i + bp));
                auto br = static_cast<const IN*>(src.ptr(j + rp, i + bp));

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
                            toFloat<IN>(tl[c]) * k0[c] +
                            toFloat<IN>(tc[c]) * k1[c] +
                            toFloat<IN>(tr[c]) * k2[c] +
                            toFloat<IN>(ml[c]) * k3[c] +
                            toFloat<IN>(mc[c]) * k4[c] +
                            toFloat<IN>(mr[c]) * k5[c] +
                            toFloat<IN>(bl[c]) * k6[c] +
                            toFloat<IN>(bc[c]) * k7[c] +
                            toFloat<IN>(br[c]) * k8[c];
                    }

                    if constexpr (sizeof...(ResidualArgs))
                        for (int idx = 0; idx < sizeof...(ResidualArgs); idx++)
                            sum = sum * scales[idx] + iptrs[idx][n];

                    out[n] = activeFunc(sum);
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
                for (int c = 0; c < cin; c++) sum += toFloat<IN>(in[c]) * k[c];
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

                for (int n = 0; n < cout; n++) out[n] = fromFloat<OUT>(toFloat<IN>(in[n * group + index]));
            }
        }
    }

    template <typename IN, typename OUT>
    inline void conv3x3_8to4_identity_pixelshuffle_4to1_generic(const Image& src, Image& dst, const float* const kernels, const float* const biases) noexcept
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

                auto tl = static_cast<const IN*>(src.ptr(j - lp, i - tp));
                auto tc = static_cast<const IN*>(src.ptr(j     , i - tp));
                auto tr = static_cast<const IN*>(src.ptr(j + rp, i - tp));
                auto ml = static_cast<const IN*>(src.ptr(j - lp, i     ));
                auto mc = static_cast<const IN*>(src.ptr(j     , i     ));
                auto mr = static_cast<const IN*>(src.ptr(j + rp, i     ));
                auto bl = static_cast<const IN*>(src.ptr(j - lp, i + bp));
                auto bc = static_cast<const IN*>(src.ptr(j     , i + bp));
                auto br = static_cast<const IN*>(src.ptr(j + rp, i + bp));

                for (int n = 0; n < 4; n++)
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
                            toFloat<IN>(tl[c]) * k0[c] +
                            toFloat<IN>(tc[c]) * k1[c] +
                            toFloat<IN>(tr[c]) * k2[c] +
                            toFloat<IN>(ml[c]) * k3[c] +
                            toFloat<IN>(mc[c]) * k4[c] +
                            toFloat<IN>(mr[c]) * k5[c] +
                            toFloat<IN>(bl[c]) * k6[c] +
                            toFloat<IN>(bc[c]) * k7[c] +
                            toFloat<IN>(br[c]) * k8[c];
                    }

                    *static_cast<OUT*>(dst.ptr(dstX + (n & 1), dstY + (n >> 1))) = fromFloat<OUT>(sum);
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
    void conv3x3_8to8_residual_identity_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale)
    {
        conv3x3_generic<float, 8, 8>(src, dst, kernels, biases, Identity(), ResidualArg{ id, scale });
    }
    void conv3x3_8to8_residual_add_identity_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale, const Image& feat)
    {
        conv3x3_generic<float, 8, 8>(src, dst, kernels, biases, Identity(), ResidualArg{ id, scale }, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_8to4_identity_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_generic<float, 8, 4>(src, dst, kernels, biases, Identity());
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
    void conv3x3_8to4_identity_pixelshuffle_4to1_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_8to4_identity_pixelshuffle_4to1_generic<float, std::uint8_t>(src, dst, kernels, biases);
            break;
        case Image::UInt16:
            conv3x3_8to4_identity_pixelshuffle_4to1_generic<float, std::uint16_t>(src, dst, kernels, biases);
            break;
        case Image::Float32:
            conv3x3_8to4_identity_pixelshuffle_4to1_generic<float, float>(src, dst, kernels, biases);
            break;
        }
    }
}
