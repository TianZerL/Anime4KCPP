#include <array>

#include <Eigen/Core>

#include "AC/Core/Image.hpp"
#include "AC/Core/Util.hpp"

namespace ac::core::cpu
{
    template <typename IN, int cin, int cout, typename ActiveFunc, typename... ResidualArgs>
    inline void conv3x3_eigen3(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc, ResidualArgs&& ...residualArg)
    {
        [[maybe_unused]] const std::array<float, sizeof...(ResidualArgs)> scales{ residualArg.scale... };

        util::parallelFor(0, src.height(), [&](const int i) {
            auto tp = i > 0 ? 1 : 0;
            auto bp = i < src.height() - 1 ? 1 : 0;

            for (int j = 0; j < src.width(); j++)
            {
                [[maybe_unused]] const std::array<const float*, sizeof...(ResidualArgs)> iptrs{ static_cast<const float*>(residualArg.image.ptr(j, i))... };

                auto out = static_cast<float*>(dst.ptr(j, i));

                auto r = [&]() -> auto {
                    auto lp = j > 0 ? 1 : 0;
                    auto rp = j < src.width() - 1 ? 1 : 0;

                    Eigen::Array<IN, cin, 9> rin{};
                    rin <<
                        Eigen::Map<const Eigen::Array<IN, cin, 1>>{ static_cast<const IN*>(src.ptr(j - lp, i - tp)) },
                        Eigen::Map<const Eigen::Array<IN, cin, 1>>{ static_cast<const IN*>(src.ptr(j     , i - tp)) },
                        Eigen::Map<const Eigen::Array<IN, cin, 1>>{ static_cast<const IN*>(src.ptr(j + rp, i - tp)) },
                        Eigen::Map<const Eigen::Array<IN, cin, 1>>{ static_cast<const IN*>(src.ptr(j - lp, i     )) },
                        Eigen::Map<const Eigen::Array<IN, cin, 1>>{ static_cast<const IN*>(src.ptr(j     , i     )) },
                        Eigen::Map<const Eigen::Array<IN, cin, 1>>{ static_cast<const IN*>(src.ptr(j + rp, i     )) },
                        Eigen::Map<const Eigen::Array<IN, cin, 1>>{ static_cast<const IN*>(src.ptr(j - lp, i + bp)) },
                        Eigen::Map<const Eigen::Array<IN, cin, 1>>{ static_cast<const IN*>(src.ptr(j     , i + bp)) },
                        Eigen::Map<const Eigen::Array<IN, cin, 1>>{ static_cast<const IN*>(src.ptr(j + rp, i + bp)) };
                    if constexpr (std::is_same_v<IN, float>)
                        return rin;
                    else if constexpr (std::is_floating_point_v<IN>)
                        return Eigen::Array<float, cin, 9>{ rin.template cast<float>() };
                    else if constexpr (std::is_unsigned_v<IN>)
                        return Eigen::Array<float, cin, 9>{ rin.template cast<float>() / std::numeric_limits<IN>::max() };
                }();

                for (int n = 0; n < cout; n++)
                {
                    Eigen::Map<const Eigen::Array<float, cin, 9>> k(kernels + n * cin * 9);
                    float sum = (k * r).sum() + biases[n];

                    if constexpr (sizeof...(ResidualArgs))
                        for (int idx = 0; idx < sizeof...(ResidualArgs); idx++)
                            sum = sum * scales[idx] + iptrs[idx][n];

                    out[n] = activeFunc(sum);
                }
            }
        });
    }
    template <typename IN, typename OUT, int cin, int cout>
    inline void deconv2x2_eigen3(const Image& src, Image& dst, const float* const kernels)
    {
        filter([=](const int i, const int j, const void* const sptr, void* const dptr) {
            auto in = static_cast<const IN*>(sptr);
            auto out = static_cast<OUT*>(dptr);

            auto index = ((i & 1) << 1) + (j & 1);

            auto r = [&]() -> auto {
                Eigen::Map<const Eigen::Array<IN, cin, 1>> rin{ in };
                if constexpr (std::is_same_v<IN, float>)
                    return rin;
                else if constexpr (std::is_floating_point_v<IN>)
                    return Eigen::Array<float, cin, 1>{ rin.template cast<float>() };
                else if constexpr (std::is_unsigned_v<IN>)
                    return Eigen::Array<float, cin, 1>{ rin.template cast<float>() / std::numeric_limits<IN>::max() };
            }();

            for (int n = 0; n < cout; n++)
            {
                Eigen::Map<const Eigen::Array<float, cin, 1>> k(kernels + n * cin * 4 + cin * index);
                out[n] = fromFloat<OUT>((k * r).sum());
            }
        }, src, dst);
    }

    template <typename IN, typename OUT>
    inline void conv3x3_8to4_identity_pixelshuffle_4to1_eigen3(const Image& src, Image& dst, const float* const kernels, const float* const biases) noexcept
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

                auto r = [&]() -> auto {
                    auto lp = j > 0 ? 1 : 0;
                    auto rp = j < src.width() - 1 ? 1 : 0;

                    Eigen::Array<IN, cin, 9> rin{};
                    rin <<
                        Eigen::Map<const Eigen::Array<IN, cin, 1>>{ static_cast<const IN*>(src.ptr(j - lp, i - tp)) },
                        Eigen::Map<const Eigen::Array<IN, cin, 1>>{ static_cast<const IN*>(src.ptr(j     , i - tp)) },
                        Eigen::Map<const Eigen::Array<IN, cin, 1>>{ static_cast<const IN*>(src.ptr(j + rp, i - tp)) },
                        Eigen::Map<const Eigen::Array<IN, cin, 1>>{ static_cast<const IN*>(src.ptr(j - lp, i     )) },
                        Eigen::Map<const Eigen::Array<IN, cin, 1>>{ static_cast<const IN*>(src.ptr(j     , i     )) },
                        Eigen::Map<const Eigen::Array<IN, cin, 1>>{ static_cast<const IN*>(src.ptr(j + rp, i     )) },
                        Eigen::Map<const Eigen::Array<IN, cin, 1>>{ static_cast<const IN*>(src.ptr(j - lp, i + bp)) },
                        Eigen::Map<const Eigen::Array<IN, cin, 1>>{ static_cast<const IN*>(src.ptr(j     , i + bp)) },
                        Eigen::Map<const Eigen::Array<IN, cin, 1>>{ static_cast<const IN*>(src.ptr(j + rp, i + bp)) };
                    if constexpr (std::is_same_v<IN, float>)
                        return rin;
                    else if constexpr (std::is_floating_point_v<IN>)
                        return Eigen::Array<float, cin, 9>{ rin.template cast<float>() };
                    else if constexpr (std::is_unsigned_v<IN>)
                        return Eigen::Array<float, cin, 9>{ rin.template cast<float>() / std::numeric_limits<IN>::max() };
                }();

                for (int n = 0; n < 4; n++)
                {
                    Eigen::Map<const Eigen::Array<float, cin, 9>> k(kernels + n * cin * 9);
                    float sum = (k * r).sum() + biases[n];

                    *static_cast<OUT*>(dst.ptr(dstX + (n & 1), dstY + (n >> 1))) = fromFloat<OUT>(sum);
                }
            }
        });
    }

    void conv3x3_1to8_relu_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_eigen3<std::uint8_t, 1, 8>(src, dst, kernels, biases, ReLU());
            break;
        case Image::UInt16:
            conv3x3_eigen3<std::uint16_t, 1, 8>(src, dst, kernels, biases, ReLU());
            break;
        case Image::Float32:
            conv3x3_eigen3<float, 1, 8>(src, dst, kernels, biases, ReLU());
            break;
        }
    }
    void conv3x3_8to8_relu_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_eigen3<float, 8, 8>(src, dst, kernels, biases, ReLU());
    }
    void deconv2x2_8to1_eigen3(const Image& src, Image& dst, const float* kernels)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            deconv2x2_eigen3<float, std::uint8_t, 8, 1>(src, dst, kernels);
            break;
        case Image::UInt16:
            deconv2x2_eigen3<float, std::uint16_t, 8, 1>(src, dst, kernels);
            break;
        case Image::Float32:
            deconv2x2_eigen3<float, float, 8, 1>(src, dst, kernels);
            break;
        }
    }

    void conv3x3_1to8_identity_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_eigen3<std::uint8_t, 1, 8>(src, dst, kernels, biases, Identity());
            break;
        case Image::UInt16:
            conv3x3_eigen3<std::uint16_t, 1, 8>(src, dst, kernels, biases, Identity());
            break;
        case Image::Float32:
            conv3x3_eigen3<float, 1, 8>(src, dst, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_8to8_lrelu_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases, const float negativeSlope)
    {
        conv3x3_eigen3<float, 8, 8>(src, dst, kernels, biases, LReLU(negativeSlope));
    }
    void conv3x3_8to8_residual_identity_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale)
    {
        conv3x3_eigen3<float, 8, 8>(src, dst, kernels, biases, Identity(), ResidualArg{ id, scale });
    }
    void conv3x3_8to8_residual_add_identity_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale, const Image& feat)
    {
        conv3x3_eigen3<float, 8, 8>(src, dst, kernels, biases, Identity(), ResidualArg{ id, scale }, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_8to4_identity_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_eigen3<float, 8, 4>(src, dst, kernels, biases, Identity());
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_8to4_identity_pixelshuffle_4to1_eigen3<float, std::uint8_t>(src, dst, kernels, biases);
            break;
        case Image::UInt16:
            conv3x3_8to4_identity_pixelshuffle_4to1_eigen3<float, std::uint16_t>(src, dst, kernels, biases);
            break;
        case Image::Float32:
            conv3x3_8to4_identity_pixelshuffle_4to1_eigen3<float, float>(src, dst, kernels, biases);
            break;
        }
    }
}
