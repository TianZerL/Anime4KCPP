#include <array>
#include <type_traits>

#include <Eigen/Core>

#include "AC/Core/Image.hpp"
#include "AC/Core/Util.hpp"

namespace ac::core::cpu
{
    template <typename IN, int cin, int cout, bool postactive = false, typename ActiveFunc, typename... ResidualArgs>
    inline void conv1x1_eigen3(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc, ResidualArgs&& ...residualArg)
    {
        [[maybe_unused]] const std::array<float, sizeof...(ResidualArgs)> scales{ residualArg.scale... };

        util::parallelFor(0, src.height(), [&](const int i) {
            for (int j = 0; j < src.width(); j++)
            {
                [[maybe_unused]] const std::array<const float*, sizeof...(ResidualArgs)> iptrs{ static_cast<const float*>(residualArg.image.ptr(j, i))... };

                auto out = static_cast<float*>(dst.ptr(j, i));

                auto r = [&]() -> auto {
                    Eigen::Map<const Eigen::Array<IN, cin, 1>> rin{ static_cast<const IN*>(src.ptr(j, i)) };

                    if constexpr (std::is_same_v<IN, float>)
                        return rin;
                    else if constexpr (std::is_floating_point_v<IN>)
                        return Eigen::Array<float, cin, 1>{ rin.template cast<float>() };
                    else if constexpr (std::is_unsigned_v<IN>)
                        return Eigen::Array<float, cin, 1>{ rin.template cast<float>() / std::numeric_limits<IN>::max() };
                }();

                for (int n = 0; n < cout; n++)
                {
                    Eigen::Map<const Eigen::Array<float, cin, 1>> k{ kernels + n * cin };
                    float sum = (k * r).sum() + biases[n];

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
    inline void conv5x5_eigen3(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc, ResidualArgs&& ...residualArg)
    {
        [[maybe_unused]] const std::array<float, sizeof...(ResidualArgs)> scales{ residualArg.scale... };

        util::parallelFor(0, src.height(), [&](const int i) {
            int ioffsets[5] = { i > 1 ? -2 : (i > 0 ? -1 : 0) , i > 0 ? -1 : 0 , 0, i < src.height() - 1 ? 1 : 0, i < src.height() - 2 ? 2 : (i < src.height() - 1 ? 1 : 0) };

            for (int j = 0; j < src.width(); j++)
            {
                [[maybe_unused]] const std::array<const float*, sizeof...(ResidualArgs)> iptrs{ static_cast<const float*>(residualArg.image.ptr(j, i))... };

                auto out = static_cast<float*>(dst.ptr(j, i));

                auto r = [&]() -> auto {
                    int joffsets[5] = { j > 1 ? -2 : (j > 0 ? -1 : 0), j > 0 ? -1 : 0, 0, j < src.width() - 1 ? 1 : 0 ,j < src.width() - 2 ? 2 : (j < src.width() - 1 ? 1 : 0) };

                    Eigen::Array<IN, cin, 25> rin{};
                    for (int in = 0; in < 5; in++)
                        for (int jn = 0; jn < 5; jn++)
                            rin.col(in * 5 + jn) = Eigen::Map<const Eigen::Array<IN, cin, 1>>{ static_cast<const IN*>(src.ptr(j + joffsets[jn], i + ioffsets[in])) };

                    if constexpr (std::is_same_v<IN, float>)
                        return rin;
                    else if constexpr (std::is_floating_point_v<IN>)
                        return Eigen::Array<float, cin, 25>{ rin.template cast<float>() };
                    else if constexpr (std::is_unsigned_v<IN>)
                        return Eigen::Array<float, cin, 25>{ rin.template cast<float>() / std::numeric_limits<IN>::max() };
                }();

                for (int n = 0; n < cout; n++)
                {
                    Eigen::Map<const Eigen::Array<float, cin, 25>> k{ kernels + n * cin * 25 };
                    float sum = (k * r).sum() + biases[n];

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
                    Eigen::Map<const Eigen::Array<float, cin, 9>> k{ kernels + n * cin * 9 };
                    float sum = (k * r).sum() + biases[n];

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

    template <typename IN, typename OUT, int cin, int upscale>
    inline void conv3x3_identity_pixelshuffle_eigen3(const Image& src, Image& dst, const float* const kernels, const float* const biases) noexcept
    {
        static constexpr int cout = upscale * upscale;

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

                for (int n = 0; n < cout; n++)
                {
                    Eigen::Map<const Eigen::Array<float, cin, 9>> k(kernels + n * cin * 9);
                    float sum = (k * r).sum() + biases[n];

                    *static_cast<OUT*>(dst.ptr(dstX + (n & 1), dstY + (n >> 1))) = fromFloat<OUT>(sum);
                }
            }
        });
    }

    template <typename IN, int cin, int ctemp, int cout, bool postactive3x3 = false, bool postactive1x1 = false, typename ActiveFunc3x3, typename ResidualArg3x3, typename ActiveFunc1x1, typename ResidualArg1x1>
    inline void conv3x3_conv1x1_eigen3(
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

                for (int n = 0; n < ctemp; n++)
                {
                    Eigen::Map<const Eigen::Array<float, cin, 9>> k{ kernels3x3 + n * cin * 9 };
                    float sum = (k * r).sum() + biases3x3[n];

                    if constexpr (!postactive3x3) sum = activeFunc3x3(sum, n);

                    if constexpr (std::is_same_v<ResidualArg3x3, ResidualArg>)
                        sum = sum * residualArg3x3.scale + static_cast<const float*>(residualArg3x3.image.ptr(j, i))[n];

                    if constexpr (postactive3x3) sum = activeFunc3x3(sum, n);

                    buffer[n] = sum;
                }

                Eigen::Map<const Eigen::Array<float, ctemp, 1>> rb{ buffer };

                for (int n = 0; n < cout; n++)
                {
                    Eigen::Map<const Eigen::Array<float, ctemp, 1>> k{ kernels1x1 + n * ctemp };

                    float sum = (k * rb).sum() + biases1x1[n];

                    if constexpr (!postactive1x1) sum = activeFunc1x1(sum, n);

                    if constexpr (std::is_same_v<ResidualArg1x1, ResidualArg>)
                        sum = sum * residualArg1x1.scale + static_cast<const float*>(residualArg1x1.image.ptr(j, i))[n];

                    if constexpr (postactive1x1) sum = activeFunc1x1(sum, n);

                    out[n] = sum;
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
    void conv3x3_8to8_identity_residual_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale)
    {
        conv3x3_eigen3<float, 8, 8>(src, dst, kernels, biases, Identity(), ResidualArg{ id, scale });
    }
    void conv3x3_8to8_identity_residual_add_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale, const Image& feat)
    {
        conv3x3_eigen3<float, 8, 8>(src, dst, kernels, biases, Identity(), ResidualArg{ id, scale }, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_eigen3<float, std::uint8_t, 8, 2>(src, dst, kernels, biases);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_eigen3<float, std::uint16_t, 8, 2>(src, dst, kernels, biases);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_eigen3<float, float, 8, 2>(src, dst, kernels, biases);
            break;
        }
    }
    void conv3x3_8to4_identity_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_eigen3<float, 8, 4>(src, dst, kernels, biases, Identity());
    }

    void conv3x3_1to16_identity_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_eigen3<std::uint8_t, 1, 16>(src, dst, kernels, biases, Identity());
            break;
        case Image::UInt16:
            conv3x3_eigen3<std::uint16_t, 1, 16>(src, dst, kernels, biases, Identity());
            break;
        case Image::Float32:
            conv3x3_eigen3<float, 1, 16>(src, dst, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_16to16_relu_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_eigen3<float, 16, 16>(src, dst, kernels, biases, ReLU());
    }
    void conv3x3_16to16_identity_add_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
        conv3x3_eigen3<float, 16, 16>(src, dst, kernels, biases, Identity(), ResidualArg{ feat, 1.0f });
    }
    void conv3x3_16to4_identity_pixelshuffle_4to1_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_eigen3<float, std::uint8_t, 16, 2>(src, dst, kernels, biases);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_eigen3<float, std::uint16_t, 16, 2>(src, dst, kernels, biases);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_eigen3<float, float, 16, 2>(src, dst, kernels, biases);
            break;
        }
    }
    void conv3x3_16to4_identity_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_eigen3<float, 16, 4>(src, dst, kernels, biases, Identity());
    }

    void conv3x3_1to32_identity_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_eigen3<std::uint8_t, 1, 32>(src, dst, kernels, biases, Identity());
            break;
        case Image::UInt16:
            conv3x3_eigen3<std::uint16_t, 1, 32>(src, dst, kernels, biases, Identity());
            break;
        case Image::Float32:
            conv3x3_eigen3<float, 1, 32>(src, dst, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_32to32_relu_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_eigen3<float, 32, 32>(src, dst, kernels, biases, ReLU());
    }
    void conv3x3_32to32_identity_add_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
        conv3x3_eigen3<float, 32, 32>(src, dst, kernels, biases, Identity(), ResidualArg{ feat, 1.0f });
    }
    void conv3x3_32to4_identity_pixelshuffle_4to1_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_eigen3<float, std::uint8_t, 32, 2>(src, dst, kernels, biases);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_eigen3<float, std::uint16_t, 32, 2>(src, dst, kernels, biases);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_eigen3<float, float, 32, 2>(src, dst, kernels, biases);
            break;
        }
    }
    void conv3x3_32to4_identity_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_eigen3<float, 32, 4>(src, dst, kernels, biases, Identity());
    }

    void conv5x5_1to8_identity_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv5x5_eigen3<std::uint8_t, 1, 8>(src, dst, kernels, biases, Identity());
            break;
        case Image::UInt16:
            conv5x5_eigen3<std::uint16_t, 1, 8>(src, dst, kernels, biases, Identity());
            break;
        case Image::Float32:
            conv5x5_eigen3<float, 1, 8>(src, dst, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_8to8_prelu_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        conv3x3_eigen3<float, 8, 8>(src, dst, kernels, biases, PReLU(alphas));
    }
    void conv3x3_8to8_prelu_conv1x1_8to8_add_prelu_eigen3(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_eigen3<float, 8, 8, 8, false, true>(
            src, dst,
            kernels1, biases1, PReLU(alphas1), nullptr,
            kernels2, biases2, PReLU(alphas2), ResidualArg{ feat, 1.0f }
        );
    }

    void conv5x5_1to16_identity_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv5x5_eigen3<std::uint8_t, 1, 16>(src, dst, kernels, biases, Identity());
            break;
        case Image::UInt16:
            conv5x5_eigen3<std::uint16_t, 1, 16>(src, dst, kernels, biases, Identity());
            break;
        case Image::Float32:
            conv5x5_eigen3<float, 1, 16>(src, dst, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_16to16_prelu_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        conv3x3_eigen3<float, 16, 16>(src, dst, kernels, biases, PReLU(alphas));
    }
    void conv3x3_16to16_prelu_conv1x1_16to16_add_prelu_eigen3(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_eigen3<float, 16, 16, 16, false, true>(
            src, dst,
            kernels1, biases1, PReLU(alphas1), nullptr,
            kernels2, biases2, PReLU(alphas2), ResidualArg{ feat, 1.0f }
        );
    }
}
