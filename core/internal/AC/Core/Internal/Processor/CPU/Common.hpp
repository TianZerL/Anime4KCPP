#ifndef AC_CORE_INTERNAL_PROCESSOR_CPU_COMMON_HPP
#define AC_CORE_INTERNAL_PROCESSOR_CPU_COMMON_HPP

#include <array>
#include <cstring>
#include <type_traits>

#include "AC/Core/Image.hpp"
#include "AC/Core/Util.hpp"

namespace ac::core::cpu
{
    template <typename OpImpl, typename OUT, int cin, int cout>
    inline void deconv2x2(const Image& src, Image& dst, const float* const kernels)
    {
        filter([=](const int i, const int j, const void* const sptr, void* const dptr) {
            auto in = static_cast<const float*>(sptr);
            auto out = static_cast<OUT*>(dptr);

            auto index = ((i & 1) << 1) + (j & 1);

            for (int n = 0; n < cout; n++) out[n] = fromFloat<OUT>(OpImpl::template dot<cin>(in, kernels + n * cin * 4 + cin * index));
        }, src, dst);
    }

    template <typename OUT, int cin, int upscale>
    inline void pixelshuffle(const Image& src, Image& dst) noexcept
    {
        constexpr int group = upscale * upscale;
        constexpr int cout = cin / group;
        static_assert(cin % group == 0 && cout > 0);

        for (int i = 0; i < dst.height(); i++)
        {
            for (int j = 0; j < dst.width(); j++)
            {
                auto in = static_cast<const float*>(src.ptr(j / upscale, i / upscale));
                auto out = static_cast<OUT*>(dst.ptr(j, i));

                auto index = (i % upscale) * upscale + (j % upscale);

                for (int n = 0; n < cout; n++) out[n] = fromFloat<OUT>(in[n * group + index]);
            }
        }
    }

    template <typename OpImpl, int cin, int cout, bool postactive = false, typename ActiveFunc, typename... ResidualArgs>
    inline void conv1x1_float(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc, ResidualArgs&& ...residualArg)
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

                OpImpl::template conv<cin, cout, 1 * 1>(rptr, sum, kernels, biases);

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

    template <typename OpImpl, int cin, int cout, bool postactive = false, typename ActiveFunc, typename... ResidualArgs>
    inline void conv3x3_float(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc, ResidualArgs&& ...residualArg)
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

                OpImpl::template conv<cin, cout, 3 * 3>(rptr, sum, kernels, biases);

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

    template <typename OpImpl, typename IN, int cout, typename ActiveFunc>
    inline void conv3x3_cin1(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc)
    {
        util::parallelFor(0, src.height(), [&](const int i) {
            auto tp = i > 0 ? 1 : 0;
            auto bp = i < src.height() - 1 ? 1 : 0;

            for (int j = 0; j < src.width(); j++)
            {
                auto out = static_cast<float*>(dst.ptr(j, i));

                auto lp = j > 0 ? 1 : 0;
                auto rp = j < src.width() - 1 ? 1 : 0;

                const float rptr[] = {
                    toFloat(*static_cast<const IN*>(src.ptr(j - lp, i - tp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j     , i - tp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + rp, i - tp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j - lp, i     ))),
                    toFloat(*static_cast<const IN*>(src.ptr(j     , i     ))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + rp, i     ))),
                    toFloat(*static_cast<const IN*>(src.ptr(j - lp, i + bp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j     , i + bp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + rp, i + bp))),
                };

                OpImpl::template conv_cin1<cout, 3 * 3>(rptr, out, kernels, biases);

                for (int n = 0; n < cout; n++) out[n] = activeFunc(out[n], n);
            }
        });
    }

    template <typename OpImpl, typename IN, int cout, typename ActiveFunc>
    inline void conv5x5_cin1(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc)
    {
        util::parallelFor(0, src.height(), [&](const int i) {
            int ioffsets[] = { i > 1 ? -2 : (i > 0 ? -1 : 0) , i > 0 ? -1 : 0 , 0, i < src.height() - 1 ? 1 : 0, i < src.height() - 2 ? 2 : (i < src.height() - 1 ? 1 : 0) };

            for (int j = 0; j < src.width(); j++)
            {
                auto out = static_cast<float*>(dst.ptr(j, i));

                int joffsets[] = { j > 1 ? -2 : (j > 0 ? -1 : 0), j > 0 ? -1 : 0, 0, j < src.width() - 1 ? 1 : 0 ,j < src.width() - 2 ? 2 : (j < src.width() - 1 ? 1 : 0) };

                float rptr[25];
                for (int in = 0; in < 5; in++)
                    for (int jn = 0; jn < 5; jn++)
                        rptr[in * 5 + jn] = toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[jn], i + ioffsets[in])));

                OpImpl::template conv_cin1<cout, 5 * 5>(rptr, out, kernels, biases);

                for (int n = 0; n < cout; n++) out[n] = activeFunc(out[n], n);
            }
        });
    }

    template <typename OpImpl, int cin, int ctemp, int cout,
        bool postactive3x3 = false,
        bool postactive1x1 = false,
        typename ActiveFunc3x3, typename ResidualArg3x3,
        typename ActiveFunc1x1, typename ResidualArg1x1>
    inline void conv3x3_conv1x1_float(
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

                OpImpl::template conv<cin, ctemp, 3 * 3>(rptr, buffer, kernels3x3, biases3x3);

                for (int n = 0; n < ctemp; n++)
                {
                    if constexpr (!postactive3x3) buffer[n] = activeFunc3x3(buffer[n], n);

                    if constexpr (std::is_same_v<ResidualArg3x3, ResidualArg>)
                        buffer[n] = buffer[n] * residualArg3x3.scale + static_cast<const float*>(residualArg3x3.image.ptr(j, i))[n];

                    if constexpr (postactive3x3) buffer[n] = activeFunc3x3(buffer[n], n);
                }

                rptr[0] = buffer;
                float sum[cout]{};

                OpImpl::template conv<ctemp, cout, 1 * 1>(rptr, sum, kernels1x1, biases1x1);

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

    template <typename OpImpl, typename OUT, int cin, int upscale, typename NearestInterpolationArg>
    inline void conv3x3_identity_pixelshuffle_float(
        const Image& src, Image& dst,
        const float* const kernels, const float* const biases,
        NearestInterpolationArg&& nearestInterpolationArg)
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

                OpImpl::template conv<cin, cout, 3 * 3>(rptr, sum, kernels, biases);

                constexpr bool addNearestInterpolation = std::is_same_v<NearestInterpolationArg, ResidualArg>;

                [[maybe_unused]] float nearestInterpolationData{};
                if constexpr (addNearestInterpolation) nearestInterpolationData = toFloat(*static_cast<const OUT*>(nearestInterpolationArg.image.ptr(j, i)));

                for (int n = 0; n < cout; n++)
                {
                    if constexpr (addNearestInterpolation) sum[n] = sum[n] * nearestInterpolationArg.scale + nearestInterpolationData;

                    *static_cast<OUT*>(dst.ptr(dstX + (n & 1), dstY + (n >> 1))) = fromFloat<OUT>(sum[n]);
                }
            }
        });
    }
}

#endif
