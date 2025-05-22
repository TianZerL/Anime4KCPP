#include <Eigen/Core>

#include "AC/Core/Image.hpp"
#include "AC/Core/Util.hpp"

namespace ac::core::cpu
{
    template <typename IN, typename OUT, int cin, int cout, bool residual = false>
    inline void conv3x3_eigen3(const Image& src, Image& dst, const float* const kernels, const float* const biases)
    {
        int w = src.width(), h = src.height();
        int step = src.stride() / src.elementSize();

        filter([=](const int i, const int j, const void* const sptr, void* const dptr) {
            auto in = static_cast<const IN*>(sptr);
            auto out = static_cast<OUT*>(dptr);

            auto sp = i < h - 1 ? +step : 0;
            auto sn = i > 0 ? -step : 0;
            auto cp = j < w - 1 ? +cin : 0;
            auto cn = j > 0 ? -cin : 0;

            auto tl = in + sn + cn, tc = in + sn, tr = in + sn + cp;
            auto ml = in + cn, mc = in, mr = in + cp;
            auto bl = in + sp + cn, bc = in + sp, br = in + sp + cp;

            auto r = [&]() -> auto {
                Eigen::Array<IN, cin, 9> rin{};
                rin <<
                    Eigen::Map<const Eigen::Array<IN, cin, 1>>{ tl },
                    Eigen::Map<const Eigen::Array<IN, cin, 1>>{ tc },
                    Eigen::Map<const Eigen::Array<IN, cin, 1>>{ tr },
                    Eigen::Map<const Eigen::Array<IN, cin, 1>>{ ml },
                    Eigen::Map<const Eigen::Array<IN, cin, 1>>{ mc },
                    Eigen::Map<const Eigen::Array<IN, cin, 1>>{ mr },
                    Eigen::Map<const Eigen::Array<IN, cin, 1>>{ bl },
                    Eigen::Map<const Eigen::Array<IN, cin, 1>>{ bc },
                    Eigen::Map<const Eigen::Array<IN, cin, 1>>{ br };
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
                float sum = (k * r).sum();
                if constexpr (residual) sum += out[n];
                out[n] = relu<OUT>(sum + biases[n]);
            }
        }, src, dst);
    }
    template <typename IN, typename OUT, int cin, int cout>
    inline void deconv2x2_eigen3(const Image& src, Image& dst, const float* const kernels)
    {
        filter([=](const int i, const int j, const void* const sptr, void* const dptr) {
            auto in = static_cast<const IN*>(sptr);
            auto out = static_cast<OUT*>(dptr);

            const int index = ((i & 1) << 1) + (j & 1);

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

    void conv3x3_1to8_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_eigen3<std::uint8_t, float, 1, 8>(src, dst, kernels, biases);
            break;
        case Image::UInt16:
            conv3x3_eigen3<std::uint16_t, float, 1, 8>(src, dst, kernels, biases);
            break;
        case Image::Float32:
            conv3x3_eigen3<float, float, 1, 8>(src, dst, kernels, biases);
            break;
        }
    }
    void conv3x3_8to8_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_eigen3<float, float, 8, 8>(src, dst, kernels, biases);
    }
    void conv3x3_residual_8to8_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_eigen3<float, float, 8, 8, true>(src, dst, kernels, biases);
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
}
