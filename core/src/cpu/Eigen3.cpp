#include <Eigen/Core>

#include "AC/Core/Image.hpp"
#include "AC/Core/Util.hpp"

namespace ac::core::cpu
{
    template <typename IN, typename OUT, int cin, int cout>
    void conv3x3_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
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

            Eigen::Map<const Eigen::Array<IN, cin, 1>> r0(tl);
            Eigen::Map<const Eigen::Array<IN, cin, 1>> r1(tc);
            Eigen::Map<const Eigen::Array<IN, cin, 1>> r2(tr);
            Eigen::Map<const Eigen::Array<IN, cin, 1>> r3(ml);
            Eigen::Map<const Eigen::Array<IN, cin, 1>> r4(mc);
            Eigen::Map<const Eigen::Array<IN, cin, 1>> r5(mr);
            Eigen::Map<const Eigen::Array<IN, cin, 1>> r6(bl);
            Eigen::Map<const Eigen::Array<IN, cin, 1>> r7(bc);
            Eigen::Map<const Eigen::Array<IN, cin, 1>> r8(br);

            Eigen::Array<float, cin, 9> r{};
            if constexpr (std::is_same_v<IN, float>)
                r << r0, r1, r2, r3, r4, r5, r6, r7, r8;
            else if constexpr (std::is_floating_point_v<IN>)
                r << r0.template cast<float>(),
                        r1.template cast<float>(),
                        r2.template cast<float>(),
                        r3.template cast<float>(),
                        r4.template cast<float>(),
                        r5.template cast<float>(),
                        r6.template cast<float>(),
                        r7.template cast<float>(),
                        r8.template cast<float>();
            else if constexpr (std::is_unsigned_v<IN>)
                r << r0.template cast<float>() / std::numeric_limits<IN>::max(),
                        r1.template cast<float>() / std::numeric_limits<IN>::max(),
                        r2.template cast<float>() / std::numeric_limits<IN>::max(),
                        r3.template cast<float>() / std::numeric_limits<IN>::max(),
                        r4.template cast<float>() / std::numeric_limits<IN>::max(),
                        r5.template cast<float>() / std::numeric_limits<IN>::max(),
                        r6.template cast<float>() / std::numeric_limits<IN>::max(),
                        r7.template cast<float>() / std::numeric_limits<IN>::max(),
                        r8.template cast<float>() / std::numeric_limits<IN>::max();

            for (int n = 0; n < cout; n++)
            {
                Eigen::Map<const Eigen::Array<float, cin, 9>> k(kernels + n * cin * 9);
                out[n] = relu<OUT>((k * r).sum() + biases[n]);
            }
        }, src, dst);
    }
    template <typename IN, typename OUT, int cin, int cout>
    void deconv2x2_eigen3(const Image& src, Image& dst, const float* kernels)
    {
        filter<2>([=](const int i, const int j, const void* const sptr, void* const dptr) {
            auto in = static_cast<const IN*>(sptr);
            auto out = static_cast<OUT*>(dptr);

            const int index = ((i & 1) << 1) + (j & 1);

            constexpr int nstep = 4 * cout;

            auto r = [&]() -> auto {
                if constexpr (std::is_same_v<IN, float>)
                    return Eigen::Map<const Eigen::Array<float, cin, 1>> {in};
                else if constexpr (std::is_floating_point_v<IN>)
                    return (Eigen::Map<const Eigen::Array<IN, cin, 1>> {in}).template cast<float>();
                else if constexpr (std::is_unsigned_v<IN>)
                    return (Eigen::Map<const Eigen::Array<IN, cin, 1>> {in}).template cast<float>() / std::numeric_limits<IN>::max();
            }();

            for (int n = 0; n < cout; n++)
            {
                Eigen::Map<const Eigen::Array<float, cin, 1>, 0, Eigen::InnerStride<nstep>> k(kernels + cout * index);
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
