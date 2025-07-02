#include "AC/Core/Image.hpp"
#include "AC/Core/Util.hpp"

namespace ac::core::cpu
{
    template <typename IN, typename OUT, int cin, int cout, bool residual = false>
    inline void conv3x3_generic(const Image& src, Image& dst, const float* const kernels, const float* const biases)
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
                if constexpr (residual) sum += out[n];

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
                out[n] = relu<OUT>(sum);
            }
        }, src, dst);
    }
    template <typename IN, typename OUT, int cin, int cout>
    inline void deconv2x2_generic(const Image& src, Image& dst, const float* const kernels)
    {
        filter([=](const int i, const int j, const void* const sptr, void* const dptr) {
            auto in = static_cast<const IN*>(sptr);
            auto out = static_cast<OUT*>(dptr);

            const int index = ((i & 1) << 1) + (j & 1);

            for (int n = 0; n < cout; n++)
            {
                auto k = kernels + n * cin * 4 + cin * index;
                float sum = 0.0f;
                for (int c = 0; c < cin; c++) sum += toFloat<IN>(in[c]) * k[c];
                out[n] = fromFloat<OUT>(sum);
            }
        }, src, dst);
    }

    void conv3x3_1to8_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_generic<std::uint8_t, float, 1, 8>(src, dst, kernels, biases);
            break;
        case Image::UInt16:
            conv3x3_generic<std::uint16_t, float, 1, 8>(src, dst, kernels, biases);
            break;
        case Image::Float32:
            conv3x3_generic<float, float, 1, 8>(src, dst, kernels, biases);
            break;
        }
    }
    void conv3x3_8to8_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_generic<float, float, 8, 8>(src, dst, kernels, biases);
    }
    void conv3x3_residual_8to8_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_generic<float, float, 8, 8, true>(src, dst, kernels, biases);
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
}
