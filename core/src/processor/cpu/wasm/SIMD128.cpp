#include <wasm_simd128.h>

#include "AC/Core/Image.hpp"
#include "AC/Core/Util.hpp"

namespace ac::core::cpu
{
    static inline float wasm_simd128_f32x4_hsum(const v128_t& v) noexcept
    {
        v128_t v64 = wasm_f32x4_add(v, wasm_i32x4_shuffle(v, v, 2, 3, 0, 0));
        v128_t v32 = wasm_f32x4_add(v64, wasm_i32x4_shuffle(v64, v64, 1, 0, 0, 0));
        return wasm_f32x4_extract_lane(v32, 0);
    }

    template <typename OUT, int cin, int cout, bool residual = false>
    inline void conv3x3_wasm_simd128_float(const Image& src, Image& dst, const float* const kernels, const float* const biases)
    {
        int w = src.width(), h = src.height();
        int step = src.stride() / src.elementSize();

        filter([=](const int i, const int j, const void* const sptr, void* const dptr) {
            auto in = static_cast<const float*>(sptr);
            auto out = static_cast<OUT*>(dptr);

            auto sp = i < h - 1 ? +step : 0;
            auto sn = i > 0 ? -step : 0;
            auto cp = j < w - 1 ? +cin : 0;
            auto cn = j > 0 ? -cin : 0;

            auto tl = in + sn + cn, tc = in + sn, tr = in + sn + cp;
            auto ml = in + cn, mc = in, mr = in + cp;
            auto bl = in + sp + cn, bc = in + sp, br = in + sp + cp;

            constexpr int vstep = 4;
            constexpr int count = cin / vstep;
            constexpr int remain = cin % vstep;

            v128_t r0[count + (remain ? 1 : 0)]{};
            v128_t r1[count + (remain ? 1 : 0)]{};
            v128_t r2[count + (remain ? 1 : 0)]{};
            v128_t r3[count + (remain ? 1 : 0)]{};
            v128_t r4[count + (remain ? 1 : 0)]{};
            v128_t r5[count + (remain ? 1 : 0)]{};
            v128_t r6[count + (remain ? 1 : 0)]{};
            v128_t r7[count + (remain ? 1 : 0)]{};
            v128_t r8[count + (remain ? 1 : 0)]{};

            for (int idx = 0; idx < count; idx++)
            {
                r0[idx] = wasm_v128_load(tl + idx * vstep);
                r1[idx] = wasm_v128_load(tc + idx * vstep);
                r2[idx] = wasm_v128_load(tr + idx * vstep);
                r3[idx] = wasm_v128_load(ml + idx * vstep);
                r4[idx] = wasm_v128_load(mc + idx * vstep);
                r5[idx] = wasm_v128_load(mr + idx * vstep);
                r6[idx] = wasm_v128_load(bl + idx * vstep);
                r7[idx] = wasm_v128_load(bc + idx * vstep);
                r8[idx] = wasm_v128_load(br + idx * vstep);
            }
            if constexpr (remain)
            {
                r0[count] = wasm_f32x4_make((tl + count * vstep)[0], remain > 1 ? (tl + count * vstep)[1] : 0.0f, remain > 2 ? (tl + count * vstep)[2] : 0.0f, 0.0f);
                r1[count] = wasm_f32x4_make((tc + count * vstep)[0], remain > 1 ? (tc + count * vstep)[1] : 0.0f, remain > 2 ? (tc + count * vstep)[2] : 0.0f, 0.0f);
                r2[count] = wasm_f32x4_make((tr + count * vstep)[0], remain > 1 ? (tr + count * vstep)[1] : 0.0f, remain > 2 ? (tr + count * vstep)[2] : 0.0f, 0.0f);
                r3[count] = wasm_f32x4_make((ml + count * vstep)[0], remain > 1 ? (ml + count * vstep)[1] : 0.0f, remain > 2 ? (ml + count * vstep)[2] : 0.0f, 0.0f);
                r4[count] = wasm_f32x4_make((mc + count * vstep)[0], remain > 1 ? (mc + count * vstep)[1] : 0.0f, remain > 2 ? (mc + count * vstep)[2] : 0.0f, 0.0f);
                r5[count] = wasm_f32x4_make((mr + count * vstep)[0], remain > 1 ? (mr + count * vstep)[1] : 0.0f, remain > 2 ? (mr + count * vstep)[2] : 0.0f, 0.0f);
                r6[count] = wasm_f32x4_make((bl + count * vstep)[0], remain > 1 ? (bl + count * vstep)[1] : 0.0f, remain > 2 ? (bl + count * vstep)[2] : 0.0f, 0.0f);
                r7[count] = wasm_f32x4_make((bc + count * vstep)[0], remain > 1 ? (bc + count * vstep)[1] : 0.0f, remain > 2 ? (bc + count * vstep)[2] : 0.0f, 0.0f);
                r8[count] = wasm_f32x4_make((br + count * vstep)[0], remain > 1 ? (br + count * vstep)[1] : 0.0f, remain > 2 ? (br + count * vstep)[2] : 0.0f, 0.0f);
            }

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
                v128_t s = wasm_f32x4_const_splat(0.0f);
                for (int idx = 0; idx < count; idx++)
                {
                    v128_t k0 = wasm_v128_load(kptr[0] + idx * vstep);
                    v128_t k1 = wasm_v128_load(kptr[1] + idx * vstep);
                    v128_t k2 = wasm_v128_load(kptr[2] + idx * vstep);
                    v128_t k3 = wasm_v128_load(kptr[3] + idx * vstep);
                    v128_t k4 = wasm_v128_load(kptr[4] + idx * vstep);
                    v128_t k5 = wasm_v128_load(kptr[5] + idx * vstep);
                    v128_t k6 = wasm_v128_load(kptr[6] + idx * vstep);
                    v128_t k7 = wasm_v128_load(kptr[7] + idx * vstep);
                    v128_t k8 = wasm_v128_load(kptr[8] + idx * vstep);

                    v128_t s0 = wasm_f32x4_mul(r0[idx], k0);
                    v128_t s1 = wasm_f32x4_mul(r1[idx], k1);
                    v128_t s2 = wasm_f32x4_mul(r2[idx], k2);
                    v128_t s3 = wasm_f32x4_mul(r3[idx], k3);
                    v128_t s4 = wasm_f32x4_mul(r4[idx], k4);
                    v128_t s5 = wasm_f32x4_mul(r5[idx], k5);
                    v128_t s6 = wasm_f32x4_mul(r6[idx], k6);
                    v128_t s7 = wasm_f32x4_mul(r7[idx], k7);
                    v128_t s8 = wasm_f32x4_mul(r8[idx], k8);

                    s = wasm_f32x4_add(s, wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_add(s0, s1), wasm_f32x4_add(s2, s3)), wasm_f32x4_add(wasm_f32x4_add(s4, s5), wasm_f32x4_add(s6, wasm_f32x4_add(s7, s8)))));
                }
                if constexpr (remain)
                {
                    v128_t k0 = wasm_f32x4_make((kptr[0] + count * vstep)[0], remain > 1 ? (kptr[0] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[0] + count * vstep)[2] : 0.0f, 0.0f);
                    v128_t k1 = wasm_f32x4_make((kptr[1] + count * vstep)[0], remain > 1 ? (kptr[1] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[1] + count * vstep)[2] : 0.0f, 0.0f);
                    v128_t k2 = wasm_f32x4_make((kptr[2] + count * vstep)[0], remain > 1 ? (kptr[2] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[2] + count * vstep)[2] : 0.0f, 0.0f);
                    v128_t k3 = wasm_f32x4_make((kptr[3] + count * vstep)[0], remain > 1 ? (kptr[3] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[3] + count * vstep)[2] : 0.0f, 0.0f);
                    v128_t k4 = wasm_f32x4_make((kptr[4] + count * vstep)[0], remain > 1 ? (kptr[4] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[4] + count * vstep)[2] : 0.0f, 0.0f);
                    v128_t k5 = wasm_f32x4_make((kptr[5] + count * vstep)[0], remain > 1 ? (kptr[5] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[5] + count * vstep)[2] : 0.0f, 0.0f);
                    v128_t k6 = wasm_f32x4_make((kptr[6] + count * vstep)[0], remain > 1 ? (kptr[6] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[6] + count * vstep)[2] : 0.0f, 0.0f);
                    v128_t k7 = wasm_f32x4_make((kptr[7] + count * vstep)[0], remain > 1 ? (kptr[7] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[7] + count * vstep)[2] : 0.0f, 0.0f);
                    v128_t k8 = wasm_f32x4_make((kptr[8] + count * vstep)[0], remain > 1 ? (kptr[8] + count * vstep)[1] : 0.0f, remain > 2 ? (kptr[8] + count * vstep)[2] : 0.0f, 0.0f);

                    v128_t s0 = wasm_f32x4_mul(r0[count], k0);
                    v128_t s1 = wasm_f32x4_mul(r1[count], k1);
                    v128_t s2 = wasm_f32x4_mul(r2[count], k2);
                    v128_t s3 = wasm_f32x4_mul(r3[count], k3);
                    v128_t s4 = wasm_f32x4_mul(r4[count], k4);
                    v128_t s5 = wasm_f32x4_mul(r5[count], k5);
                    v128_t s6 = wasm_f32x4_mul(r6[count], k6);
                    v128_t s7 = wasm_f32x4_mul(r7[count], k7);
                    v128_t s8 = wasm_f32x4_mul(r8[count], k8);

                    s = wasm_f32x4_add(s, wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_add(s0, s1), wasm_f32x4_add(s2, s3)), wasm_f32x4_add(wasm_f32x4_add(s4, s5), wasm_f32x4_add(s6, wasm_f32x4_add(s7, s8)))));
                }
                float sum = wasm_simd128_f32x4_hsum(s);
                if constexpr (residual) sum += out[n];
                out[n] = relu<OUT>(sum + biases[n]);
            }
        }, src, dst);
    }
    template <typename IN, typename OUT, int cout>
    inline void conv3x3_wasm_simd128_cin1(const Image& src, Image& dst, const float* const kernels, const float* const biases)
    {
        int w = src.width(), h = src.height();
        int step = src.stride() / src.elementSize();

        filter([=](const int i, const int j, const void* const sptr, void* const dptr) {
            auto in = static_cast<const IN*>(sptr);
            auto out = static_cast<OUT*>(dptr);

            auto sp = i < h - 1 ? +step : 0;
            auto sn = i > 0 ? -step : 0;
            auto cp = j < w - 1 ? +1 : 0;
            auto cn = j > 0 ? -1 : 0;

            auto tl = in + sn + cn, tc = in + sn, tr = in + sn + cp;
            auto ml = in + cn, mc = in, mr = in + cp;
            auto bl = in + sp + cn, bc = in + sp, br = in + sp + cp;

            v128_t r0 = wasm_f32x4_make(toFloat<IN>(*tl), toFloat<IN>(*tc), toFloat<IN>(*tr), toFloat<IN>(*ml));
            v128_t r4 = wasm_f32x4_make(toFloat<IN>(*mc), toFloat<IN>(*mr), toFloat<IN>(*bl), toFloat<IN>(*bc));
            auto r8 = toFloat<IN>(*br);

            for (int n = 0; n < cout; n++)
            {
                v128_t k0 = wasm_v128_load(kernels + n * 9 + 0);
                v128_t k4 = wasm_v128_load(kernels + n * 9 + 4);
                auto sum = wasm_simd128_f32x4_hsum(wasm_f32x4_add(wasm_f32x4_mul(r0, k0), wasm_f32x4_mul(r4, k4)));
                auto k8 = *(kernels + n * 9 + 8);
                out[n] = relu<OUT>(sum + k8 * r8 + biases[n]);
            }
        }, src, dst);
    }
    template <typename OUT, int cin, int cout>
    inline void deconv2x2_wasm_simd128_float(const Image& src, Image& dst, const float* const kernels)
    {
        filter([=](const int i, const int j, const void* const sptr, void* const dptr) {
            auto in = static_cast<const float*>(sptr);
            auto out = static_cast<OUT*>(dptr);

            const int index = ((i & 1) << 1) + (j & 1);

            constexpr int vstep = 4;
            constexpr int count = cin / vstep;
            constexpr int remain = cin % vstep;

            v128_t r[count + (remain ? 1 : 0)]{};
            for (int idx = 0; idx < count; idx++) r[idx] = wasm_v128_load(in + idx * vstep);
            if constexpr (remain) r[count] = wasm_f32x4_make((in + count * vstep)[0], remain > 1 ? (in + count * vstep)[1] : 0.0f, remain > 2 ? (in + count * vstep)[2] : 0.0f, 0.0f);

            for (int n = 0; n < cout; n++)
            {
                auto kptr = kernels + n * cin * 4 + cin * index;
                float sum = 0.0f;
                v128_t k[count + (remain ? 1 : 0)]{};
                for (int idx = 0; idx < count; idx++)
                {
                    k[idx] = wasm_v128_load(kptr + idx * vstep);
                    sum += wasm_simd128_f32x4_hsum(wasm_f32x4_mul(r[idx], k[idx]));
                }
                if constexpr (remain)
                {
                    k[count] = wasm_f32x4_make((kptr + count * vstep)[0], remain > 1 ? (kptr + count * vstep)[1] : 0.0f, remain > 2 ? (kptr + count * vstep)[2] : 0.0f, 0.0f);
                    sum += wasm_simd128_f32x4_hsum(wasm_f32x4_mul(r[count], k[count]));
                }
                out[n] = fromFloat<OUT>(sum);
            }
        }, src, dst);
    }

    void conv3x3_1to8_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_wasm_simd128_cin1<std::uint8_t, float, 8>(src, dst, kernels, biases);
            break;
        case Image::UInt16:
            conv3x3_wasm_simd128_cin1<std::uint16_t, float, 8>(src, dst, kernels, biases);
            break;
        case Image::Float32:
            conv3x3_wasm_simd128_cin1<float, float, 8>(src, dst, kernels, biases);
            break;
        }
    }
    void conv3x3_8to8_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_wasm_simd128_float<float, 8, 8>(src, dst, kernels, biases);
    }
    void conv3x3_residual_8to8_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_wasm_simd128_float<float, 8, 8, true>(src, dst, kernels, biases);
    }
    void deconv2x2_8to1_wasm_simd128(const Image& src, Image& dst, const float* kernels)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            deconv2x2_wasm_simd128_float<std::uint8_t, 8, 1>(src, dst, kernels);
            break;
        case Image::UInt16:
            deconv2x2_wasm_simd128_float<std::uint16_t, 8, 1>(src, dst, kernels);
            break;
        case Image::Float32:
            deconv2x2_wasm_simd128_float<float, 8, 1>(src, dst, kernels);
            break;
        }
    }
}
