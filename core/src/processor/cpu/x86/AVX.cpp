#include <immintrin.h>

#include "AC/Core/SIMD.hpp"
#include "AC/Core/Image.hpp"
#include "AC/Core/Util.hpp"

namespace ac::core::cpu
{
    static inline float avx_hsum_ps(const __m256& v) noexcept
    {
        __m128 v128 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 0x01));
        __m128 v64 = _mm_add_ps(v128, _mm_movehl_ps(v128, v128));
        __m128 v32 = _mm_add_ss(v64, _mm_movehdup_ps(v64));
        return _mm_cvtss_f32(v32);
    }

    template <typename OUT, int cin, int cout, bool fma, bool residual = false>
    inline void conv3x3_avx_float(const Image& src, Image& dst, const float* const kernels, const float* const biases)
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

            constexpr int vstep = 8;
            constexpr int count = cin / vstep;
            constexpr int remain = cin % vstep;

            __m256 r0[count + (remain ? 1 : 0)]{};
            __m256 r1[count + (remain ? 1 : 0)]{};
            __m256 r2[count + (remain ? 1 : 0)]{};
            __m256 r3[count + (remain ? 1 : 0)]{};
            __m256 r4[count + (remain ? 1 : 0)]{};
            __m256 r5[count + (remain ? 1 : 0)]{};
            __m256 r6[count + (remain ? 1 : 0)]{};
            __m256 r7[count + (remain ? 1 : 0)]{};
            __m256 r8[count + (remain ? 1 : 0)]{};

            for (int idx = 0; idx < count; idx++)
            {
                r0[idx] = _mm256_loadu_ps(tl + idx * vstep);
                r1[idx] = _mm256_loadu_ps(tc + idx * vstep);
                r2[idx] = _mm256_loadu_ps(tr + idx * vstep);
                r3[idx] = _mm256_loadu_ps(ml + idx * vstep);
                r4[idx] = _mm256_loadu_ps(mc + idx * vstep);
                r5[idx] = _mm256_loadu_ps(mr + idx * vstep);
                r6[idx] = _mm256_loadu_ps(bl + idx * vstep);
                r7[idx] = _mm256_loadu_ps(bc + idx * vstep);
                r8[idx] = _mm256_loadu_ps(br + idx * vstep);
            }
            if constexpr (remain)
            {
                r0[count] = _mm256_set_ps(0.0f, remain > 6 ? (tl + count * vstep)[6] : 0.0f, remain > 5 ? (tl + count * vstep)[5] : 0.0f, remain > 4 ? (tl + count * vstep)[4] : 0.0f, remain > 3 ? (tl + count * vstep)[3] : 0.0f, remain > 2 ? (tl + count * vstep)[2] : 0.0f, remain > 1 ? (tl + count * vstep)[1] : 0.0f, (tl + count * vstep)[0]);
                r1[count] = _mm256_set_ps(0.0f, remain > 6 ? (tc + count * vstep)[6] : 0.0f, remain > 5 ? (tc + count * vstep)[5] : 0.0f, remain > 4 ? (tc + count * vstep)[4] : 0.0f, remain > 3 ? (tc + count * vstep)[3] : 0.0f, remain > 2 ? (tc + count * vstep)[2] : 0.0f, remain > 1 ? (tc + count * vstep)[1] : 0.0f, (tc + count * vstep)[0]);
                r2[count] = _mm256_set_ps(0.0f, remain > 6 ? (tr + count * vstep)[6] : 0.0f, remain > 5 ? (tr + count * vstep)[5] : 0.0f, remain > 4 ? (tr + count * vstep)[4] : 0.0f, remain > 3 ? (tr + count * vstep)[3] : 0.0f, remain > 2 ? (tr + count * vstep)[2] : 0.0f, remain > 1 ? (tr + count * vstep)[1] : 0.0f, (tr + count * vstep)[0]);
                r3[count] = _mm256_set_ps(0.0f, remain > 6 ? (ml + count * vstep)[6] : 0.0f, remain > 5 ? (ml + count * vstep)[5] : 0.0f, remain > 4 ? (ml + count * vstep)[4] : 0.0f, remain > 3 ? (ml + count * vstep)[3] : 0.0f, remain > 2 ? (ml + count * vstep)[2] : 0.0f, remain > 1 ? (ml + count * vstep)[1] : 0.0f, (ml + count * vstep)[0]);
                r4[count] = _mm256_set_ps(0.0f, remain > 6 ? (mc + count * vstep)[6] : 0.0f, remain > 5 ? (mc + count * vstep)[5] : 0.0f, remain > 4 ? (mc + count * vstep)[4] : 0.0f, remain > 3 ? (mc + count * vstep)[3] : 0.0f, remain > 2 ? (mc + count * vstep)[2] : 0.0f, remain > 1 ? (mc + count * vstep)[1] : 0.0f, (mc + count * vstep)[0]);
                r5[count] = _mm256_set_ps(0.0f, remain > 6 ? (mr + count * vstep)[6] : 0.0f, remain > 5 ? (mr + count * vstep)[5] : 0.0f, remain > 4 ? (mr + count * vstep)[4] : 0.0f, remain > 3 ? (mr + count * vstep)[3] : 0.0f, remain > 2 ? (mr + count * vstep)[2] : 0.0f, remain > 1 ? (mr + count * vstep)[1] : 0.0f, (mr + count * vstep)[0]);
                r6[count] = _mm256_set_ps(0.0f, remain > 6 ? (bl + count * vstep)[6] : 0.0f, remain > 5 ? (bl + count * vstep)[5] : 0.0f, remain > 4 ? (bl + count * vstep)[4] : 0.0f, remain > 3 ? (bl + count * vstep)[3] : 0.0f, remain > 2 ? (bl + count * vstep)[2] : 0.0f, remain > 1 ? (bl + count * vstep)[1] : 0.0f, (bl + count * vstep)[0]);
                r7[count] = _mm256_set_ps(0.0f, remain > 6 ? (bc + count * vstep)[6] : 0.0f, remain > 5 ? (bc + count * vstep)[5] : 0.0f, remain > 4 ? (bc + count * vstep)[4] : 0.0f, remain > 3 ? (bc + count * vstep)[3] : 0.0f, remain > 2 ? (bc + count * vstep)[2] : 0.0f, remain > 1 ? (bc + count * vstep)[1] : 0.0f, (bc + count * vstep)[0]);
                r8[count] = _mm256_set_ps(0.0f, remain > 6 ? (br + count * vstep)[6] : 0.0f, remain > 5 ? (br + count * vstep)[5] : 0.0f, remain > 4 ? (br + count * vstep)[4] : 0.0f, remain > 3 ? (br + count * vstep)[3] : 0.0f, remain > 2 ? (br + count * vstep)[2] : 0.0f, remain > 1 ? (br + count * vstep)[1] : 0.0f, (br + count * vstep)[0]);
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
                __m256 s = _mm256_setzero_ps();
                for (int idx = 0; idx < count; idx++)
                {
                    __m256 k0 = _mm256_loadu_ps(kptr[0] + idx * vstep);
                    __m256 k1 = _mm256_loadu_ps(kptr[1] + idx * vstep);
                    __m256 k2 = _mm256_loadu_ps(kptr[2] + idx * vstep);
                    __m256 k3 = _mm256_loadu_ps(kptr[3] + idx * vstep);
                    __m256 k4 = _mm256_loadu_ps(kptr[4] + idx * vstep);
                    __m256 k5 = _mm256_loadu_ps(kptr[5] + idx * vstep);
                    __m256 k6 = _mm256_loadu_ps(kptr[6] + idx * vstep);
                    __m256 k7 = _mm256_loadu_ps(kptr[7] + idx * vstep);
                    __m256 k8 = _mm256_loadu_ps(kptr[8] + idx * vstep);

                    if constexpr (fma)
                    {
#                   ifdef AC_CORE_WITH_FMA
                        __m256& s0 = s;
                        __m256 s1 = _mm256_setzero_ps();
                        __m256 s2 = _mm256_setzero_ps();

                        s0 = _mm256_fmadd_ps(r0[idx], k0, s0);
                        s1 = _mm256_fmadd_ps(r1[idx], k1, s1);
                        s2 = _mm256_fmadd_ps(r2[idx], k2, s2);
                        s0 = _mm256_fmadd_ps(r3[idx], k3, s0);
                        s1 = _mm256_fmadd_ps(r4[idx], k4, s1);
                        s2 = _mm256_fmadd_ps(r5[idx], k5, s2);
                        s0 = _mm256_fmadd_ps(r6[idx], k6, s0);
                        s1 = _mm256_fmadd_ps(r7[idx], k7, s1);
                        s2 = _mm256_fmadd_ps(r8[idx], k8, s2);

                        s0 = _mm256_add_ps(s0, _mm256_add_ps(s1, s2));
#                   endif
                    }
                    else
                    {
                        __m256 s0 = _mm256_mul_ps(r0[idx], k0);
                        __m256 s1 = _mm256_mul_ps(r1[idx], k1);
                        __m256 s2 = _mm256_mul_ps(r2[idx], k2);
                        __m256 s3 = _mm256_mul_ps(r3[idx], k3);
                        __m256 s4 = _mm256_mul_ps(r4[idx], k4);
                        __m256 s5 = _mm256_mul_ps(r5[idx], k5);
                        __m256 s6 = _mm256_mul_ps(r6[idx], k6);
                        __m256 s7 = _mm256_mul_ps(r7[idx], k7);
                        __m256 s8 = _mm256_mul_ps(r8[idx], k8);

                        s = _mm256_add_ps(s, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)), _mm256_add_ps(_mm256_add_ps(s4, s5), _mm256_add_ps(s6, _mm256_add_ps(s7, s8)))));
                    }

                }
                if constexpr (remain)
                {
                    __m256 k0 = _mm256_set_ps(0.0f, remain > 6 ? (kptr[0] + count * vstep)[6] : 0.0f, remain > 5 ? (kptr[0] + count * vstep)[5] : 0.0f, remain > 4 ? (kptr[0] + count * vstep)[4] : 0.0f, remain > 3 ? (kptr[0] + count * vstep)[3] : 0.0f, remain > 2 ? (kptr[0] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[0] + count * vstep)[1] : 0.0f, (kptr[0] + count * vstep)[0]);
                    __m256 k1 = _mm256_set_ps(0.0f, remain > 6 ? (kptr[1] + count * vstep)[6] : 0.0f, remain > 5 ? (kptr[1] + count * vstep)[5] : 0.0f, remain > 4 ? (kptr[1] + count * vstep)[4] : 0.0f, remain > 3 ? (kptr[1] + count * vstep)[3] : 0.0f, remain > 2 ? (kptr[1] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[1] + count * vstep)[1] : 0.0f, (kptr[1] + count * vstep)[0]);
                    __m256 k2 = _mm256_set_ps(0.0f, remain > 6 ? (kptr[2] + count * vstep)[6] : 0.0f, remain > 5 ? (kptr[2] + count * vstep)[5] : 0.0f, remain > 4 ? (kptr[2] + count * vstep)[4] : 0.0f, remain > 3 ? (kptr[2] + count * vstep)[3] : 0.0f, remain > 2 ? (kptr[2] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[2] + count * vstep)[1] : 0.0f, (kptr[2] + count * vstep)[0]);
                    __m256 k3 = _mm256_set_ps(0.0f, remain > 6 ? (kptr[3] + count * vstep)[6] : 0.0f, remain > 5 ? (kptr[3] + count * vstep)[5] : 0.0f, remain > 4 ? (kptr[3] + count * vstep)[4] : 0.0f, remain > 3 ? (kptr[3] + count * vstep)[3] : 0.0f, remain > 2 ? (kptr[3] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[3] + count * vstep)[1] : 0.0f, (kptr[3] + count * vstep)[0]);
                    __m256 k4 = _mm256_set_ps(0.0f, remain > 6 ? (kptr[4] + count * vstep)[6] : 0.0f, remain > 5 ? (kptr[4] + count * vstep)[5] : 0.0f, remain > 4 ? (kptr[4] + count * vstep)[4] : 0.0f, remain > 3 ? (kptr[4] + count * vstep)[3] : 0.0f, remain > 2 ? (kptr[4] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[4] + count * vstep)[1] : 0.0f, (kptr[4] + count * vstep)[0]);
                    __m256 k5 = _mm256_set_ps(0.0f, remain > 6 ? (kptr[5] + count * vstep)[6] : 0.0f, remain > 5 ? (kptr[5] + count * vstep)[5] : 0.0f, remain > 4 ? (kptr[5] + count * vstep)[4] : 0.0f, remain > 3 ? (kptr[5] + count * vstep)[3] : 0.0f, remain > 2 ? (kptr[5] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[5] + count * vstep)[1] : 0.0f, (kptr[5] + count * vstep)[0]);
                    __m256 k6 = _mm256_set_ps(0.0f, remain > 6 ? (kptr[6] + count * vstep)[6] : 0.0f, remain > 5 ? (kptr[6] + count * vstep)[5] : 0.0f, remain > 4 ? (kptr[6] + count * vstep)[4] : 0.0f, remain > 3 ? (kptr[6] + count * vstep)[3] : 0.0f, remain > 2 ? (kptr[6] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[6] + count * vstep)[1] : 0.0f, (kptr[6] + count * vstep)[0]);
                    __m256 k7 = _mm256_set_ps(0.0f, remain > 6 ? (kptr[7] + count * vstep)[6] : 0.0f, remain > 5 ? (kptr[7] + count * vstep)[5] : 0.0f, remain > 4 ? (kptr[7] + count * vstep)[4] : 0.0f, remain > 3 ? (kptr[7] + count * vstep)[3] : 0.0f, remain > 2 ? (kptr[7] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[7] + count * vstep)[1] : 0.0f, (kptr[7] + count * vstep)[0]);
                    __m256 k8 = _mm256_set_ps(0.0f, remain > 6 ? (kptr[8] + count * vstep)[6] : 0.0f, remain > 5 ? (kptr[8] + count * vstep)[5] : 0.0f, remain > 4 ? (kptr[8] + count * vstep)[4] : 0.0f, remain > 3 ? (kptr[8] + count * vstep)[3] : 0.0f, remain > 2 ? (kptr[8] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[8] + count * vstep)[1] : 0.0f, (kptr[8] + count * vstep)[0]);

                    if constexpr (fma)
                    {
#                   ifdef AC_CORE_WITH_FMA
                        __m256& s0 = s;
                        __m256 s1 = _mm256_setzero_ps();
                        __m256 s2 = _mm256_setzero_ps();

                        s0 = _mm256_fmadd_ps(r0[count], k0, s0);
                        s1 = _mm256_fmadd_ps(r1[count], k1, s1);
                        s2 = _mm256_fmadd_ps(r2[count], k2, s2);
                        s0 = _mm256_fmadd_ps(r3[count], k3, s0);
                        s1 = _mm256_fmadd_ps(r4[count], k4, s1);
                        s2 = _mm256_fmadd_ps(r5[count], k5, s2);
                        s0 = _mm256_fmadd_ps(r6[count], k6, s0);
                        s1 = _mm256_fmadd_ps(r7[count], k7, s1);
                        s2 = _mm256_fmadd_ps(r8[count], k8, s2);

                        s0 = _mm256_add_ps(s0, _mm256_add_ps(s1, s2));
#                   endif
                    }
                    else
                    {
                        __m256 s0 = _mm256_mul_ps(r0[count], k0);
                        __m256 s1 = _mm256_mul_ps(r1[count], k1);
                        __m256 s2 = _mm256_mul_ps(r2[count], k2);
                        __m256 s3 = _mm256_mul_ps(r3[count], k3);
                        __m256 s4 = _mm256_mul_ps(r4[count], k4);
                        __m256 s5 = _mm256_mul_ps(r5[count], k5);
                        __m256 s6 = _mm256_mul_ps(r6[count], k6);
                        __m256 s7 = _mm256_mul_ps(r7[count], k7);
                        __m256 s8 = _mm256_mul_ps(r8[count], k8);

                        s = _mm256_add_ps(s, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)), _mm256_add_ps(_mm256_add_ps(s4, s5), _mm256_add_ps(s6, _mm256_add_ps(s7, s8)))));
                    }
                }
                float sum = avx_hsum_ps(s);
                if constexpr (residual) sum += out[n];
                out[n] = relu<OUT>(sum + biases[n]);
            }
        }, src, dst);
    }
    template <typename IN, typename OUT, int cout>
    inline void conv3x3_avx_cin1(const Image& src, Image& dst, const float* const kernels, const float* const biases)
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

            __m256 r = _mm256_set_ps(
                toFloat<IN>(*bc),
                toFloat<IN>(*bl),
                toFloat<IN>(*mr),
                toFloat<IN>(*mc),
                toFloat<IN>(*ml),
                toFloat<IN>(*tr),
                toFloat<IN>(*tc),
                toFloat<IN>(*tl));
            auto r8 = toFloat<IN>(*br);

            for (int n = 0; n < cout; n++)
            {
                __m256 k = _mm256_loadu_ps(kernels + n * 9 + 0);
                auto sum = avx_hsum_ps(_mm256_mul_ps(r, k));
                auto k8 = *(kernels + n * 9 + 8);
                out[n] = relu<OUT>(sum + k8 * r8 + biases[n]);
            }
        }, src, dst);
    }
    template <typename OUT, int cin, int cout>
    inline void deconv2x2_avx_float(const Image& src, Image& dst, const float* const kernels)
    {
        filter([=](const int i, const int j, const void* const sptr, void* const dptr) {
            auto in = static_cast<const float*>(sptr);
            auto out = static_cast<OUT*>(dptr);

            const int index = ((i & 1) << 1) + (j & 1);

            constexpr int vstep = 8;
            constexpr int count = cin / vstep;
            constexpr int remain = cin % vstep;

            __m256 r[count + (remain ? 1 : 0)]{};
            for (int idx = 0; idx < count; idx++)  r[idx] = _mm256_loadu_ps(in + idx * vstep);
            if constexpr (remain) r[count] = _mm256_set_ps(0.0f, remain > 6 ? (in + count * vstep)[6] : 0.0f, remain > 5 ? (in + count * vstep)[5] : 0.0f, remain > 4 ? (in + count * vstep)[4] : 0.0f, remain > 3 ? (in + count * vstep)[3] : 0.0f, remain > 2 ? (in + count * vstep)[2] : 0.0f, remain > 1 ? (in + count * vstep)[1] : 0.0f, (in + count * vstep)[0]);

            for (int n = 0; n < cout; n++)
            {
                auto kptr = kernels + n * cin * 4 + cin * index;
                float sum = 0.0f;
                __m256 k[count + (remain ? 1 : 0)]{};
                for (int idx = 0; idx < count; idx++)
                {
                    k[idx] = _mm256_loadu_ps(kptr + idx * vstep);
                    sum += avx_hsum_ps(_mm256_mul_ps(r[idx], k[idx]));
                }
                if constexpr (remain)
                {
                    k[count] = _mm256_set_ps(0.0f, remain > 6 ? (kptr + count * vstep)[6] : 0.0f, remain > 5 ? (kptr + count * vstep)[5] : 0.0f, remain > 4 ? (kptr + count * vstep)[4] : 0.0f, remain > 3 ? (kptr + count * vstep)[3] : 0.0f, remain > 2 ? (kptr + count * vstep)[2] : 0.0f, remain > 1 ? (kptr + count * vstep)[1] : 0.0f, (kptr + count * vstep)[0]);
                    sum += avx_hsum_ps(_mm256_mul_ps(r[count], k[count]));
                }
                out[n] = fromFloat<OUT>(sum);
            }
        }, src, dst);
    }

    void conv3x3_1to8_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_avx_cin1<std::uint8_t, float, 8>(src, dst, kernels, biases);
            break;
        case Image::UInt16:
            conv3x3_avx_cin1<std::uint16_t, float, 8>(src, dst, kernels, biases);
            break;
        case Image::Float32:
            conv3x3_avx_cin1<float, float, 8>(src, dst, kernels, biases);
            break;
        }
    }
    void conv3x3_8to8_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_avx_float<float, 8, 8, true>(src, dst, kernels, biases);
        else
#   endif
            conv3x3_avx_float<float, 8, 8, false>(src, dst, kernels, biases);
    }
    void conv3x3_residual_8to8_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_avx_float<float, 8, 8, true, true>(src, dst, kernels, biases);
        else
#   endif
            conv3x3_avx_float<float, 8, 8, false, true>(src, dst, kernels, biases);
    }
    void deconv2x2_8to1_avx(const Image& src, Image& dst, const float* kernels)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            deconv2x2_avx_float<std::uint8_t, 8, 1>(src, dst, kernels);
            break;
        case Image::UInt16:
            deconv2x2_avx_float<std::uint16_t, 8, 1>(src, dst, kernels);
            break;
        case Image::Float32:
            deconv2x2_avx_float<float, 8, 1>(src, dst, kernels);
            break;
        }
    }
}
