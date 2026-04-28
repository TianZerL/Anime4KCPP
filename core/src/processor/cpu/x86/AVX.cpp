#include <immintrin.h>

#include "AC/Core/SIMD.hpp"
#include "AC/Core/Image.hpp"
#include "AC/Core/Util.hpp"

#include "AC/Core/Internal/Processor/CPU/Common.hpp"

namespace ac::core::cpu
{
    static inline float avx_hsum_ps(const __m256& v) noexcept
    {
        __m128 v128 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 0x01));
        __m128 v64 = _mm_add_ps(v128, _mm_movehl_ps(v128, v128));
        __m128 v32 = _mm_add_ss(v64, _mm_movehdup_ps(v64));
        return _mm_cvtss_f32(v32);
    }

    template <typename IN, int cout, typename ActiveFunc>
    inline void conv3x3_cin1_avx(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc)
    {
        util::parallelFor(0, src.height(), [&](const int i) {
            auto tp = i > 0 ? 1 : 0;
            auto bp = i < src.height() - 1 ? 1 : 0;

            for (int j = 0; j < src.width(); j++)
            {
                auto out = static_cast<float*>(dst.ptr(j, i));

                auto lp = j > 0 ? 1 : 0;
                auto rp = j < src.width() - 1 ? 1 : 0;

                __m256 r = _mm256_set_ps(
                    toFloat(*static_cast<const IN*>(src.ptr(j     , i + bp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j - lp, i + bp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + rp, i     ))),
                    toFloat(*static_cast<const IN*>(src.ptr(j     , i     ))),
                    toFloat(*static_cast<const IN*>(src.ptr(j - lp, i     ))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + rp, i - tp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j     , i - tp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j - lp, i - tp))));
                auto r8 = toFloat(*static_cast<const IN*>(src.ptr(j + rp, i + bp)));

                for (int n = 0; n < cout; n++)
                {
                    __m256 k = _mm256_loadu_ps(kernels + n * 9 + 0);
                    auto sum = avx_hsum_ps(_mm256_mul_ps(r, k));
                    auto k8 = *(kernels + n * 9 + 8);
                    out[n] = activeFunc(sum + r8 * k8 + biases[n], n);
                }
            }
        });
    }
    template <typename IN, int cout, bool fma, typename ActiveFunc>
    inline void conv5x5_cin1_avx(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc)
    {
        util::parallelFor(0, src.height(), [&](const int i) {
            int ioffsets[5] = { i > 1 ? -2 : (i > 0 ? -1 : 0) , i > 0 ? -1 : 0 , 0, i < src.height() - 1 ? 1 : 0, i < src.height() - 2 ? 2 : (i < src.height() - 1 ? 1 : 0) };

            for (int j = 0; j < src.width(); j++)
            {
                auto out = static_cast<float*>(dst.ptr(j, i));

                int joffsets[5] = { j > 1 ? -2 : (j > 0 ? -1 : 0), j > 0 ? -1 : 0, 0, j < src.width() - 1 ? 1 : 0 ,j < src.width() - 2 ? 2 : (j < src.width() - 1 ? 1 : 0) };

                __m256 r0 = _mm256_set_ps(
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[2], i + ioffsets[1]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[1], i + ioffsets[1]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[0], i + ioffsets[1]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[4], i + ioffsets[0]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[3], i + ioffsets[0]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[2], i + ioffsets[0]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[1], i + ioffsets[0]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[0], i + ioffsets[0]))));
                __m256 r8 = _mm256_set_ps(
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[0], i + ioffsets[3]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[4], i + ioffsets[2]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[3], i + ioffsets[2]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[2], i + ioffsets[2]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[1], i + ioffsets[2]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[0], i + ioffsets[2]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[4], i + ioffsets[1]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[3], i + ioffsets[1]))));
                __m256 r16 = _mm256_set_ps(
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[3], i + ioffsets[4]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[2], i + ioffsets[4]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[1], i + ioffsets[4]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[0], i + ioffsets[4]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[4], i + ioffsets[3]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[3], i + ioffsets[3]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[2], i + ioffsets[3]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[1], i + ioffsets[3]))));
                auto r24 = toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[4], i + ioffsets[4])));

                for (int n = 0; n < cout; n++)
                {
                    __m256 k0 = _mm256_loadu_ps(kernels + n * 25 + 0);
                    __m256 k8 = _mm256_loadu_ps(kernels + n * 25 + 8);
                    __m256 k16 = _mm256_loadu_ps(kernels + n * 25 + 16);

                    auto sum = biases[n];

#           ifdef AC_CORE_WITH_FMA
                    if constexpr (fma)
                        sum += avx_hsum_ps(_mm256_fmadd_ps(r0, k0, _mm256_fmadd_ps(r8, k8, _mm256_mul_ps(r16, k16))));
                    else
#           endif
                        sum += avx_hsum_ps(_mm256_add_ps(_mm256_mul_ps(r0, k0), _mm256_add_ps(_mm256_mul_ps(r8, k8), _mm256_mul_ps(r16, k16))));

                    auto k24 = *(kernels + n * 25 + 24);
                    out[n] = activeFunc(sum + r24 * k24, n);
                }
            }
        });
    }
    template <typename OUT, int cin, int cout>
    inline void deconv2x2_float_avx(const Image& src, Image& dst, const float* const kernels)
    {
        filter([=](const int i, const int j, const void* const sptr, void* const dptr) {
            auto in = static_cast<const float*>(sptr);
            auto out = static_cast<OUT*>(dptr);

            auto index = ((i & 1) << 1) + (j & 1);

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

    template <bool fma>
    struct ConvImplAVX
    {
        template <int cin, int cout>
        static void conv1x1(const float* rptr[], float* const out, const float* const kernels, const float* const biases) noexcept
        {
            constexpr int vstep = 8;
            constexpr int count = cin / vstep;
            constexpr int remain = cin % vstep;

            std::memcpy(out, biases, sizeof(float) * cout);

            for (int idx = 0; idx < count; idx++)
            {
                __m256 r = _mm256_loadu_ps(rptr[0] + idx * vstep);

                for (int n = 0; n < cout; n++)
                {
                    auto kptr = kernels + n * cin;

                    __m256 k = _mm256_loadu_ps(kptr + idx * vstep);

                    out[n] += avx_hsum_ps(_mm256_mul_ps(r, k));
                }
            }
            if constexpr (remain)
            {
                __m256 r = _mm256_set_ps(0.0f, remain > 6 ? (rptr[0] + count * vstep)[6] : 0.0f, remain > 5 ? (rptr[0] + count * vstep)[5] : 0.0f, remain > 4 ? (rptr[0] + count * vstep)[4] : 0.0f, remain > 3 ? (rptr[0] + count * vstep)[3] : 0.0f, remain > 2 ? (rptr[0] + count * vstep)[2] : 0.0f, remain > 1 ? (rptr[0] + count * vstep)[1] : 0.0f, (rptr[0] + count * vstep)[0]);

                for (int n = 0; n < cout; n++)
                {
                    auto kptr = kernels + n * cin;

                    __m256 k = _mm256_set_ps(0.0f, remain > 6 ? (kptr + count * vstep)[6] : 0.0f, remain > 5 ? (kptr + count * vstep)[5] : 0.0f, remain > 4 ? (kptr + count * vstep)[4] : 0.0f, remain > 3 ? (kptr + count * vstep)[3] : 0.0f, remain > 2 ? (kptr + count * vstep)[2] : 0.0f, remain > 1 ? (kptr + count * vstep)[1] : 0.0f, (kptr + count * vstep)[0]);

                    out[n] += avx_hsum_ps(_mm256_mul_ps(r, k));
                }
            }
        }

        template <int cin, int cout>
        static void conv3x3(const float* rptr[], float* const out, const float* const kernels, const float* const biases) noexcept
        {
            constexpr int vstep = 8;
            constexpr int count = cin / vstep;
            constexpr int remain = cin % vstep;

            std::memcpy(out, biases, sizeof(float) * cout);

            for (int idx = 0; idx < count; idx++)
            {
                __m256 r0 = _mm256_loadu_ps(rptr[0] + idx * vstep);
                __m256 r1 = _mm256_loadu_ps(rptr[1] + idx * vstep);
                __m256 r2 = _mm256_loadu_ps(rptr[2] + idx * vstep);
                __m256 r3 = _mm256_loadu_ps(rptr[3] + idx * vstep);
                __m256 r4 = _mm256_loadu_ps(rptr[4] + idx * vstep);
                __m256 r5 = _mm256_loadu_ps(rptr[5] + idx * vstep);
                __m256 r6 = _mm256_loadu_ps(rptr[6] + idx * vstep);
                __m256 r7 = _mm256_loadu_ps(rptr[7] + idx * vstep);
                __m256 r8 = _mm256_loadu_ps(rptr[8] + idx * vstep);

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

                    __m256 s0 = _mm256_setzero_ps();
                    __m256 s1 = _mm256_setzero_ps();
                    __m256 s2 = _mm256_setzero_ps();

                    __m256 k0 = _mm256_loadu_ps(kptr[0] + idx * vstep);
                    __m256 k1 = _mm256_loadu_ps(kptr[1] + idx * vstep);
                    __m256 k2 = _mm256_loadu_ps(kptr[2] + idx * vstep);
                    __m256 k3 = _mm256_loadu_ps(kptr[3] + idx * vstep);
                    __m256 k4 = _mm256_loadu_ps(kptr[4] + idx * vstep);
                    __m256 k5 = _mm256_loadu_ps(kptr[5] + idx * vstep);
                    __m256 k6 = _mm256_loadu_ps(kptr[6] + idx * vstep);
                    __m256 k7 = _mm256_loadu_ps(kptr[7] + idx * vstep);
                    __m256 k8 = _mm256_loadu_ps(kptr[8] + idx * vstep);

#               ifdef AC_CORE_WITH_FMA
                    if constexpr (fma)
                    {
                        s0 = _mm256_fmadd_ps(r0, k0, s0);
                        s1 = _mm256_fmadd_ps(r1, k1, s1);
                        s2 = _mm256_fmadd_ps(r2, k2, s2);
                        s0 = _mm256_fmadd_ps(r3, k3, s0);
                        s1 = _mm256_fmadd_ps(r4, k4, s1);
                        s2 = _mm256_fmadd_ps(r5, k5, s2);
                        s0 = _mm256_fmadd_ps(r6, k6, s0);
                        s1 = _mm256_fmadd_ps(r7, k7, s1);
                        s2 = _mm256_fmadd_ps(r8, k8, s2);
                    }
                    else
#               endif
                    {
                        s0 = _mm256_add_ps(_mm256_mul_ps(r0, k0), s0);
                        s1 = _mm256_add_ps(_mm256_mul_ps(r1, k1), s1);
                        s2 = _mm256_add_ps(_mm256_mul_ps(r2, k2), s2);
                        s0 = _mm256_add_ps(_mm256_mul_ps(r3, k3), s0);
                        s1 = _mm256_add_ps(_mm256_mul_ps(r4, k4), s1);
                        s2 = _mm256_add_ps(_mm256_mul_ps(r5, k5), s2);
                        s0 = _mm256_add_ps(_mm256_mul_ps(r6, k6), s0);
                        s1 = _mm256_add_ps(_mm256_mul_ps(r7, k7), s1);
                        s2 = _mm256_add_ps(_mm256_mul_ps(r8, k8), s2);
                    }

                    out[n] += avx_hsum_ps(_mm256_add_ps(s0, _mm256_add_ps(s1, s2)));
                }
            }
            if constexpr (remain)
            {
                __m256 r0 = _mm256_set_ps(0.0f, remain > 6 ? (rptr[0] + count * vstep)[6] : 0.0f, remain > 5 ? (rptr[0] + count * vstep)[5] : 0.0f, remain > 4 ? (rptr[0] + count * vstep)[4] : 0.0f, remain > 3 ? (rptr[0] + count * vstep)[3] : 0.0f, remain > 2 ? (rptr[0] + count * vstep)[2] : 0.0f, remain > 1 ? (rptr[0] + count * vstep)[1] : 0.0f, (rptr[0] + count * vstep)[0]);
                __m256 r1 = _mm256_set_ps(0.0f, remain > 6 ? (rptr[1] + count * vstep)[6] : 0.0f, remain > 5 ? (rptr[1] + count * vstep)[5] : 0.0f, remain > 4 ? (rptr[1] + count * vstep)[4] : 0.0f, remain > 3 ? (rptr[1] + count * vstep)[3] : 0.0f, remain > 2 ? (rptr[1] + count * vstep)[2] : 0.0f, remain > 1 ? (rptr[1] + count * vstep)[1] : 0.0f, (rptr[1] + count * vstep)[0]);
                __m256 r2 = _mm256_set_ps(0.0f, remain > 6 ? (rptr[2] + count * vstep)[6] : 0.0f, remain > 5 ? (rptr[2] + count * vstep)[5] : 0.0f, remain > 4 ? (rptr[2] + count * vstep)[4] : 0.0f, remain > 3 ? (rptr[2] + count * vstep)[3] : 0.0f, remain > 2 ? (rptr[2] + count * vstep)[2] : 0.0f, remain > 1 ? (rptr[2] + count * vstep)[1] : 0.0f, (rptr[2] + count * vstep)[0]);
                __m256 r3 = _mm256_set_ps(0.0f, remain > 6 ? (rptr[3] + count * vstep)[6] : 0.0f, remain > 5 ? (rptr[3] + count * vstep)[5] : 0.0f, remain > 4 ? (rptr[3] + count * vstep)[4] : 0.0f, remain > 3 ? (rptr[3] + count * vstep)[3] : 0.0f, remain > 2 ? (rptr[3] + count * vstep)[2] : 0.0f, remain > 1 ? (rptr[3] + count * vstep)[1] : 0.0f, (rptr[3] + count * vstep)[0]);
                __m256 r4 = _mm256_set_ps(0.0f, remain > 6 ? (rptr[4] + count * vstep)[6] : 0.0f, remain > 5 ? (rptr[4] + count * vstep)[5] : 0.0f, remain > 4 ? (rptr[4] + count * vstep)[4] : 0.0f, remain > 3 ? (rptr[4] + count * vstep)[3] : 0.0f, remain > 2 ? (rptr[4] + count * vstep)[2] : 0.0f, remain > 1 ? (rptr[4] + count * vstep)[1] : 0.0f, (rptr[4] + count * vstep)[0]);
                __m256 r5 = _mm256_set_ps(0.0f, remain > 6 ? (rptr[5] + count * vstep)[6] : 0.0f, remain > 5 ? (rptr[5] + count * vstep)[5] : 0.0f, remain > 4 ? (rptr[5] + count * vstep)[4] : 0.0f, remain > 3 ? (rptr[5] + count * vstep)[3] : 0.0f, remain > 2 ? (rptr[5] + count * vstep)[2] : 0.0f, remain > 1 ? (rptr[5] + count * vstep)[1] : 0.0f, (rptr[5] + count * vstep)[0]);
                __m256 r6 = _mm256_set_ps(0.0f, remain > 6 ? (rptr[6] + count * vstep)[6] : 0.0f, remain > 5 ? (rptr[6] + count * vstep)[5] : 0.0f, remain > 4 ? (rptr[6] + count * vstep)[4] : 0.0f, remain > 3 ? (rptr[6] + count * vstep)[3] : 0.0f, remain > 2 ? (rptr[6] + count * vstep)[2] : 0.0f, remain > 1 ? (rptr[6] + count * vstep)[1] : 0.0f, (rptr[6] + count * vstep)[0]);
                __m256 r7 = _mm256_set_ps(0.0f, remain > 6 ? (rptr[7] + count * vstep)[6] : 0.0f, remain > 5 ? (rptr[7] + count * vstep)[5] : 0.0f, remain > 4 ? (rptr[7] + count * vstep)[4] : 0.0f, remain > 3 ? (rptr[7] + count * vstep)[3] : 0.0f, remain > 2 ? (rptr[7] + count * vstep)[2] : 0.0f, remain > 1 ? (rptr[7] + count * vstep)[1] : 0.0f, (rptr[7] + count * vstep)[0]);
                __m256 r8 = _mm256_set_ps(0.0f, remain > 6 ? (rptr[8] + count * vstep)[6] : 0.0f, remain > 5 ? (rptr[8] + count * vstep)[5] : 0.0f, remain > 4 ? (rptr[8] + count * vstep)[4] : 0.0f, remain > 3 ? (rptr[8] + count * vstep)[3] : 0.0f, remain > 2 ? (rptr[8] + count * vstep)[2] : 0.0f, remain > 1 ? (rptr[8] + count * vstep)[1] : 0.0f, (rptr[8] + count * vstep)[0]);

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

                    __m256 s0 = _mm256_setzero_ps();
                    __m256 s1 = _mm256_setzero_ps();
                    __m256 s2 = _mm256_setzero_ps();

                    __m256 k0 = _mm256_set_ps(0.0f, remain > 6 ? (kptr[0] + count * vstep)[6] : 0.0f, remain > 5 ? (kptr[0] + count * vstep)[5] : 0.0f, remain > 4 ? (kptr[0] + count * vstep)[4] : 0.0f, remain > 3 ? (kptr[0] + count * vstep)[3] : 0.0f, remain > 2 ? (kptr[0] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[0] + count * vstep)[1] : 0.0f, (kptr[0] + count * vstep)[0]);
                    __m256 k1 = _mm256_set_ps(0.0f, remain > 6 ? (kptr[1] + count * vstep)[6] : 0.0f, remain > 5 ? (kptr[1] + count * vstep)[5] : 0.0f, remain > 4 ? (kptr[1] + count * vstep)[4] : 0.0f, remain > 3 ? (kptr[1] + count * vstep)[3] : 0.0f, remain > 2 ? (kptr[1] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[1] + count * vstep)[1] : 0.0f, (kptr[1] + count * vstep)[0]);
                    __m256 k2 = _mm256_set_ps(0.0f, remain > 6 ? (kptr[2] + count * vstep)[6] : 0.0f, remain > 5 ? (kptr[2] + count * vstep)[5] : 0.0f, remain > 4 ? (kptr[2] + count * vstep)[4] : 0.0f, remain > 3 ? (kptr[2] + count * vstep)[3] : 0.0f, remain > 2 ? (kptr[2] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[2] + count * vstep)[1] : 0.0f, (kptr[2] + count * vstep)[0]);
                    __m256 k3 = _mm256_set_ps(0.0f, remain > 6 ? (kptr[3] + count * vstep)[6] : 0.0f, remain > 5 ? (kptr[3] + count * vstep)[5] : 0.0f, remain > 4 ? (kptr[3] + count * vstep)[4] : 0.0f, remain > 3 ? (kptr[3] + count * vstep)[3] : 0.0f, remain > 2 ? (kptr[3] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[3] + count * vstep)[1] : 0.0f, (kptr[3] + count * vstep)[0]);
                    __m256 k4 = _mm256_set_ps(0.0f, remain > 6 ? (kptr[4] + count * vstep)[6] : 0.0f, remain > 5 ? (kptr[4] + count * vstep)[5] : 0.0f, remain > 4 ? (kptr[4] + count * vstep)[4] : 0.0f, remain > 3 ? (kptr[4] + count * vstep)[3] : 0.0f, remain > 2 ? (kptr[4] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[4] + count * vstep)[1] : 0.0f, (kptr[4] + count * vstep)[0]);
                    __m256 k5 = _mm256_set_ps(0.0f, remain > 6 ? (kptr[5] + count * vstep)[6] : 0.0f, remain > 5 ? (kptr[5] + count * vstep)[5] : 0.0f, remain > 4 ? (kptr[5] + count * vstep)[4] : 0.0f, remain > 3 ? (kptr[5] + count * vstep)[3] : 0.0f, remain > 2 ? (kptr[5] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[5] + count * vstep)[1] : 0.0f, (kptr[5] + count * vstep)[0]);
                    __m256 k6 = _mm256_set_ps(0.0f, remain > 6 ? (kptr[6] + count * vstep)[6] : 0.0f, remain > 5 ? (kptr[6] + count * vstep)[5] : 0.0f, remain > 4 ? (kptr[6] + count * vstep)[4] : 0.0f, remain > 3 ? (kptr[6] + count * vstep)[3] : 0.0f, remain > 2 ? (kptr[6] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[6] + count * vstep)[1] : 0.0f, (kptr[6] + count * vstep)[0]);
                    __m256 k7 = _mm256_set_ps(0.0f, remain > 6 ? (kptr[7] + count * vstep)[6] : 0.0f, remain > 5 ? (kptr[7] + count * vstep)[5] : 0.0f, remain > 4 ? (kptr[7] + count * vstep)[4] : 0.0f, remain > 3 ? (kptr[7] + count * vstep)[3] : 0.0f, remain > 2 ? (kptr[7] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[7] + count * vstep)[1] : 0.0f, (kptr[7] + count * vstep)[0]);
                    __m256 k8 = _mm256_set_ps(0.0f, remain > 6 ? (kptr[8] + count * vstep)[6] : 0.0f, remain > 5 ? (kptr[8] + count * vstep)[5] : 0.0f, remain > 4 ? (kptr[8] + count * vstep)[4] : 0.0f, remain > 3 ? (kptr[8] + count * vstep)[3] : 0.0f, remain > 2 ? (kptr[8] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[8] + count * vstep)[1] : 0.0f, (kptr[8] + count * vstep)[0]);

#               ifdef AC_CORE_WITH_FMA
                    if constexpr (fma)
                    {
                        s0 = _mm256_fmadd_ps(r0, k0, s0);
                        s1 = _mm256_fmadd_ps(r1, k1, s1);
                        s2 = _mm256_fmadd_ps(r2, k2, s2);
                        s0 = _mm256_fmadd_ps(r3, k3, s0);
                        s1 = _mm256_fmadd_ps(r4, k4, s1);
                        s2 = _mm256_fmadd_ps(r5, k5, s2);
                        s0 = _mm256_fmadd_ps(r6, k6, s0);
                        s1 = _mm256_fmadd_ps(r7, k7, s1);
                        s2 = _mm256_fmadd_ps(r8, k8, s2);
                    }
                    else
#               endif
                    {
                        s0 = _mm256_add_ps(_mm256_mul_ps(r0, k0), s0);
                        s1 = _mm256_add_ps(_mm256_mul_ps(r1, k1), s1);
                        s2 = _mm256_add_ps(_mm256_mul_ps(r2, k2), s2);
                        s0 = _mm256_add_ps(_mm256_mul_ps(r3, k3), s0);
                        s1 = _mm256_add_ps(_mm256_mul_ps(r4, k4), s1);
                        s2 = _mm256_add_ps(_mm256_mul_ps(r5, k5), s2);
                        s0 = _mm256_add_ps(_mm256_mul_ps(r6, k6), s0);
                        s1 = _mm256_add_ps(_mm256_mul_ps(r7, k7), s1);
                        s2 = _mm256_add_ps(_mm256_mul_ps(r8, k8), s2);
                    }

                    out[n] += avx_hsum_ps(_mm256_add_ps(s0, _mm256_add_ps(s1, s2)));
                }
            }
        }
    };

    void conv3x3_1to8_relu_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1_avx<std::uint8_t, 8>(src, dst, kernels, biases, ReLU{});
            break;
        case Image::UInt16:
            conv3x3_cin1_avx<std::uint16_t, 8>(src, dst, kernels, biases, ReLU{});
            break;
        case Image::Float32:
            conv3x3_cin1_avx<float, 8>(src, dst, kernels, biases, ReLU{});
            break;
        }
    }
    void conv3x3_8to8_relu_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_float<ConvImplAVX<true>, 8, 8>(src, dst, kernels, biases, ReLU{});
        else
#   endif
            conv3x3_float<ConvImplAVX<false>, 8, 8>(src, dst, kernels, biases, ReLU{});
    }
    void deconv2x2_8to1_avx(const Image& src, Image& dst, const float* kernels)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            deconv2x2_float_avx<std::uint8_t, 8, 1>(src, dst, kernels);
            break;
        case Image::UInt16:
            deconv2x2_float_avx<std::uint16_t, 8, 1>(src, dst, kernels);
            break;
        case Image::Float32:
            deconv2x2_float_avx<float, 8, 1>(src, dst, kernels);
            break;
        }
    }

    void conv3x3_1to8_prelu_avx(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1_avx<std::uint8_t, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        case Image::UInt16:
            conv3x3_cin1_avx<std::uint16_t, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        case Image::Float32:
            conv3x3_cin1_avx<float, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        }
    }

    void conv3x3_1to8_identity_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1_avx<std::uint8_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1_avx<std::uint16_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1_avx<float, 8>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_8to8_prelu_avx(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_float<ConvImplAVX<true>, 8, 8>(src, dst, kernels, biases, PReLU{ alphas });
        else
#   endif
            conv3x3_float<ConvImplAVX<false>, 8, 8>(src, dst, kernels, biases, PReLU{ alphas });
    }
    void conv3x3_8to8_identity_residual_avx(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_float<ConvImplAVX<true>, 8, 8>(src, dst, kernels, biases, Identity{}, ResidualArg{ id, scale });
        else
#   endif
            conv3x3_float<ConvImplAVX<false>, 8, 8>(src, dst, kernels, biases, Identity{}, ResidualArg{ id, scale });
    }
    void conv3x3_8to8_identity_residual_conv1x1_8to8_prelu_add_avx(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1,
        const Image& id, const float scale,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_conv1x1_float<ConvImplAVX<true>, 8, 8, 8, false, false>(
                src, dst,
                kernels1, biases1, Identity{}, ResidualArg{ id, scale },
                kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
            );
        else
#   endif
            conv3x3_conv1x1_float<ConvImplAVX<false>, 8, 8, 8, false, false>(
                src, dst,
                kernels1, biases1, Identity{}, ResidualArg{ id, scale },
                kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
            );
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_add_avx(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
        {
            switch (dst.type())
            {
            case Image::UInt8:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<true>, std::uint8_t, 8, 2>(src, dst, kernels, biases, ResidualArg{ id, 1.0f });
                break;
            case Image::UInt16:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<true>, std::uint16_t, 8, 2>(src, dst, kernels, biases, ResidualArg{ id, 1.0f });
                break;
            case Image::Float32:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<true>, float, 8, 2>(src, dst, kernels, biases, ResidualArg{ id, 1.0f });
                break;
            }
        }
        else
#   endif
        {
            switch (dst.type())
            {
            case Image::UInt8:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<false>, std::uint8_t, 8, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::UInt16:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<false>, std::uint16_t, 8, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::Float32:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<false>, float, 8, 2>(src, dst, kernels, biases, nullptr);
                break;
            }
        }
    }

    void conv3x3_1to16_identity_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1_avx<std::uint8_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1_avx<std::uint16_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1_avx<float, 16>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_16to16_relu_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_float<ConvImplAVX<true>, 16, 16>(src, dst, kernels, biases, ReLU{});
        else
#   endif
            conv3x3_float<ConvImplAVX<false>, 16, 16>(src, dst, kernels, biases, ReLU{});
    }
    void conv3x3_16to16_identity_add_avx(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_float<ConvImplAVX<true>, 16, 16>(src, dst, kernels, biases, Identity{}, ResidualArg{ feat, 1.0f });
        else
#   endif
            conv3x3_float<ConvImplAVX<false>, 16, 16>(src, dst, kernels, biases, Identity{}, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_16to4_identity_pixelshuffle_4to1_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
        {
            switch (dst.type())
            {
            case Image::UInt8:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<true>, std::uint8_t, 16, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::UInt16:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<true>, std::uint16_t, 16, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::Float32:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<true>, float, 16, 2>(src, dst, kernels, biases, nullptr);
                break;
            }
        }
        else
#   endif
        {
            switch (dst.type())
            {
            case Image::UInt8:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<false>, std::uint8_t, 16, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::UInt16:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<false>, std::uint16_t, 16, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::Float32:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<false>, float, 16, 2>(src, dst, kernels, biases, nullptr);
                break;
            }
        }
    }

    void conv3x3_1to32_identity_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1_avx<std::uint8_t, 32>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1_avx<std::uint16_t, 32>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1_avx<float, 32>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_32to32_relu_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_float<ConvImplAVX<true>, 32, 32>(src, dst, kernels, biases, ReLU{});
        else
#   endif
            conv3x3_float<ConvImplAVX<false>, 32, 32>(src, dst, kernels, biases, ReLU{});
    }
    void conv3x3_32to32_identity_add_avx(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_float<ConvImplAVX<true>, 32, 32>(src, dst, kernels, biases, Identity{}, ResidualArg{ feat, 1.0f });
        else
#   endif
            conv3x3_float<ConvImplAVX<false>, 32, 32>(src, dst, kernels, biases, Identity{}, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_32to4_identity_pixelshuffle_4to1_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
        {
            switch (dst.type())
            {
            case Image::UInt8:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<true>, std::uint8_t, 32, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::UInt16:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<true>, std::uint16_t, 32, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::Float32:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<true>, float, 32, 2>(src, dst, kernels, biases, nullptr);
                break;
            }
        }
        else
#   endif
        {
            switch (dst.type())
            {
            case Image::UInt8:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<false>, std::uint8_t, 32, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::UInt16:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<false>, std::uint16_t, 32, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::Float32:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<false>, float, 32, 2>(src, dst, kernels, biases, nullptr);
                break;
            }
        }
    }

    void conv5x5_1to8_identity_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
        {
            switch (src.type())
            {
            case Image::UInt8:
                conv5x5_cin1_avx<std::uint8_t, 8, true>(src, dst, kernels, biases, Identity{});
                break;
            case Image::UInt16:
                conv5x5_cin1_avx<std::uint16_t, 8, true>(src, dst, kernels, biases, Identity{});
                break;
            case Image::Float32:
                conv5x5_cin1_avx<float, 8, true>(src, dst, kernels, biases, Identity{});
                break;
            }
        }
        else
#   endif
        {
            switch (src.type())
            {
            case Image::UInt8:
                conv5x5_cin1_avx<std::uint8_t, 8, false>(src, dst, kernels, biases, Identity{});
                break;
            case Image::UInt16:
                conv5x5_cin1_avx<std::uint16_t, 8, false>(src, dst, kernels, biases, Identity{});
                break;
            case Image::Float32:
                conv5x5_cin1_avx<float, 8, false>(src, dst, kernels, biases, Identity{});
                break;
            }
        }
    }
    void conv3x3_8to8_prelu_conv1x1_8to8_add_prelu_avx(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_conv1x1_float<ConvImplAVX<true>, 8, 8, 8, false, true>(
                src, dst,
                kernels1, biases1, PReLU{ alphas1 }, nullptr,
                kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
            );
        else
#   endif
            conv3x3_conv1x1_float<ConvImplAVX<false>, 8, 8, 8, false, true>(
                src, dst,
                kernels1, biases1, PReLU{ alphas1 }, nullptr,
                kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
            );
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
        {
            switch (dst.type())
            {
            case Image::UInt8:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<true>, std::uint8_t, 8, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::UInt16:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<true>, std::uint16_t, 8, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::Float32:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<true>, float, 8, 2>(src, dst, kernels, biases, nullptr);
                break;
            }
        }
        else
#   endif
        {
            switch (dst.type())
            {
            case Image::UInt8:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<false>, std::uint8_t, 8, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::UInt16:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<false>, std::uint16_t, 8, 2>(src, dst, kernels, biases, nullptr);
                break;
            case Image::Float32:
                conv3x3_identity_pixelshuffle_float<ConvImplAVX<false>, float, 8, 2>(src, dst, kernels, biases, nullptr);
                break;
            }
        }
    }

    void conv5x5_1to16_identity_avx(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
        {
            switch (src.type())
            {
            case Image::UInt8:
                conv5x5_cin1_avx<std::uint8_t, 16, true>(src, dst, kernels, biases, Identity{});
                break;
            case Image::UInt16:
                conv5x5_cin1_avx<std::uint16_t, 16, true>(src, dst, kernels, biases, Identity{});
                break;
            case Image::Float32:
                conv5x5_cin1_avx<float, 16, true>(src, dst, kernels, biases, Identity{});
                break;
            }
        }
        else
#   endif
        {
            switch (src.type())
            {
            case Image::UInt8:
                conv5x5_cin1_avx<std::uint8_t, 16, false>(src, dst, kernels, biases, Identity{});
                break;
            case Image::UInt16:
                conv5x5_cin1_avx<std::uint16_t, 16, false>(src, dst, kernels, biases, Identity{});
                break;
            case Image::Float32:
                conv5x5_cin1_avx<float, 16, false>(src, dst, kernels, biases, Identity{});
                break;
            }
        }
    }
    void conv3x3_16to16_prelu_avx(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_float<ConvImplAVX<true>, 16, 16>(src, dst, kernels, biases, PReLU{ alphas });
        else
#   endif
            conv3x3_float<ConvImplAVX<false>, 16, 16>(src, dst, kernels, biases, PReLU{ alphas });
    }
    void conv3x3_16to16_prelu_conv1x1_16to16_add_prelu_avx(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
#   ifdef AC_CORE_WITH_FMA
        if (simd::supportFMA())
            conv3x3_conv1x1_float<ConvImplAVX<true>, 16, 16, 16, false, true>(
                src, dst,
                kernels1, biases1, PReLU{ alphas1 }, nullptr,
                kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
            );
        else
#   endif
            conv3x3_conv1x1_float<ConvImplAVX<false>, 16, 16, 16, false, true>(
                src, dst,
                kernels1, biases1, PReLU{ alphas1 }, nullptr,
                kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
            );
    }
}
