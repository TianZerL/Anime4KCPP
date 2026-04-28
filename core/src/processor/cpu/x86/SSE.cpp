#include <xmmintrin.h>

#include "AC/Core/Image.hpp"
#include "AC/Core/Util.hpp"

#include "AC/Core/Internal/Processor/CPU/Common.hpp"

namespace ac::core::cpu
{
    static inline float sse_hsum_ps(const __m128& v) noexcept
    {
        __m128 v64 = _mm_add_ps(v, _mm_movehl_ps(v, v));
        __m128 v32 = _mm_add_ss(v64, _mm_shuffle_ps(v64, v64, _MM_SHUFFLE(3, 3, 1, 1)));
        return _mm_cvtss_f32(v32);
    }

    template <typename IN, int cout, typename ActiveFunc>
    inline void conv3x3_cin1_sse(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc)
    {
        util::parallelFor(0, src.height(), [&](const int i) {
            auto tp = i > 0 ? 1 : 0;
            auto bp = i < src.height() - 1 ? 1 : 0;

            for (int j = 0; j < src.width(); j++)
            {
                auto out = static_cast<float*>(dst.ptr(j, i));

                auto lp = j > 0 ? 1 : 0;
                auto rp = j < src.width() - 1 ? 1 : 0;

                __m128 r0 = _mm_set_ps(
                    toFloat(*static_cast<const IN*>(src.ptr(j - lp, i     ))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + rp, i - tp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j     , i - tp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j - lp, i - tp))));
                __m128 r4 = _mm_set_ps(
                    toFloat(*static_cast<const IN*>(src.ptr(j     , i + bp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j - lp, i + bp))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + rp, i     ))),
                    toFloat(*static_cast<const IN*>(src.ptr(j     , i     ))));
                auto r8 = toFloat(*static_cast<const IN*>(src.ptr(j + rp, i + bp)));

                for (int n = 0; n < cout; n++)
                {
                    __m128 k0 = _mm_loadu_ps(kernels + n * 9 + 0);
                    __m128 k4 = _mm_loadu_ps(kernels + n * 9 + 4);
                    auto sum = sse_hsum_ps(_mm_add_ps(_mm_mul_ps(r0, k0), _mm_mul_ps(r4, k4)));
                    auto k8 = *(kernels + n * 9 + 8);
                    out[n] = activeFunc(sum + r8 * k8 + biases[n], n);
                }
            }
        });
    }
    template <typename IN, int cout, typename ActiveFunc>
    inline void conv5x5_cin1_sse(const Image& src, Image& dst, const float* const kernels, const float* const biases, ActiveFunc&& activeFunc)
    {
        util::parallelFor(0, src.height(), [&](const int i) {
            int ioffsets[5] = { i > 1 ? -2 : (i > 0 ? -1 : 0) , i > 0 ? -1 : 0 , 0, i < src.height() - 1 ? 1 : 0, i < src.height() - 2 ? 2 : (i < src.height() - 1 ? 1 : 0) };

            for (int j = 0; j < src.width(); j++)
            {
                auto out = static_cast<float*>(dst.ptr(j, i));

                int joffsets[5] = { j > 1 ? -2 : (j > 0 ? -1 : 0), j > 0 ? -1 : 0, 0, j < src.width() - 1 ? 1 : 0 ,j < src.width() - 2 ? 2 : (j < src.width() - 1 ? 1 : 0) };

                __m128 r0 = _mm_set_ps(
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[3], i + ioffsets[0]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[2], i + ioffsets[0]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[1], i + ioffsets[0]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[0], i + ioffsets[0]))));
                __m128 r4 = _mm_set_ps(
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[2], i + ioffsets[1]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[1], i + ioffsets[1]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[0], i + ioffsets[1]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[4], i + ioffsets[0]))));
                __m128 r8 = _mm_set_ps(
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[1], i + ioffsets[2]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[0], i + ioffsets[2]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[4], i + ioffsets[1]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[3], i + ioffsets[1]))));
                __m128 r12 = _mm_set_ps(
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[0], i + ioffsets[3]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[4], i + ioffsets[2]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[3], i + ioffsets[2]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[2], i + ioffsets[2]))));
                __m128 r16 = _mm_set_ps(
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[4], i + ioffsets[3]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[3], i + ioffsets[3]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[2], i + ioffsets[3]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[1], i + ioffsets[3]))));
                __m128 r20 = _mm_set_ps(
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[3], i + ioffsets[4]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[2], i + ioffsets[4]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[1], i + ioffsets[4]))),
                    toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[0], i + ioffsets[4]))));
                auto r24 = toFloat(*static_cast<const IN*>(src.ptr(j + joffsets[4], i + ioffsets[4])));

                for (int n = 0; n < cout; n++)
                {
                    __m128 k0 = _mm_loadu_ps(kernels + n * 25 + 0);
                    __m128 k4 = _mm_loadu_ps(kernels + n * 25 + 4);
                    __m128 k8 = _mm_loadu_ps(kernels + n * 25 + 8);
                    __m128 k12 = _mm_loadu_ps(kernels + n * 25 + 12);
                    __m128 k16 = _mm_loadu_ps(kernels + n * 25 + 16);
                    __m128 k20 = _mm_loadu_ps(kernels + n * 25 + 20);

                    __m128 s0 = _mm_setzero_ps();
                    __m128 s1 = _mm_setzero_ps();
                    __m128 s2 = _mm_setzero_ps();

                    s0 = _mm_add_ps(_mm_mul_ps(r0, k0), s0);
                    s1 = _mm_add_ps(_mm_mul_ps(r4, k4), s1);
                    s2 = _mm_add_ps(_mm_mul_ps(r8, k8), s2);
                    s0 = _mm_add_ps(_mm_mul_ps(r12, k12), s0);
                    s1 = _mm_add_ps(_mm_mul_ps(r16, k16), s1);
                    s2 = _mm_add_ps(_mm_mul_ps(r20, k20), s2);

                    auto sum = sse_hsum_ps(_mm_add_ps(s0, _mm_add_ps(s1, s2)));

                    auto k24 = *(kernels + n * 25 + 24);
                    out[n] = activeFunc(sum + r24 * k24 + biases[n], n);
                }
            }
        });
    }
    template <typename OUT, int cin, int cout>
    inline void deconv2x2_float_sse(const Image& src, Image& dst, const float* const kernels)
    {
        filter([=](const int i, const int j, const void* const sptr, void* const dptr) {
            auto in = static_cast<const float*>(sptr);
            auto out = static_cast<OUT*>(dptr);

            auto index = ((i & 1) << 1) + (j & 1);

            constexpr int vstep = 4;
            constexpr int count = cin / vstep;
            constexpr int remain = cin % vstep;

            __m128 r[count + (remain ? 1 : 0)]{};
            for (int idx = 0; idx < count; idx++) r[idx] = _mm_loadu_ps(in + idx * vstep);
            if constexpr (remain) r[count] = _mm_set_ps(0.0f, remain > 2 ? (in + count * vstep)[2] : 0.0f, remain > 1 ? (in + count * vstep)[1] : 0.0f, (in + count * vstep)[0]);

            for (int n = 0; n < cout; n++)
            {
                auto kptr = kernels + n * cin * 4 + cin * index;
                float sum = 0.0f;
                __m128 k[count + (remain ? 1 : 0)]{};
                for (int idx = 0; idx < count; idx++)
                {
                    k[idx] = _mm_loadu_ps(kptr + idx * vstep);
                    sum += sse_hsum_ps(_mm_mul_ps(r[idx], k[idx]));
                }
                if constexpr (remain)
                {
                    k[count] = _mm_set_ps(0.0f, remain > 2 ? (kptr + count * vstep)[2] : 0.0f, remain > 1 ? (kptr + count * vstep)[1] : 0.0f, (kptr + count * vstep)[0]);
                    sum += sse_hsum_ps(_mm_mul_ps(r[count], k[count]));
                }
                out[n] = fromFloat<OUT>(sum);
            }
        }, src, dst);
    }

    struct ConvImplSSE
    {
        template <int cin, int cout>
        static void conv1x1(const float* rptr[], float* const out, const float* const kernels, const float* const biases) noexcept
        {
            constexpr int vstep = 4;
            constexpr int count = cin / vstep;
            constexpr int remain = cin % vstep;

            std::memcpy(out, biases, sizeof(float) * cout);

            for (int idx = 0; idx < count; idx++)
            {
                __m128 r = _mm_loadu_ps(rptr[0] + idx * vstep);

                for (int n = 0; n < cout; n++)
                {
                    auto kptr = kernels + n * cin;

                    __m128 k = _mm_loadu_ps(kptr + idx * vstep);

                    out[n] += sse_hsum_ps(_mm_mul_ps(r, k));
                }
            }
            if constexpr (remain)
            {
                __m128 r = _mm_set_ps(0.0f, remain > 2 ? (rptr[0] + count * vstep)[2] : 0.0f, remain > 1 ? (rptr[0] + count * vstep)[1] : 0.0f, (rptr[0] + count * vstep)[0]);

                for (int n = 0; n < cout; n++)
                {
                    auto kptr = kernels + n * cin;

                    __m128 k = _mm_set_ps(0.0f, remain > 2 ? (kptr + count * vstep)[2] : 0.0f, remain > 1 ? (kptr + count * vstep)[1] : 0.0f, (kptr + count * vstep)[0]);

                    out[n] += sse_hsum_ps(_mm_mul_ps(r, k));
                }
            }
        }

        template <int cin, int cout>
        static void conv3x3(const float* rptr[], float* const out, const float* const kernels, const float* const biases) noexcept
        {
            constexpr int vstep = 4;
            constexpr int count = cin / vstep;
            constexpr int remain = cin % vstep;

            std::memcpy(out, biases, sizeof(float) * cout);

            for (int idx = 0; idx < count; idx++)
            {
                __m128 r0 = _mm_loadu_ps(rptr[0] + idx * vstep);
                __m128 r1 = _mm_loadu_ps(rptr[1] + idx * vstep);
                __m128 r2 = _mm_loadu_ps(rptr[2] + idx * vstep);
                __m128 r3 = _mm_loadu_ps(rptr[3] + idx * vstep);
                __m128 r4 = _mm_loadu_ps(rptr[4] + idx * vstep);
                __m128 r5 = _mm_loadu_ps(rptr[5] + idx * vstep);
                __m128 r6 = _mm_loadu_ps(rptr[6] + idx * vstep);
                __m128 r7 = _mm_loadu_ps(rptr[7] + idx * vstep);
                __m128 r8 = _mm_loadu_ps(rptr[8] + idx * vstep);

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

                    __m128 s0 = _mm_setzero_ps();
                    __m128 s1 = _mm_setzero_ps();
                    __m128 s2 = _mm_setzero_ps();

                    __m128 k0 = _mm_loadu_ps(kptr[0] + idx * vstep);
                    __m128 k1 = _mm_loadu_ps(kptr[1] + idx * vstep);
                    __m128 k2 = _mm_loadu_ps(kptr[2] + idx * vstep);
                    __m128 k3 = _mm_loadu_ps(kptr[3] + idx * vstep);
                    __m128 k4 = _mm_loadu_ps(kptr[4] + idx * vstep);
                    __m128 k5 = _mm_loadu_ps(kptr[5] + idx * vstep);
                    __m128 k6 = _mm_loadu_ps(kptr[6] + idx * vstep);
                    __m128 k7 = _mm_loadu_ps(kptr[7] + idx * vstep);
                    __m128 k8 = _mm_loadu_ps(kptr[8] + idx * vstep);

                    s0 = _mm_add_ps(_mm_mul_ps(r0, k0), s0);
                    s1 = _mm_add_ps(_mm_mul_ps(r1, k1), s1);
                    s2 = _mm_add_ps(_mm_mul_ps(r2, k2), s2);
                    s0 = _mm_add_ps(_mm_mul_ps(r3, k3), s0);
                    s1 = _mm_add_ps(_mm_mul_ps(r4, k4), s1);
                    s2 = _mm_add_ps(_mm_mul_ps(r5, k5), s2);
                    s0 = _mm_add_ps(_mm_mul_ps(r6, k6), s0);
                    s1 = _mm_add_ps(_mm_mul_ps(r7, k7), s1);
                    s2 = _mm_add_ps(_mm_mul_ps(r8, k8), s2);

                    out[n] += sse_hsum_ps(_mm_add_ps(s0, _mm_add_ps(s1, s2)));
                }
            }
            if constexpr (remain)
            {
                __m128 r0 = _mm_set_ps(0.0f, remain > 2 ? (rptr[0] + count * vstep)[2] : 0.0f, remain > 1 ? (rptr[0] + count * vstep)[1] : 0.0f, (rptr[0] + count * vstep)[0]);
                __m128 r1 = _mm_set_ps(0.0f, remain > 2 ? (rptr[1] + count * vstep)[2] : 0.0f, remain > 1 ? (rptr[1] + count * vstep)[1] : 0.0f, (rptr[1] + count * vstep)[0]);
                __m128 r2 = _mm_set_ps(0.0f, remain > 2 ? (rptr[2] + count * vstep)[2] : 0.0f, remain > 1 ? (rptr[2] + count * vstep)[1] : 0.0f, (rptr[2] + count * vstep)[0]);
                __m128 r3 = _mm_set_ps(0.0f, remain > 2 ? (rptr[3] + count * vstep)[2] : 0.0f, remain > 1 ? (rptr[3] + count * vstep)[1] : 0.0f, (rptr[3] + count * vstep)[0]);
                __m128 r4 = _mm_set_ps(0.0f, remain > 2 ? (rptr[4] + count * vstep)[2] : 0.0f, remain > 1 ? (rptr[4] + count * vstep)[1] : 0.0f, (rptr[4] + count * vstep)[0]);
                __m128 r5 = _mm_set_ps(0.0f, remain > 2 ? (rptr[5] + count * vstep)[2] : 0.0f, remain > 1 ? (rptr[5] + count * vstep)[1] : 0.0f, (rptr[5] + count * vstep)[0]);
                __m128 r6 = _mm_set_ps(0.0f, remain > 2 ? (rptr[6] + count * vstep)[2] : 0.0f, remain > 1 ? (rptr[6] + count * vstep)[1] : 0.0f, (rptr[6] + count * vstep)[0]);
                __m128 r7 = _mm_set_ps(0.0f, remain > 2 ? (rptr[7] + count * vstep)[2] : 0.0f, remain > 1 ? (rptr[7] + count * vstep)[1] : 0.0f, (rptr[7] + count * vstep)[0]);
                __m128 r8 = _mm_set_ps(0.0f, remain > 2 ? (rptr[8] + count * vstep)[2] : 0.0f, remain > 1 ? (rptr[8] + count * vstep)[1] : 0.0f, (rptr[8] + count * vstep)[0]);


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

                    __m128 s0 = _mm_setzero_ps();
                    __m128 s1 = _mm_setzero_ps();
                    __m128 s2 = _mm_setzero_ps();

                    __m128 k0 = _mm_set_ps(0.0f, remain > 2 ? (kptr[0] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[0] + count * vstep)[1] : 0.0f, (kptr[0] + count * vstep)[0]);
                    __m128 k1 = _mm_set_ps(0.0f, remain > 2 ? (kptr[1] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[1] + count * vstep)[1] : 0.0f, (kptr[1] + count * vstep)[0]);
                    __m128 k2 = _mm_set_ps(0.0f, remain > 2 ? (kptr[2] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[2] + count * vstep)[1] : 0.0f, (kptr[2] + count * vstep)[0]);
                    __m128 k3 = _mm_set_ps(0.0f, remain > 2 ? (kptr[3] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[3] + count * vstep)[1] : 0.0f, (kptr[3] + count * vstep)[0]);
                    __m128 k4 = _mm_set_ps(0.0f, remain > 2 ? (kptr[4] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[4] + count * vstep)[1] : 0.0f, (kptr[4] + count * vstep)[0]);
                    __m128 k5 = _mm_set_ps(0.0f, remain > 2 ? (kptr[5] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[5] + count * vstep)[1] : 0.0f, (kptr[5] + count * vstep)[0]);
                    __m128 k6 = _mm_set_ps(0.0f, remain > 2 ? (kptr[6] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[6] + count * vstep)[1] : 0.0f, (kptr[6] + count * vstep)[0]);
                    __m128 k7 = _mm_set_ps(0.0f, remain > 2 ? (kptr[7] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[7] + count * vstep)[1] : 0.0f, (kptr[7] + count * vstep)[0]);
                    __m128 k8 = _mm_set_ps(0.0f, remain > 2 ? (kptr[8] + count * vstep)[2] : 0.0f, remain > 1 ? (kptr[8] + count * vstep)[1] : 0.0f, (kptr[8] + count * vstep)[0]);

                    s0 = _mm_add_ps(_mm_mul_ps(r0, k0), s0);
                    s1 = _mm_add_ps(_mm_mul_ps(r1, k1), s1);
                    s2 = _mm_add_ps(_mm_mul_ps(r2, k2), s2);
                    s0 = _mm_add_ps(_mm_mul_ps(r3, k3), s0);
                    s1 = _mm_add_ps(_mm_mul_ps(r4, k4), s1);
                    s2 = _mm_add_ps(_mm_mul_ps(r5, k5), s2);
                    s0 = _mm_add_ps(_mm_mul_ps(r6, k6), s0);
                    s1 = _mm_add_ps(_mm_mul_ps(r7, k7), s1);
                    s2 = _mm_add_ps(_mm_mul_ps(r8, k8), s2);

                    out[n] += sse_hsum_ps(_mm_add_ps(s0, _mm_add_ps(s1, s2)));
                }
            }
        }
    };

    void conv3x3_1to8_relu_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1_sse<std::uint8_t, 8>(src, dst, kernels, biases, ReLU{});
            break;
        case Image::UInt16:
            conv3x3_cin1_sse<std::uint16_t, 8>(src, dst, kernels, biases, ReLU{});
            break;
        case Image::Float32:
            conv3x3_cin1_sse<float, 8>(src, dst, kernels, biases, ReLU{});
            break;
        }
    }
    void conv3x3_8to8_relu_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_float<ConvImplSSE, 8, 8>(src, dst, kernels, biases, ReLU{});
    }
    void deconv2x2_8to1_sse(const Image& src, Image& dst, const float* kernels)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            deconv2x2_float_sse<std::uint8_t, 8, 1>(src, dst, kernels);
            break;
        case Image::UInt16:
            deconv2x2_float_sse<std::uint16_t, 8, 1>(src, dst, kernels);
            break;
        case Image::Float32:
            deconv2x2_float_sse<float, 8, 1>(src, dst, kernels);
            break;
        }
    }

    void conv3x3_1to8_identity_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1_sse<std::uint8_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1_sse<std::uint16_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1_sse<float, 8>(src, dst, kernels, biases, Identity{});
            break;
        }
    }

    void conv3x3_1to8_prelu_sse(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1_sse<std::uint8_t, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        case Image::UInt16:
            conv3x3_cin1_sse<std::uint16_t, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        case Image::Float32:
            conv3x3_cin1_sse<float, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        }
    }

    void conv3x3_8to8_prelu_sse(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        conv3x3_float<ConvImplSSE, 8, 8>(src, dst, kernels, biases, PReLU{ alphas });
    }
    void conv3x3_8to8_identity_residual_sse(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale)
    {
        conv3x3_float<ConvImplSSE, 8, 8>(src, dst, kernels, biases, Identity{}, ResidualArg{ id, scale });
    }
    void conv3x3_8to8_identity_residual_conv1x1_8to8_prelu_add_sse(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1,
        const Image& id, const float scale,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_float<ConvImplSSE, 8, 8, 8, false, false>(
            src, dst,
            kernels1, biases1, Identity{}, ResidualArg{ id, scale },
            kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
        );
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_add_sse(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<ConvImplSSE, std::uint8_t, 8, 2>(src, dst, kernels, biases, ResidualArg{ id, 1.0f });
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<ConvImplSSE, std::uint16_t, 8, 2>(src, dst, kernels, biases, ResidualArg{ id, 1.0f });
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<ConvImplSSE, float, 8, 2>(src, dst, kernels, biases, ResidualArg{ id, 1.0f });
            break;
        }
    }

    void conv3x3_1to16_identity_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1_sse<std::uint8_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1_sse<std::uint16_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1_sse<float, 16>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_16to16_relu_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_float<ConvImplSSE, 16, 16>(src, dst, kernels, biases, ReLU{});
    }
    void conv3x3_16to16_identity_add_sse(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
        conv3x3_float<ConvImplSSE, 16, 16>(src, dst, kernels, biases, Identity{}, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_16to4_identity_pixelshuffle_4to1_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<ConvImplSSE, std::uint8_t, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<ConvImplSSE, std::uint16_t, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<ConvImplSSE, float, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        }
    }

    void conv3x3_1to32_identity_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1_sse<std::uint8_t, 32>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1_sse<std::uint16_t, 32>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1_sse<float, 32>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_32to32_relu_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_float<ConvImplSSE, 32, 32>(src, dst, kernels, biases, ReLU{});
    }
    void conv3x3_32to32_identity_add_sse(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
        conv3x3_float<ConvImplSSE, 32, 32>(src, dst, kernels, biases, Identity{}, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_32to4_identity_pixelshuffle_4to1_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<ConvImplSSE, std::uint8_t, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<ConvImplSSE, std::uint16_t, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<ConvImplSSE, float, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        }
    }

    void conv5x5_1to8_identity_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv5x5_cin1_sse<std::uint8_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv5x5_cin1_sse<std::uint16_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv5x5_cin1_sse<float, 8>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_8to8_prelu_conv1x1_8to8_add_prelu_sse(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_float<ConvImplSSE, 8, 8, 8, false, true>(
            src, dst,
            kernels1, biases1, PReLU{ alphas1 }, nullptr,
            kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
        );
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<ConvImplSSE, std::uint8_t, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<ConvImplSSE, std::uint16_t, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<ConvImplSSE, float, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        }
    }

    void conv5x5_1to16_identity_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv5x5_cin1_sse<std::uint8_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv5x5_cin1_sse<std::uint16_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv5x5_cin1_sse<float, 16>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_16to16_prelu_sse(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        conv3x3_float<ConvImplSSE, 16, 16>(src, dst, kernels, biases, PReLU{ alphas });
    }
    void conv3x3_16to16_prelu_conv1x1_16to16_add_prelu_sse(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_float<ConvImplSSE, 16, 16, 16, false, true>(
            src, dst,
            kernels1, biases1, PReLU{ alphas1 }, nullptr,
            kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
        );
    }
}
