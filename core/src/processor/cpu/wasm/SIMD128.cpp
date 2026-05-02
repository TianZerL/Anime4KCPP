#include <cstdint>

#include <wasm_simd128.h>

#include "AC/Core/Image.hpp"
#include "AC/Util/Macro.hpp"

#include "AC/Core/Internal/Processor/CPU/Common.hpp"

namespace ac::core::cpu
{
    struct OpImplWASM
    {
    private:
        static AC_FORCE_INLINE float hsum(const v128_t& v) noexcept
        {
            v128_t v64 = wasm_f32x4_add(v, wasm_i32x4_shuffle(v, v, 2, 3, 0, 0));
            v128_t v32 = wasm_f32x4_add(v64, wasm_i32x4_shuffle(v64, v64, 1, 0, 0, 0));
            return wasm_f32x4_extract_lane(v32, 0);
        }

        template <int cin, int scount, int cpos>
        static AC_FORCE_INLINE void conv_kernel(const int sgroupIdx, const float** const rptr, v128_t* const s, float* const out, const float* const kernels) noexcept
        {
            constexpr int vstep = 4;
            constexpr int count = cin / vstep;
            constexpr int remain = cin % vstep;

            for (int n = 0; n < scount; n++) s[n] = wasm_f32x4_splat(0.0f);
            for (int p = 0; p < cpos; p++)
            {
                for (int idx = 0; idx < count; idx++)
                {
                    v128_t r = wasm_v128_load(rptr[p] + idx * vstep);
                    for (int n = 0; n < scount; n++)
                    {
                        v128_t k = wasm_v128_load(kernels + (sgroupIdx * scount + n) * cin * cpos + cin * p + idx * vstep);
                        s[n] = wasm_f32x4_add(wasm_f32x4_mul(r, k), s[n]);
                    }
                }
                if constexpr (remain)
                    for (int c = count * vstep; c < cin; c++)
                        for (int n = 0; n < scount; n++)
                            out[sgroupIdx * scount + n] += rptr[p][c] * kernels[(sgroupIdx * scount + n) * cin * cpos + cin * p + c];
            }
            for (int n = 0; n < scount; n++) out[sgroupIdx * scount + n] += hsum(s[n]);
        }

    public:
        template <int vsize>
        static float dot(const float* const v1, const float* const v2) noexcept
        {
            constexpr int vstep = 4;
            constexpr int count = vsize / vstep;
            constexpr int remain = vsize % vstep;

            float sum = 0.0f;

            v128_t s = wasm_f32x4_splat(0.0f);
            for (int idx = 0; idx < count; idx++)
            {
                v128_t r1 = wasm_v128_load(v1 + idx * vstep);
                v128_t r2 = wasm_v128_load(v2 + idx * vstep);
                s = wasm_f32x4_add(wasm_f32x4_mul(r1, r2), s);
            }
            sum += hsum(s);

            if constexpr (remain)
                for (int i = count * vstep; i < vsize; i++)
                    sum += v1[i] * v2[i];

            return sum;
        }

        template <int cout, int cpos>
        static void conv_cin1(const float* const rptr, float* const out, const float* const kernels, const float* const biases) noexcept
        {
            constexpr int vstep = 4;
            constexpr int count = cpos / vstep;
            constexpr int remain = cpos % vstep;

            v128_t r[count];
            for (int idx = 0; idx < count; idx++) r[idx] = wasm_v128_load(rptr + idx * vstep);

            for (int n = 0; n < cout; n++)
            {
                v128_t s = wasm_f32x4_splat(0.0f);
                auto kptr = kernels + n * cpos;
                for (int idx = 0; idx < count; idx++)
                {
                    v128_t k = wasm_v128_load(kptr + idx * vstep);
                    s = wasm_f32x4_add(wasm_f32x4_mul(r[idx], k), s);
                }
                auto sum = hsum(s);

                for (int i = 0; i < remain; i++) sum += rptr[count * vstep + i] * kptr[count * vstep + i];

                out[n] = sum + biases[n];
            }
        }

        template <int cin, int cout, int cpos>
        static void conv(const float** const rptr, float* const out, const float* const kernels, const float* const biases) noexcept
        {
            constexpr int scount = 8;
            constexpr int sgroup = cout / scount;
            constexpr int sremian = cout % scount;

            std::memcpy(out, biases, sizeof(float) * cout);

            v128_t s[sgroup > 0 ? scount : sremian];

            if constexpr (sgroup)
                for (int i = 0; i < sgroup; i++)
                    conv_kernel<cin, scount, cpos>(i, rptr, s, out, kernels);
            if constexpr (sremian)
                conv_kernel<cin, sremian, cpos>(sgroup, rptr, s, out, kernels);
        }  
    };

    void conv3x3_1to8_relu_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplWASM, std::uint8_t, 8>(src, dst, kernels, biases, ReLU{});
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplWASM, std::uint16_t, 8>(src, dst, kernels, biases, ReLU{});
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplWASM, float, 8>(src, dst, kernels, biases, ReLU{});
            break;
        }
    }
    void conv3x3_8to8_relu_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_float<OpImplWASM, 8, 8>(src, dst, kernels, biases, ReLU{});
    }
    void deconv2x2_8to1_wasm_simd128(const Image& src, Image& dst, const float* kernels)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            deconv2x2<OpImplWASM, std::uint8_t, 8, 1>(src, dst, kernels);
            break;
        case Image::UInt16:
            deconv2x2<OpImplWASM, std::uint16_t, 8, 1>(src, dst, kernels);
            break;
        case Image::Float32:
            deconv2x2<OpImplWASM, float, 8, 1>(src, dst, kernels);
            break;
        }
    }

    void conv3x3_1to8_prelu_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplWASM, std::uint8_t, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplWASM, std::uint16_t, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplWASM, float, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        }
    }

    void conv3x3_1to8_identity_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplWASM, std::uint8_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplWASM, std::uint16_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplWASM, float, 8>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_8to8_prelu_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        conv3x3_float<OpImplWASM, 8, 8>(src, dst, kernels, biases, PReLU{ alphas });
    }
    void conv3x3_8to8_identity_residual_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale)
    {
        conv3x3_float<OpImplWASM, 8, 8>(src, dst, kernels, biases, Identity{}, ResidualArg{ id, scale });
    }
    void conv3x3_8to8_identity_residual_conv1x1_8to8_prelu_add_wasm_simd128(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1,
        const Image& id, const float scale,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_float<OpImplWASM, 8, 8, 8, false, false>(
            src, dst,
            kernels1, biases1, Identity{}, ResidualArg{ id, scale },
            kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
        );
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_add_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<OpImplWASM, std::uint8_t, 8, 2>(src, dst, kernels, biases, ResidualArg{ id, 1.0f });
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<OpImplWASM, std::uint16_t, 8, 2>(src, dst, kernels, biases, ResidualArg{ id, 1.0f });
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<OpImplWASM, float, 8, 2>(src, dst, kernels, biases, ResidualArg{ id, 1.0f });
            break;
        }
    }

    void conv3x3_1to16_identity_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplWASM, std::uint8_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplWASM, std::uint16_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplWASM, float, 16>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_16to16_relu_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_float<OpImplWASM, 16, 16>(src, dst, kernels, biases, ReLU{});
    }
    void conv3x3_16to16_identity_add_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
        conv3x3_float<OpImplWASM, 16, 16>(src, dst, kernels, biases, Identity{}, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_16to4_identity_pixelshuffle_4to1_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<OpImplWASM, std::uint8_t, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<OpImplWASM, std::uint16_t, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<OpImplWASM, float, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        }
    }

    void conv3x3_1to32_identity_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplWASM, std::uint8_t, 32>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplWASM, std::uint16_t, 32>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplWASM, float, 32>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_32to32_relu_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_float<OpImplWASM, 32, 32>(src, dst, kernels, biases, ReLU{});
    }
    void conv3x3_32to32_identity_add_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
        conv3x3_float<OpImplWASM, 32, 32>(src, dst, kernels, biases, Identity{}, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_32to4_identity_pixelshuffle_4to1_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<OpImplWASM, std::uint8_t, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<OpImplWASM, std::uint16_t, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<OpImplWASM, float, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        }
    }

    void conv5x5_1to8_identity_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv5x5_cin1<OpImplWASM, std::uint8_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv5x5_cin1<OpImplWASM, std::uint16_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv5x5_cin1<OpImplWASM, float, 8>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_8to8_prelu_conv1x1_8to8_add_prelu_wasm_simd128(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_float<OpImplWASM, 8, 8, 8, false, true>(
            src, dst,
            kernels1, biases1, PReLU{ alphas1 }, nullptr,
            kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
        );
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<OpImplWASM, std::uint8_t, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<OpImplWASM, std::uint16_t, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<OpImplWASM, float, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        }
    }

    void conv5x5_1to16_identity_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv5x5_cin1<OpImplWASM, std::uint8_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv5x5_cin1<OpImplWASM, std::uint16_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv5x5_cin1<OpImplWASM, float, 16>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_16to16_prelu_wasm_simd128(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        conv3x3_float<OpImplWASM, 16, 16>(src, dst, kernels, biases, PReLU{ alphas });
    }
    void conv3x3_16to16_prelu_conv1x1_16to16_add_prelu_wasm_simd128(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_float<OpImplWASM, 16, 16, 16, false, true>(
            src, dst,
            kernels1, biases1, PReLU{ alphas1 }, nullptr,
            kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
        );
    }
}
