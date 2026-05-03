#include "AC/Core/Image.hpp"

#include "AC/Core/Internal/DataType.hpp"
#include "AC/Core/Internal/Processor/CPU/Common.hpp"

namespace ac::core::cpu
{
    struct OpImplGeneric
    {
        template <int vsize>
        static float dot(const float* const v1, const float* const v2) noexcept
        {
            float sum = 0.0f;

            for (int i = 0; i < vsize; i++) sum += v1[i] * v2[i];

            return sum;
        }

        template <int cout, int cpos>
        static void conv_cin1(const float* const rptr, float* const out, const float* const kernels, const float* const biases) noexcept
        {
            for (int n = 0; n < cout; n++)
            {
                auto kptr = kernels + n * cpos;
                out[n] = biases[n];
                for (int i = 0; i < cpos; i++) out[n] += rptr[i] * kptr[i];
            }
        }

        template <int cin, int cout, int cpos>
        static void conv(const float** const rptr, float* const out, const float* const kernels, const float* const biases) noexcept
        {
            std::memcpy(out, biases, sizeof(float) * cout);

            for (int p = 0; p < cpos; p++)
                for (int c = 0; c < cin; c++)
                    for (int n = 0; n < cout; n++)
                        out[n] += rptr[p][c] * kernels[n * cin * cpos + cin * p + c];
        }
    };

    void conv3x3_1to8_relu_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplGeneric, DataType::UInt8, 8>(src, dst, kernels, biases, ReLU{});
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplGeneric, DataType::UInt16, 8>(src, dst, kernels, biases, ReLU{});
            break;
        case Image::Float16:
            conv3x3_cin1<OpImplGeneric, DataType::Float16, 8>(src, dst, kernels, biases, ReLU{});
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplGeneric, DataType::Float32, 8>(src, dst, kernels, biases, ReLU{});
            break;
        }
    }
    void conv3x3_8to8_relu_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_float<OpImplGeneric, 8, 8>(src, dst, kernels, biases, ReLU{});
    }
    void deconv2x2_8to1_generic(const Image& src, Image& dst, const float* kernels)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            deconv2x2<OpImplGeneric, DataType::UInt8, 8, 1>(src, dst, kernels);
            break;
        case Image::UInt16:
            deconv2x2<OpImplGeneric, DataType::UInt16, 8, 1>(src, dst, kernels);
            break;
        case Image::Float16:
            deconv2x2<OpImplGeneric, DataType::Float16, 8, 1>(src, dst, kernels);
            break;
        case Image::Float32:
            deconv2x2<OpImplGeneric, DataType::Float32, 8, 1>(src, dst, kernels);
            break;
        }
    }

    void conv3x3_1to8_prelu_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplGeneric, DataType::UInt8, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplGeneric, DataType::UInt16, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        case Image::Float16:
            conv3x3_cin1<OpImplGeneric, DataType::Float16, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplGeneric, DataType::Float32, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        }
    }

    void conv3x3_1to8_identity_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplGeneric, DataType::UInt8, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplGeneric, DataType::UInt16, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float16:
            conv3x3_cin1<OpImplGeneric, DataType::Float16, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplGeneric, DataType::Float32, 8>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_8to8_prelu_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        conv3x3_float<OpImplGeneric, 8, 8>(src, dst, kernels, biases, PReLU{ alphas });
    }
    void conv3x3_8to8_identity_residual_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale)
    {
        conv3x3_float<OpImplGeneric, 8, 8>(src, dst, kernels, biases, Identity{}, ResidualArg{ id, scale });
    }
    void conv3x3_8to8_identity_residual_conv1x1_8to8_prelu_add_generic(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1,
        const Image& id, const float scale,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_float<OpImplGeneric, 8, 8, 8, false, false>(
            src, dst,
            kernels1, biases1, Identity{}, ResidualArg{ id, scale },
            kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
        );
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_add_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<OpImplGeneric, DataType::UInt8, 8, 2>(src, dst, kernels, biases, ResidualArg{id, 1.0f});
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<OpImplGeneric, DataType::UInt16, 8, 2>(src, dst, kernels, biases, ResidualArg{id, 1.0f});
            break;
        case Image::Float16:
            conv3x3_identity_pixelshuffle_float<OpImplGeneric, DataType::Float16, 8, 2>(src, dst, kernels, biases, ResidualArg{id, 1.0f});
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<OpImplGeneric, DataType::Float32, 8, 2>(src, dst, kernels, biases, ResidualArg{id, 1.0f});
            break;
        }
    }

    void conv3x3_1to16_identity_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplGeneric, DataType::UInt8, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplGeneric, DataType::UInt16, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float16:
            conv3x3_cin1<OpImplGeneric, DataType::Float16, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplGeneric, DataType::Float32, 16>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_16to16_relu_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_float<OpImplGeneric, 16, 16>(src, dst, kernels, biases, ReLU{});
    }
    void conv3x3_16to16_identity_add_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
        conv3x3_float<OpImplGeneric, 16, 16>(src, dst, kernels, biases, Identity{}, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_16to4_identity_pixelshuffle_4to1_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<OpImplGeneric, DataType::UInt8, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<OpImplGeneric, DataType::UInt16, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float16:
            conv3x3_identity_pixelshuffle_float<OpImplGeneric, DataType::Float16, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<OpImplGeneric, DataType::Float32, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        }
    }

    void conv3x3_1to32_identity_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplGeneric, DataType::UInt8, 32>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplGeneric, DataType::UInt16, 32>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float16:
            conv3x3_cin1<OpImplGeneric, DataType::Float16, 32>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplGeneric, DataType::Float32, 32>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_32to32_relu_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_float<OpImplGeneric, 32, 32>(src, dst, kernels, biases, ReLU{});
    }
    void conv3x3_32to32_identity_add_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
        conv3x3_float<OpImplGeneric, 32, 32>(src, dst, kernels, biases, Identity{}, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_32to4_identity_pixelshuffle_4to1_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<OpImplGeneric, DataType::UInt8, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<OpImplGeneric, DataType::UInt16, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float16:
            conv3x3_identity_pixelshuffle_float<OpImplGeneric, DataType::Float16, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<OpImplGeneric, DataType::Float32, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        }
    }

    void conv5x5_1to8_identity_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv5x5_cin1<OpImplGeneric, DataType::UInt8, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv5x5_cin1<OpImplGeneric, DataType::UInt16, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float16:
            conv5x5_cin1<OpImplGeneric, DataType::Float16, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv5x5_cin1<OpImplGeneric, DataType::Float32, 8>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_8to8_prelu_conv1x1_8to8_add_prelu_generic(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_float<OpImplGeneric, 8, 8, 8, false, true>(
            src, dst,
            kernels1, biases1, PReLU{ alphas1 }, nullptr,
            kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
        );
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<OpImplGeneric, DataType::UInt8, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<OpImplGeneric, DataType::UInt16, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float16:
            conv3x3_identity_pixelshuffle_float<OpImplGeneric, DataType::Float16, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<OpImplGeneric, DataType::Float32, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        }
    }

    void conv5x5_1to16_identity_generic(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv5x5_cin1<OpImplGeneric, DataType::UInt8, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv5x5_cin1<OpImplGeneric, DataType::UInt16, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float16:
            conv5x5_cin1<OpImplGeneric, DataType::Float16, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv5x5_cin1<OpImplGeneric, DataType::Float32, 16>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_16to16_prelu_generic(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        conv3x3_float<OpImplGeneric, 16, 16>(src, dst, kernels, biases, PReLU{ alphas });
    }
    void conv3x3_16to16_prelu_conv1x1_16to16_add_prelu_generic(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_float<OpImplGeneric, 16, 16, 16, false, true>(
            src, dst,
            kernels1, biases1, PReLU{ alphas1 }, nullptr,
            kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
        );
    }

    void pixelshuffle_4to1_generic(const Image& src, Image& dst)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            pixelshuffle<DataType::UInt8, 4, 2>(src, dst);
            break;
        case Image::UInt16:
            pixelshuffle<DataType::UInt16, 4, 2>(src, dst);
            break;
        case Image::Float16:
            pixelshuffle<DataType::Float16, 4, 2>(src, dst);
            break;
        case Image::Float32:
            pixelshuffle<DataType::Float32, 4, 2>(src, dst);
            break;
        }
    }
}
