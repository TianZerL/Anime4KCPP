#include "AC/Core/Image.hpp"

#include "AC/Core/Internal/DataType.hpp"
#include "AC/Core/Internal/Processor/CPU/Common.hpp"
#include "AC/Core/Internal/Processor/CPU/X86/SSE.hpp"

namespace ac::core::cpu
{
    void conv3x3_1to8_relu_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplSSE, DataType::UInt8, 8>(src, dst, kernels, biases, ReLU{});
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplSSE, DataType::UInt16, 8>(src, dst, kernels, biases, ReLU{});
            break;
        case Image::Float16:
            conv3x3_cin1<OpImplSSE, DataType::Float16, 8>(src, dst, kernels, biases, ReLU{});
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplSSE, DataType::Float32, 8>(src, dst, kernels, biases, ReLU{});
            break;
        }
    }
    void conv3x3_8to8_relu_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_float<OpImplSSE, 8, 8>(src, dst, kernels, biases, ReLU{});
    }
    void deconv2x2_8to1_sse(const Image& src, Image& dst, const float* kernels)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            deconv2x2<OpImplSSE, DataType::UInt8, 8, 1>(src, dst, kernels);
            break;
        case Image::UInt16:
            deconv2x2<OpImplSSE, DataType::UInt16, 8, 1>(src, dst, kernels);
            break;
        case Image::Float16:
            deconv2x2<OpImplSSE, DataType::Float16, 8, 1>(src, dst, kernels);
            break;
        case Image::Float32:
            deconv2x2<OpImplSSE, DataType::Float32, 8, 1>(src, dst, kernels);
            break;
        }
    }

    void conv3x3_1to8_identity_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplSSE, DataType::UInt8, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplSSE, DataType::UInt16, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float16:
            conv3x3_cin1<OpImplSSE, DataType::Float16, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplSSE, DataType::Float32, 8>(src, dst, kernels, biases, Identity{});
            break;
        }
    }

    void conv3x3_1to8_prelu_sse(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplSSE, DataType::UInt8, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplSSE, DataType::UInt16, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        case Image::Float16:
            conv3x3_cin1<OpImplSSE, DataType::Float16, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplSSE, DataType::Float32, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        }
    }

    void conv3x3_8to8_prelu_sse(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        conv3x3_float<OpImplSSE, 8, 8>(src, dst, kernels, biases, PReLU{ alphas });
    }
    void conv3x3_8to8_identity_residual_sse(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale)
    {
        conv3x3_float<OpImplSSE, 8, 8>(src, dst, kernels, biases, Identity{}, ResidualArg{ id, scale });
    }
    void conv3x3_8to8_identity_residual_conv1x1_8to8_prelu_add_sse(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1,
        const Image& id, const float scale,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_float<OpImplSSE, 8, 8, 8, false, false>(
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
            conv3x3_identity_pixelshuffle_float<OpImplSSE, DataType::UInt8, 8, 2>(src, dst, kernels, biases, ResidualArg{ id, 1.0f });
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<OpImplSSE, DataType::UInt16, 8, 2>(src, dst, kernels, biases, ResidualArg{ id, 1.0f });
            break;
        case Image::Float16:
            conv3x3_identity_pixelshuffle_float<OpImplSSE, DataType::Float16, 8, 2>(src, dst, kernels, biases, ResidualArg{ id, 1.0f });
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<OpImplSSE, DataType::Float32, 8, 2>(src, dst, kernels, biases, ResidualArg{ id, 1.0f });
            break;
        }
    }

    void conv3x3_1to16_identity_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplSSE, DataType::UInt8, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplSSE, DataType::UInt16, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float16:
            conv3x3_cin1<OpImplSSE, DataType::Float16, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplSSE, DataType::Float32, 16>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_16to16_relu_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_float<OpImplSSE, 16, 16>(src, dst, kernels, biases, ReLU{});
    }
    void conv3x3_16to16_identity_add_sse(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
        conv3x3_float<OpImplSSE, 16, 16>(src, dst, kernels, biases, Identity{}, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_16to4_identity_pixelshuffle_4to1_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<OpImplSSE, DataType::UInt8, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<OpImplSSE, DataType::UInt16, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float16:
            conv3x3_identity_pixelshuffle_float<OpImplSSE, DataType::Float16, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<OpImplSSE, DataType::Float32, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        }
    }

    void conv3x3_1to32_identity_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplSSE, DataType::UInt8, 32>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplSSE, DataType::UInt16, 32>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float16:
            conv3x3_cin1<OpImplSSE, DataType::Float16, 32>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplSSE, DataType::Float32, 32>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_32to32_relu_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_float<OpImplSSE, 32, 32>(src, dst, kernels, biases, ReLU{});
    }
    void conv3x3_32to32_identity_add_sse(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
        conv3x3_float<OpImplSSE, 32, 32>(src, dst, kernels, biases, Identity{}, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_32to4_identity_pixelshuffle_4to1_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<OpImplSSE, DataType::UInt8, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<OpImplSSE, DataType::UInt16, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float16:
            conv3x3_identity_pixelshuffle_float<OpImplSSE, DataType::Float16, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<OpImplSSE, DataType::Float32, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        }
    }

    void conv5x5_1to8_identity_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv5x5_cin1<OpImplSSE, DataType::UInt8, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv5x5_cin1<OpImplSSE, DataType::UInt16, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float16:
            conv5x5_cin1<OpImplSSE, DataType::Float16, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv5x5_cin1<OpImplSSE, DataType::Float32, 8>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_8to8_prelu_conv1x1_8to8_add_prelu_sse(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_float<OpImplSSE, 8, 8, 8, false, true>(
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
            conv3x3_identity_pixelshuffle_float<OpImplSSE, DataType::UInt8, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<OpImplSSE, DataType::UInt16, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float16:
            conv3x3_identity_pixelshuffle_float<OpImplSSE, DataType::Float16, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<OpImplSSE, DataType::Float32, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        }
    }

    void conv5x5_1to16_identity_sse(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv5x5_cin1<OpImplSSE, DataType::UInt8, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv5x5_cin1<OpImplSSE, DataType::UInt16, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float16:
            conv5x5_cin1<OpImplSSE, DataType::Float16, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv5x5_cin1<OpImplSSE, DataType::Float32, 16>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_16to16_prelu_sse(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        conv3x3_float<OpImplSSE, 16, 16>(src, dst, kernels, biases, PReLU{ alphas });
    }
    void conv3x3_16to16_prelu_conv1x1_16to16_add_prelu_sse(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_float<OpImplSSE, 16, 16, 16, false, true>(
            src, dst,
            kernels1, biases1, PReLU{ alphas1 }, nullptr,
            kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
        );
    }
}
