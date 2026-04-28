#include <Eigen/Core>

#include "AC/Core/Image.hpp"
#include "AC/Core/Util.hpp"

#include "AC/Core/Internal/Processor/CPU/Common.hpp"

namespace ac::core::cpu
{
    struct OpImplEigen3
    {
        template <int vsize>
        static float dot(const float* v1, const float* v2) noexcept
        {
            Eigen::Map<const Eigen::Vector<float, vsize>> r1{ v1 };
            Eigen::Map<const Eigen::Vector<float, vsize>> r2{ v2 };

            return r1.dot(r2);
        }

        template <int cout, int cpos>
        static void conv_cin1(const float rptr[], float* const out, const float* const kernels, const float* const biases) noexcept
        {
            Eigen::Map<const Eigen::Vector<float, cpos>> r{ rptr };

            for (int n = 0; n < cout; n++)
            {
                Eigen::Map<const Eigen::Vector<float, cpos>> k{ kernels + n * cpos };
                out[n] = r.dot(k) + biases[n];
            }
        }

        template <int cin, int cout, int cpos>
        static void conv(const float* rptr[], float* const out, const float* const kernels, const float* const biases) noexcept
        {
            Eigen::Matrix<float, cin * cpos, 1> r;
            for (int p = 0; p < cpos; p++) r.template segment<cin>(p * cin) = Eigen::Map<const Eigen::Matrix<float, cin, 1>>{ rptr[p] };

            Eigen::Map<const Eigen::Matrix<float, cout, cin * cpos, Eigen::RowMajor>> k{ kernels };
            Eigen::Map<const Eigen::Matrix<float, cout, 1>> b{ biases };
            Eigen::Map<Eigen::Matrix<float, cout, 1>> s{ out };

            s.noalias() = k * r + b;
        }
    };

    void conv3x3_1to8_relu_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplEigen3, std::uint8_t, 8>(src, dst, kernels, biases, ReLU{});
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplEigen3, std::uint16_t, 8>(src, dst, kernels, biases, ReLU{});
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplEigen3, float, 8>(src, dst, kernels, biases, ReLU{});
            break;
        }
    }
    void conv3x3_8to8_relu_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_float<OpImplEigen3, 8, 8>(src, dst, kernels, biases, ReLU{});
    }
    void deconv2x2_8to1_eigen3(const Image& src, Image& dst, const float* kernels)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            deconv2x2<OpImplEigen3, std::uint8_t, 8, 1>(src, dst, kernels);
            break;
        case Image::UInt16:
            deconv2x2<OpImplEigen3, std::uint16_t, 8, 1>(src, dst, kernels);
            break;
        case Image::Float32:
            deconv2x2<OpImplEigen3, float, 8, 1>(src, dst, kernels);
            break;
        }
    }

    void conv3x3_1to8_prelu_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplEigen3, std::uint8_t, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplEigen3, std::uint16_t, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplEigen3, float, 8>(src, dst, kernels, biases, PReLU{ alphas });
            break;
        }
    }

    void conv3x3_1to8_identity_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplEigen3, std::uint8_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplEigen3, std::uint16_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplEigen3, float, 8>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_8to8_prelu_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        conv3x3_float<OpImplEigen3, 8, 8>(src, dst, kernels, biases, PReLU{ alphas });
    }
    void conv3x3_8to8_identity_residual_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id, const float scale)
    {
        conv3x3_float<OpImplEigen3, 8, 8>(src, dst, kernels, biases, Identity{}, ResidualArg{ id, scale });
    }
    void conv3x3_8to8_identity_residual_conv1x1_8to8_prelu_add_eigen3(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1,
        const Image& id, const float scale,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_float<OpImplEigen3, 8, 8, 8, false, false>(
            src, dst,
            kernels1, biases1, Identity{}, ResidualArg{ id, scale },
            kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
        );
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_add_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& id)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<OpImplEigen3, std::uint8_t, 8, 2>(src, dst, kernels, biases, ResidualArg{id, 1.0f});
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<OpImplEigen3, std::uint16_t, 8, 2>(src, dst, kernels, biases, ResidualArg{id, 1.0f});
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<OpImplEigen3, float, 8, 2>(src, dst, kernels, biases, ResidualArg{id, 1.0f});
            break;
        }
    }

    void conv3x3_1to16_identity_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplEigen3, std::uint8_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplEigen3, std::uint16_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplEigen3, float, 16>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_16to16_relu_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_float<OpImplEigen3, 16, 16>(src, dst, kernels, biases, ReLU{});
    }
    void conv3x3_16to16_identity_add_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
        conv3x3_float<OpImplEigen3, 16, 16>(src, dst, kernels, biases, Identity{}, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_16to4_identity_pixelshuffle_4to1_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<OpImplEigen3, std::uint8_t, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<OpImplEigen3, std::uint16_t, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<OpImplEigen3, float, 16, 2>(src, dst, kernels, biases, nullptr);
            break;
        }
    }

    void conv3x3_1to32_identity_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv3x3_cin1<OpImplEigen3, std::uint8_t, 32>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv3x3_cin1<OpImplEigen3, std::uint16_t, 32>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv3x3_cin1<OpImplEigen3, float, 32>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_32to32_relu_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        conv3x3_float<OpImplEigen3, 32, 32>(src, dst, kernels, biases, ReLU{});
    }
    void conv3x3_32to32_identity_add_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases, const Image& feat)
    {
        conv3x3_float<OpImplEigen3, 32, 32>(src, dst, kernels, biases, Identity{}, ResidualArg{ feat, 1.0f });
    }
    void conv3x3_32to4_identity_pixelshuffle_4to1_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<OpImplEigen3, std::uint8_t, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<OpImplEigen3, std::uint16_t, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<OpImplEigen3, float, 32, 2>(src, dst, kernels, biases, nullptr);
            break;
        }
    }

    void conv5x5_1to8_identity_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv5x5_cin1<OpImplEigen3, std::uint8_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv5x5_cin1<OpImplEigen3, std::uint16_t, 8>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv5x5_cin1<OpImplEigen3, float, 8>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_8to8_prelu_conv1x1_8to8_add_prelu_eigen3(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_float<OpImplEigen3, 8, 8, 8, false, true>(
            src, dst,
            kernels1, biases1, PReLU{ alphas1 }, nullptr,
            kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
        );
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (dst.type())
        {
        case Image::UInt8:
            conv3x3_identity_pixelshuffle_float<OpImplEigen3, std::uint8_t, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::UInt16:
            conv3x3_identity_pixelshuffle_float<OpImplEigen3, std::uint16_t, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        case Image::Float32:
            conv3x3_identity_pixelshuffle_float<OpImplEigen3, float, 8, 2>(src, dst, kernels, biases, nullptr);
            break;
        }
    }

    void conv5x5_1to16_identity_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases)
    {
        switch (src.type())
        {
        case Image::UInt8:
            conv5x5_cin1<OpImplEigen3, std::uint8_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            conv5x5_cin1<OpImplEigen3, std::uint16_t, 16>(src, dst, kernels, biases, Identity{});
            break;
        case Image::Float32:
            conv5x5_cin1<OpImplEigen3, float, 16>(src, dst, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_16to16_prelu_eigen3(const Image& src, Image& dst, const float* kernels, const float* biases, const float* alphas)
    {
        conv3x3_float<OpImplEigen3, 16, 16>(src, dst, kernels, biases, PReLU{ alphas });
    }
    void conv3x3_16to16_prelu_conv1x1_16to16_add_prelu_eigen3(
        const Image& src, Image& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const Image& feat)
    {
        conv3x3_conv1x1_float<OpImplEigen3, 16, 16, 16, false, true>(
            src, dst,
            kernels1, biases1, PReLU{ alphas1 }, nullptr,
            kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat, 1.0f }
        );
    }
}
