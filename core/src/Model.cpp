#include "AC/Core/Model.hpp"
#include "AC/Core/Model/Param.hpp"

ac::core::model::ACNetClassic::ACNetClassic(const Variant v) noexcept : kptr(nullptr), bptr(nullptr)
{
    switch (v)
    {
    case Variant::GAN:
        kptr = param::ACNetClassic_GAN_NHWC_kernels;
        bptr = param::ACNetClassic_GAN_NHWC_biases;
        break;
    case Variant::HDN0:
        kptr = param::ACNetClassic_HDN0_NHWC_kernels;
        bptr = param::ACNetClassic_HDN0_NHWC_biases;
        break;
    case Variant::HDN1:
        kptr = param::ACNetClassic_HDN1_NHWC_kernels;
        bptr = param::ACNetClassic_HDN1_NHWC_biases;
        break;
    case Variant::HDN2:
        kptr = param::ACNetClassic_HDN2_NHWC_kernels;
        bptr = param::ACNetClassic_HDN2_NHWC_biases;
        break;
    case Variant::HDN3:
        kptr = param::ACNetClassic_HDN3_NHWC_kernels;
        bptr = param::ACNetClassic_HDN3_NHWC_biases;
        break;
    }
}

template<int F>
ac::core::model::ACNet<F>::ACNet(const Variant v) noexcept : kptr(nullptr), bptr(nullptr), blockNum(0)
{
    if constexpr (F == 8)
    {
        switch (v)
        {
        case Variant::B8_NORMAL:
            blockNum = 8;
            //kptr = param::ACNet_F8B8_NHWC_kernels;
            //bptr = param::ACNet_F8B8_NHWC_biases;
            break;
        case Variant::B8_HDN:
            blockNum = 16;
            //kptr = param::ACNet_F8B8_NHWC_kernels;
            //bptr = param::ACNet_F8B8_NHWC_biases;
            break;
        }
    }
    else static_assert(F == 8, "Unsupported ACNet model");
}

template class ac::core::model::ACNet<8>;

template<int F>
ac::core::model::ARNet<F>::ARNet(const Variant v) noexcept : kptr(nullptr), bptr(nullptr), aptr(nullptr), blockNum(0)
{
    if constexpr (F == 8)
    {
        switch (v)
        {
        case Variant::B8_NORMAL:
            blockNum = 8;
            kptr = param::ARNet_F8B8_NHWC_kernels;
            bptr = param::ARNet_F8B8_NHWC_biases;
            aptr = param::ARNet_F8B8_NHWC_alphas;
            break;
        case Variant::B16_NORMAL:
            blockNum = 16;
            kptr = param::ARNet_F8B16_NHWC_kernels;
            bptr = param::ARNet_F8B16_NHWC_biases;
            aptr = param::ARNet_F8B16_NHWC_alphas;
            break;
        case Variant::B32_NORMAL:
            blockNum = 32;
            kptr = param::ARNet_F8B32_NHWC_kernels;
            bptr = param::ARNet_F8B32_NHWC_biases;
            aptr = param::ARNet_F8B32_NHWC_alphas;
            break;
        case Variant::B64_NORMAL:
            blockNum = 64;
            kptr = param::ARNet_F8B64_NHWC_kernels;
            bptr = param::ARNet_F8B64_NHWC_biases;
            aptr = param::ARNet_F8B64_NHWC_alphas;
            break;
        }
    }
    else static_assert(F == 8, "Unsupported ARNet model");
}

template class ac::core::model::ARNet<8>;

template<int F>
ac::core::model::ArtCNN<F>::ArtCNN(const Variant v) noexcept : kptr(nullptr), bptr(nullptr), blockNum(0)
{
    if constexpr (F == 16)
    {
        switch (v)
        {
        case Variant::NORMAL:
            blockNum = 4;
            kptr = param::ArtCNN_C4F16_NHWC_kernels;
            bptr = param::ArtCNN_C4F16_NHWC_biases;
            break;
        case Variant::DN:
            blockNum = 4;
            kptr = param::ArtCNN_C4F16_DN_NHWC_kernels;
            bptr = param::ArtCNN_C4F16_DN_NHWC_biases;
            break;
        case Variant::DS:
            blockNum = 4;
            kptr = param::ArtCNN_C4F16_DS_NHWC_kernels;
            bptr = param::ArtCNN_C4F16_DS_NHWC_biases;
            break;
        }
    }
    else if constexpr (F == 32)
    {
        switch (v)
        {
        case Variant::NORMAL:
            blockNum = 4;
            kptr = param::ArtCNN_C4F32_NHWC_kernels;
            bptr = param::ArtCNN_C4F32_NHWC_biases;
            break;
        case Variant::DN:
            blockNum = 4;
            kptr = param::ArtCNN_C4F32_DN_NHWC_kernels;
            bptr = param::ArtCNN_C4F32_DN_NHWC_biases;
            break;
        case Variant::DS:
            blockNum = 4;
            kptr = param::ArtCNN_C4F32_DS_NHWC_kernels;
            bptr = param::ArtCNN_C4F32_DS_NHWC_biases;
            break;
        }
    }
    else static_assert(F == 32, "Unsupported ArtCNN model");
}

template class ac::core::model::ArtCNN<16>;
template class ac::core::model::ArtCNN<32>;

template<int F>
ac::core::model::FSRCNNX<F>::FSRCNNX(const Variant v) noexcept : kptr(nullptr), bptr(nullptr), aptr(nullptr), blockNum(0)
{
    if constexpr (F == 8)
    {
        switch (v)
        {
        case Variant::NORMAL:
            blockNum = 4;
            kptr = param::FSRCNNX_F8_NHWC_kernels;
            bptr = param::FSRCNNX_F8_NHWC_biases;
            aptr = param::FSRCNNX_F8_NHWC_alphas;
            break;
        case Variant::DISTORT_PLUS:
            blockNum = 4;
            kptr = param::FSRCNNX_F8_DistortPlus_NHWC_kernels;
            bptr = param::FSRCNNX_F8_DistortPlus_NHWC_biases;
            aptr = param::FSRCNNX_F8_DistortPlus_NHWC_alphas;
            break;
        }
    }
    else if constexpr (F == 16)
    {
        switch (v)
        {
        case Variant::NORMAL:
            blockNum = 4;
            kptr = param::FSRCNNX_F16_NHWC_kernels;
            bptr = param::FSRCNNX_F16_NHWC_biases;
            aptr = param::FSRCNNX_F16_NHWC_alphas;
            break;
        case Variant::DISTORT_PLUS:
            blockNum = 4;
            kptr = param::FSRCNNX_F16_DistortPlus_NHWC_kernels;
            bptr = param::FSRCNNX_F16_DistortPlus_NHWC_biases;
            aptr = param::FSRCNNX_F16_DistortPlus_NHWC_alphas;
            break;
        }
    }
    else static_assert(F == 16, "Unsupported FSRCNNX model");
}

template class ac::core::model::FSRCNNX<8>;
template class ac::core::model::FSRCNNX<16>;
