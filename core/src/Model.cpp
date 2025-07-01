#include "AC/Core/Half.hpp"
#include "AC/Core/Model.hpp"
#include "AC/Core/Model/Param.hpp"

ac::core::model::ACNet::ACNet(const Variant v) noexcept : variant(v) {};
template<>
const void* ac::core::model::ACNet::kernels<float>() const noexcept
{
    switch (variant)
    {
    case Variant::GAN: return param::ACNet_GAN_NHWC_kernels_float;
    case Variant::HDN0: return param::ACNet_HDN0_NHWC_kernels_float;
    case Variant::HDN1: return param::ACNet_HDN1_NHWC_kernels_float;
    case Variant::HDN2: return param::ACNet_HDN2_NHWC_kernels_float;
    case Variant::HDN3: return param::ACNet_HDN3_NHWC_kernels_float;
    default: return nullptr;
    }
}
template<>
const void* ac::core::model::ACNet::biases<float>() const noexcept
{
    switch (variant)
    {
    case Variant::GAN: return param::ACNet_GAN_NHWC_biases_float;
    case Variant::HDN0: return param::ACNet_HDN0_NHWC_biases_float;
    case Variant::HDN1: return param::ACNet_HDN1_NHWC_biases_float;
    case Variant::HDN2: return param::ACNet_HDN2_NHWC_biases_float;
    case Variant::HDN3: return param::ACNet_HDN3_NHWC_biases_float;
    default: return nullptr;
    }
}
template<>
const void* ac::core::model::ACNet::kernels<ac::core::Half>() const noexcept
{
    switch (variant)
    {
    case Variant::GAN: return param::ACNet_GAN_NHWC_kernels_half;
    case Variant::HDN0: return param::ACNet_HDN0_NHWC_kernels_half;
    case Variant::HDN1: return param::ACNet_HDN1_NHWC_kernels_half;
    case Variant::HDN2: return param::ACNet_HDN2_NHWC_kernels_half;
    case Variant::HDN3: return param::ACNet_HDN3_NHWC_kernels_half;
    default: return nullptr;
    }
}
template<>
const void* ac::core::model::ACNet::biases<ac::core::Half>() const noexcept
{
    switch (variant)
    {
    case Variant::GAN: return param::ACNet_GAN_NHWC_biases_half;
    case Variant::HDN0: return param::ACNet_HDN0_NHWC_biases_half;
    case Variant::HDN1: return param::ACNet_HDN1_NHWC_biases_half;
    case Variant::HDN2: return param::ACNet_HDN2_NHWC_biases_half;
    case Variant::HDN3: return param::ACNet_HDN3_NHWC_biases_half;
    default: return nullptr;
    }
}

ac::core::model::ARNet::ARNet(const Variant v) noexcept : variant(v) {};
template<>
const void* ac::core::model::ARNet::kernels<float>() const noexcept
{
    switch (variant)
    {
    case Variant::HDN: return param::ARNet_HDN_NHWC_kernels_float;
    default: return nullptr;
    }
}
template<>
const void* ac::core::model::ARNet::biases<float>() const noexcept
{
    switch (variant)
    {
    case Variant::HDN: return param::ARNet_HDN_NHWC_biases_float;
    default: return nullptr;
    }
}
template<>
const void* ac::core::model::ARNet::kernels<ac::core::Half>() const noexcept
{
    switch (variant)
    {
    case Variant::HDN: return param::ARNet_HDN_NHWC_kernels_half;
    default: return nullptr;
    }
}
template<>
const void* ac::core::model::ARNet::biases<ac::core::Half>() const noexcept
{
    switch (variant)
    {
    case Variant::HDN: return param::ARNet_HDN_NHWC_biases_half;
    default: return nullptr;
    }
}
