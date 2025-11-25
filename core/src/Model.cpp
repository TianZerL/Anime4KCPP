#include "AC/Core/Model.hpp"
#include "AC/Core/Model/Param.hpp"

ac::core::model::ACNet::ACNet(const Variant v) noexcept : kptr(nullptr), bptr(nullptr)
{
    switch (v)
    {
    case Variant::GAN:
        kptr = param::ACNet_GAN_NHWC_kernels;
        bptr = param::ACNet_GAN_NHWC_biases;
        break;
    case Variant::HDN0:
        kptr = param::ACNet_HDN0_NHWC_kernels;
        bptr = param::ACNet_HDN0_NHWC_biases;
        break;
    case Variant::HDN1:
        kptr = param::ACNet_HDN1_NHWC_kernels;
        bptr = param::ACNet_HDN1_NHWC_biases;
        break;
    case Variant::HDN2:
        kptr = param::ACNet_HDN2_NHWC_kernels;
        bptr = param::ACNet_HDN2_NHWC_biases;
        break;
    case Variant::HDN3:
        kptr = param::ACNet_HDN3_NHWC_kernels;
        bptr = param::ACNet_HDN3_NHWC_biases;
        break;
    }
}

ac::core::model::ARNet::ARNet(const Variant v) noexcept : kptr(nullptr), bptr(nullptr), blockNum(0)
{
    switch (v)
    {
    case Variant::S_HDN:
        blockNum = 8;
        kptr = param::ARNet_S_HDN_NHWC_kernels;
        bptr = param::ARNet_S_HDN_NHWC_biases;
        break;
    case Variant::S_LE:
        blockNum = 8;
        kptr = param::ARNet_S_LE_NHWC_kernels;
        bptr = param::ARNet_S_LE_NHWC_biases;
        break;
    case Variant::M_HDN:
        blockNum = 16;
        kptr = param::ARNet_M_HDN_NHWC_kernels;
        bptr = param::ARNet_M_HDN_NHWC_biases;
        break;
    case Variant::M_LE:
        blockNum = 16;
        kptr = param::ARNet_M_LE_NHWC_kernels;
        bptr = param::ARNet_M_LE_NHWC_biases;
        break;
    }
}
