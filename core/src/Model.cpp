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
    case Variant::B4_HDN:
        blockNum = 4;
        kptr = param::ARNet_B4_HDN_NHWC_kernels;
        bptr = param::ARNet_B4_HDN_NHWC_biases;
        break;
    case Variant::B4_HDNS:
        blockNum = 4;
        kptr = param::ARNet_B4_HDNS_NHWC_kernels;
        bptr = param::ARNet_B4_HDNS_NHWC_biases;
        break;
    case Variant::B4_LE:
        blockNum = 4;
        kptr = param::ARNet_B4_LE_NHWC_kernels;
        bptr = param::ARNet_B4_LE_NHWC_biases;
        break;
    case Variant::B4_LS:
        blockNum = 4;
        kptr = param::ARNet_B4_LS_NHWC_kernels;
        bptr = param::ARNet_B4_LS_NHWC_biases;
        break;
    case Variant::B8_HDN:
        blockNum = 8;
        kptr = param::ARNet_B8_HDN_NHWC_kernels;
        bptr = param::ARNet_B8_HDN_NHWC_biases;
        break;
    case Variant::B8_HDNS:
        blockNum = 8;
        kptr = param::ARNet_B8_HDNS_NHWC_kernels;
        bptr = param::ARNet_B8_HDNS_NHWC_biases;
        break;
    case Variant::B8_LE:
        blockNum = 8;
        kptr = param::ARNet_B8_LE_NHWC_kernels;
        bptr = param::ARNet_B8_LE_NHWC_biases;
        break;
    case Variant::B8_LS:
        blockNum = 8;
        kptr = param::ARNet_B8_LS_NHWC_kernels;
        bptr = param::ARNet_B8_LS_NHWC_biases;
        break;
    case Variant::B16_HDN:
        blockNum = 16;
        kptr = param::ARNet_B16_HDN_NHWC_kernels;
        bptr = param::ARNet_B16_HDN_NHWC_biases;
        break;
    case Variant::B16_HDNS:
        blockNum = 16;
        kptr = param::ARNet_B16_HDNS_NHWC_kernels;
        bptr = param::ARNet_B16_HDNS_NHWC_biases;
        break;
    case Variant::B16_LE:
        blockNum = 16;
        kptr = param::ARNet_B16_LE_NHWC_kernels;
        bptr = param::ARNet_B16_LE_NHWC_biases;
        break;
    case Variant::B16_LS:
        blockNum = 16;
        kptr = param::ARNet_B16_LS_NHWC_kernels;
        bptr = param::ARNet_B16_LS_NHWC_biases;
        break;
    case Variant::B24_HDN:
        blockNum = 24;
        kptr = param::ARNet_B24_HDN_NHWC_kernels;
        bptr = param::ARNet_B24_HDN_NHWC_biases;
        break;
    case Variant::B24_HDNS:
        blockNum = 24;
        kptr = param::ARNet_B24_HDNS_NHWC_kernels;
        bptr = param::ARNet_B24_HDNS_NHWC_biases;
        break;
    case Variant::B24_LE:
        blockNum = 24;
        kptr = param::ARNet_B24_LE_NHWC_kernels;
        bptr = param::ARNet_B24_LE_NHWC_biases;
        break;
    case Variant::B24_LS:
        blockNum = 24;
        kptr = param::ARNet_B24_LS_NHWC_kernels;
        bptr = param::ARNet_B24_LS_NHWC_biases;
        break;
    case Variant::B32_HDN:
        blockNum = 32;
        kptr = param::ARNet_B32_HDN_NHWC_kernels;
        bptr = param::ARNet_B32_HDN_NHWC_biases;
        break;
    case Variant::B32_HDNS:
        blockNum = 32;
        kptr = param::ARNet_B32_HDNS_NHWC_kernels;
        bptr = param::ARNet_B32_HDNS_NHWC_biases;
        break;
    case Variant::B32_LE:
        blockNum = 32;
        kptr = param::ARNet_B32_LE_NHWC_kernels;
        bptr = param::ARNet_B32_LE_NHWC_biases;
        break;
    case Variant::B32_LS:
        blockNum = 32;
        kptr = param::ARNet_B32_LS_NHWC_kernels;
        bptr = param::ARNet_B32_LS_NHWC_biases;
        break;
    case Variant::B48_HDN:
        blockNum = 48;
        kptr = param::ARNet_B48_HDN_NHWC_kernels;
        bptr = param::ARNet_B48_HDN_NHWC_biases;
        break;
    case Variant::B48_HDNS:
        blockNum = 48;
        kptr = param::ARNet_B48_HDNS_NHWC_kernels;
        bptr = param::ARNet_B48_HDNS_NHWC_biases;
        break;
    case Variant::B48_LE:
        blockNum = 48;
        kptr = param::ARNet_B48_LE_NHWC_kernels;
        bptr = param::ARNet_B48_LE_NHWC_biases;
        break;
    case Variant::B48_LS:
        blockNum = 48;
        kptr = param::ARNet_B48_LS_NHWC_kernels;
        bptr = param::ARNet_B48_LS_NHWC_biases;
        break;
    case Variant::B64_HDN:
        blockNum = 64;
        kptr = param::ARNet_B64_HDN_NHWC_kernels;
        bptr = param::ARNet_B64_HDN_NHWC_biases;
        break;
    case Variant::B64_HDNS:
        blockNum = 64;
        kptr = param::ARNet_B64_HDNS_NHWC_kernels;
        bptr = param::ARNet_B64_HDNS_NHWC_biases;
        break;
    case Variant::B64_LE:
        blockNum = 64;
        kptr = param::ARNet_B64_LE_NHWC_kernels;
        bptr = param::ARNet_B64_LE_NHWC_biases;
        break;
    case Variant::B64_LS:
        blockNum = 64;
        kptr = param::ARNet_B64_LS_NHWC_kernels;
        bptr = param::ARNet_B64_LS_NHWC_biases;
        break;
    }
}
