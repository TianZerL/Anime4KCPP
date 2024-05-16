#include "AC/Core/Model/ACNet.hpp"

#include "AC/Core/Model/Param/ACNet.p"

ac::core::model::ACNet::ACNet(const Variant v) noexcept : kptr(nullptr), bptr(nullptr), kptrFP16(nullptr), bptrFP16(nullptr)
{
    switch (v)
    {
    case Variant::HDN0:
        kptr = param::ACNet_HDN0_NHWC_kernels;
        bptr = param::ACNet_HDN0_NHWC_biases;
        kptrFP16 = param::ACNet_HDN0_NHWC_FP16_kernels;
        bptrFP16 = param::ACNet_HDN0_NHWC_FP16_biases;
        break;
    case Variant::HDN1:
        kptr = param::ACNet_HDN1_NHWC_kernels;
        bptr = param::ACNet_HDN1_NHWC_biases;
        kptrFP16 = param::ACNet_HDN1_NHWC_FP16_kernels;
        bptrFP16 = param::ACNet_HDN1_NHWC_FP16_biases;
        break;
    case Variant::HDN2:
        kptr = param::ACNet_HDN2_NHWC_kernels;
        bptr = param::ACNet_HDN2_NHWC_biases;
        kptrFP16 = param::ACNet_HDN2_NHWC_FP16_kernels;
        bptrFP16 = param::ACNet_HDN2_NHWC_FP16_biases;
        break;
    case Variant::HDN3:
        kptr = param::ACNet_HDN3_NHWC_kernels;
        bptr = param::ACNet_HDN3_NHWC_biases;
        kptrFP16 = param::ACNet_HDN3_NHWC_FP16_kernels;
        bptrFP16 = param::ACNet_HDN3_NHWC_FP16_biases;
        break;
    }
}
