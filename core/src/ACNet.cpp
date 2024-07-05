#include "AC/Core/Model/ACNet.hpp"

namespace ac::core::model::param
{
#include "AC/Core/Model/Param/ACNet.p"
}

ac::core::model::ACNet::ACNet(const Variant v) noexcept : kptr(nullptr), bptr(nullptr)
{
    switch (v)
    {
    case Variant::HDN0:
        kptr = param::ACNet_HDN0_NHWC_Kernels;
        bptr = param::ACNet_HDN0_NHWC_Biases;
        break;
    case Variant::HDN1:
        kptr = param::ACNet_HDN1_NHWC_Kernels;
        bptr = param::ACNet_HDN1_NHWC_Biases;
        break;
    case Variant::HDN2:
        kptr = param::ACNet_HDN2_NHWC_Kernels;
        bptr = param::ACNet_HDN2_NHWC_Biases;
        break;
    case Variant::HDN3:
        kptr = param::ACNet_HDN3_NHWC_Kernels;
        bptr = param::ACNet_HDN3_NHWC_Biases;
        break;
    }
}
