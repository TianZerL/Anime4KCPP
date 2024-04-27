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
ac::core::model::ACNet::ACNet(const ACNet&) noexcept = default;
ac::core::model::ACNet::ACNet(ACNet&&) noexcept = default;
ac::core::model::ACNet::~ACNet() noexcept = default;
ac::core::model::ACNet& ac::core::model::ACNet::operator=(const ACNet&) noexcept = default;
ac::core::model::ACNet& ac::core::model::ACNet::operator=(ACNet&&) noexcept = default;

const float* ac::core::model::ACNet::kernels(const int idx) const noexcept
{
    if (idx == 0)
        return kptr;
    if (idx > 0 && idx < 10)
        return kptr + (idx - 1) * 8 * 8 * 9 + 8 * 9;
    return nullptr;
}
const float* ac::core::model::ACNet::biases(const int idx) const noexcept
{
    if (idx >= 0 && idx < 9)
        return bptr + idx * 8;
    return nullptr;
}
