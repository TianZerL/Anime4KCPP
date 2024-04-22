#include "AC/Core/Model/ACNet.hpp"

namespace ac::core::model::param
{
#include "AC/Core/Model/Param/ACNet.p"
}

struct ac::core::model::ACNet::ACNetData
{
    const float* kptr = nullptr;
    const float* bptr = nullptr;
};

ac::core::model::ACNet::ACNet(Variant v) : dptr(std::make_shared<ACNetData>())
{
    switch (v)
    {
    case Variant::HDN0:
        dptr->kptr = param::ACNet_HDN0_NHWC_Kernels;
        dptr->bptr = param::ACNet_HDN0_NHWC_Biases;
        break;
    case Variant::HDN1:
        dptr->kptr = param::ACNet_HDN1_NHWC_Kernels;
        dptr->bptr = param::ACNet_HDN1_NHWC_Biases;
        break;
    case Variant::HDN2:
        dptr->kptr = param::ACNet_HDN2_NHWC_Kernels;
        dptr->bptr = param::ACNet_HDN2_NHWC_Biases;
        break;
    case Variant::HDN3:
        dptr->kptr = param::ACNet_HDN3_NHWC_Kernels;
        dptr->bptr = param::ACNet_HDN3_NHWC_Biases;
        break;
    }
}
ac::core::model::ACNet::~ACNet() noexcept = default;

const float* ac::core::model::ACNet::kernels(const int idx) const noexcept
{
    if (idx == 0)
        return dptr->kptr;
    if (idx > 0 && idx < 10)
        return dptr->kptr + (idx - 1) * 8 * 8 * 9 + 8 * 9;
    return nullptr;
}
const float* ac::core::model::ACNet::biases(const int idx) const noexcept
{
    if (idx >= 0 && idx < 9)
        return dptr->bptr + idx * 8;
    return nullptr;
}
