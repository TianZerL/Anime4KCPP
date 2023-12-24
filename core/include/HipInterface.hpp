#ifndef ANIME4KCPP_CORE_HIP_INTERFACE_HPP
#define ANIME4KCPP_CORE_HIP_INTERFACE_HPP

#ifdef ENABLE_HIP

#include <string>
#include <hip/hip_runtime_api.h>

namespace Anime4KCPP::Hip
{
    enum class ACHipDataType
    {
        AC_8U, AC_16U, AC_32F
    };

    struct ACHipParamAnime4K09
    {
        int orgW, orgH;
        int W, H;
        std::size_t stride;
        int passes, pushColorCount;
        float strengthColor, strengthGradient;
    };

    struct ACHipParamACNet
    {
        int orgW, orgH;
        std::size_t stride;
    };

    void cuRunKernelAnime4K09(const void* inputData, void* outputData, ACHipDataType type, ACHipParamAnime4K09* param, hipStream_t& stream);

    void cuRunKernelACNetHDN0(const void* inputData, void* outputData, ACHipDataType type, ACHipParamACNet* param);
    void cuRunKernelACNetHDN1(const void* inputData, void* outputData, ACHipDataType type, ACHipParamACNet* param);
    void cuRunKernelACNetHDN2(const void* inputData, void* outputData, ACHipDataType type, ACHipParamACNet* param);
    void cuRunKernelACNetHDN3(const void* inputData, void* outputData, ACHipDataType type, ACHipParamACNet* param);

    void cuSetDeviceID(const int id) noexcept;
    int cuGetDeviceID() noexcept;
    void cuReleaseHip() noexcept;

    int cuGetDeviceCount() noexcept;
    std::string cuGetDeviceInfo(const int id);
    std::string cuGetHipInfo();
    bool cuCheckDeviceSupport(const int id) noexcept;
}

#endif // ENABLE_HIP

#endif // !ANIME4KCPP_CORE_HIP_INTERFACE_HPP
