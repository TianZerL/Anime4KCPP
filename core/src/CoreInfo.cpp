#include"CoreInfo.hpp"

const char* Anime4KCPP::CoreInfo::version()
{
    return 
        ANIME4KCPP_CORE_VERSION
        "-"
        ANIME4KCPP_CORE_VERSION_STATUS
        " "
        ANIME4KCPP_CORE_BUILD_DATE
        " "
        __TIME__;
}

const char* Anime4KCPP::CoreInfo::CPUOptimizationMode()
{
#if defined(ENABLE_OPENCV_DNN)
    return "OpenCV DNN";
#elif defined(USE_RYZEN)
    return "SIMD (AVX2)";
#elif defined(USE_EIGEN3)
    return "Eigen3";
#else
    return "Normal";
#endif // ENABLE_OPENCV_DNN
}

#define AC_ENUM_ITEM
#include"ACRegister.hpp"
#undef AC_ENUM_ITEM
#define PROCESSOR_STRING(S) PROCESSOR_STRING_IMPL(S)
#define PROCESSOR_STRING_IMPL(S) #S
const char* Anime4KCPP::CoreInfo::supportedProcessors()
{
    return
        PROCESSOR_STRING(PROCESSOR_ENUM);
}
