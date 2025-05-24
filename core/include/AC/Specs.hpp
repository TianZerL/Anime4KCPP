#ifndef AC_SPECS_HPP
#define AC_SPECS_HPP

namespace ac::specs
{
    constexpr const char* ModelNameList[] = {
        "acnet-gan",
        "acnet-hdn0",
        "acnet-hdn1",
        "acnet-hdn2",
        "acnet-hdn3",
        "arnet-hdn",
    };

    constexpr const char* ModelDescriptionList[] = {
        "Lightweight CNN, detail enhancement.",    // acnet-gan
        "Lightweight CNN, mild denoising.",        // acnet-hdn0
        "Lightweight CNN, moderate denoising.",    // acnet-hdn1
        "Lightweight CNN, heavy denoising.",       // acnet-hdn2
        "Lightweight CNN, extreme denoising.",     // acnet-hdn3
        "Lightweight ResNet, mild denoising.",     // arnet-hdn
    };

    constexpr const char* ProcessorNameList[] = {
        "cpu",
#   ifdef AC_CORE_WITH_OPENCL
        "opencl",
#   endif
#   ifdef AC_CORE_WITH_CUDA
        "cuda",
#   endif
    };

    constexpr const char* ProcessorDescriptionList[] = {
        "General-purpose CPU processing with optional SIMD acceleration.",      // cpu
#   ifdef AC_CORE_WITH_OPENCL
        "Cross-platform acceleration requiring OpenCL 1.2+ compliant devices.", // opencl
#   endif
#   ifdef AC_CORE_WITH_CUDA
        "NVIDIA GPU acceleration requiring Compute Capability 5.0+.",           // cuda
#   endif
    };
}

#endif
