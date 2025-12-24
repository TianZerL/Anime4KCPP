#ifndef AC_SPECS_HPP
#define AC_SPECS_HPP

namespace ac::specs
{
    constexpr const char* ModelList[] = {
        "acnet-gan",
        "acnet-hdn0",
        "acnet-hdn1",
        "acnet-hdn2",
        "acnet-hdn3",
        "arnet-b4-hdn",
        "arnet-b4-hdns",
        "arnet-b4-le",
        "arnet-b4-ls",
        "arnet-b8-hdn",
        "arnet-b8-hdns",
        "arnet-b8-le",
        "arnet-b8-ls",
        "arnet-b16-hdn",
        "arnet-b16-hdns",
        "arnet-b16-le",
        "arnet-b16-ls",
        "arnet-b24-hdn",
        "arnet-b24-hdns",
        "arnet-b24-le",
        "arnet-b24-ls",
        "arnet-b32-hdn",
        "arnet-b32-hdns",
        "arnet-b32-le",
        "arnet-b32-ls",
        "arnet-b48-hdn",
        "arnet-b48-hdns",
        "arnet-b48-le",
        "arnet-b48-ls",
        "arnet-b64-hdn",
        "arnet-b64-hdns",
        "arnet-b64-le",
        "arnet-b64-ls",
    };

    constexpr const char* ModelDescriptionList[] = {
        "Lightweight CNN, detail enhancement.",    // acnet-gan
        "Lightweight CNN, mild denoising.",        // acnet-hdn0
        "Lightweight CNN, moderate denoising.",    // acnet-hdn1
        "Lightweight CNN, heavy denoising.",       // acnet-hdn2
        "Lightweight CNN, extreme denoising.",     // acnet-hdn3
        "Lightweight ResNet, mild denoising.",     // arnet-b4-hdn
        "Lightweight ResNet, hdn with sharpening.",// arnet-b4-hdns
        "Lightweight ResNet, line enhancing.",     // arnet-b4-le
        "Lightweight ResNet, line sharpening.",    // arnet-b4-ls
        "Lightweight ResNet, mild denoising.",     // arnet-b8-hdn
        "Lightweight ResNet, hdn with sharpening.",// arnet-b8-hdns
        "Lightweight ResNet, line enhancing.",     // arnet-b8-le
        "Lightweight ResNet, line sharpening.",    // arnet-b4-ls
        "Lightweight ResNet, mild denoising.",     // arnet-b16-hdn
        "Lightweight ResNet, hdn with sharpening.",// arnet-b16-hdns
        "Lightweight ResNet, line enhancing.",     // arnet-b16-le
        "Lightweight ResNet, line sharpening.",    // arnet-b16-ls
        "Lightweight ResNet, mild denoising.",     // arnet-b24-hdn
        "Lightweight ResNet, hdn with sharpening.",// arnet-b24-hdns
        "Lightweight ResNet, line enhancing.",     // arnet-b24-le
        "Lightweight ResNet, line sharpening.",    // arnet-b24-ls
        "Lightweight ResNet, mild denoising.",     // arnet-b32-hdn
        "Lightweight ResNet, hdn with sharpening.",// arnet-b32-hdns
        "Lightweight ResNet, line enhancing.",     // arnet-b32-le
        "Lightweight ResNet, line sharpening.",    // arnet-b32-ls
        "Lightweight ResNet, mild denoising.",     // arnet-b48-hdn
        "Lightweight ResNet, hdn with sharpening.",// arnet-b48-hdns
        "Lightweight ResNet, line enhancing.",     // arnet-b48-le
        "Lightweight ResNet, line sharpening.",    // arnet-b48-ls
        "Lightweight ResNet, mild denoising.",     // arnet-b64-hdn
        "Lightweight ResNet, hdn with sharpening.",// arnet-b64-hdns
        "Lightweight ResNet, line enhancing.",     // arnet-b64-le
        "Lightweight ResNet, line sharpening.",    // arnet-b64-ls
    };

    constexpr const char* ProcessorList[] = {
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
