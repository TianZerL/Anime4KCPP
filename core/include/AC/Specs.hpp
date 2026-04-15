#ifndef AC_SPECS_HPP
#define AC_SPECS_HPP

namespace ac::specs
{
    constexpr const char* ModelList[] = {
        "acnet-classic-gan",
        "acnet-classic-hdn0",
        "acnet-classic-hdn1",
        "acnet-classic-hdn2",
        "acnet-classic-hdn3",
        "acnet-f8b8",
        "acnet-f8b8-hdn",
        "arnet-f8b8",
        "arnet-f8b8-hdn",
        "arnet-f8b16",
        "arnet-f8b16-hdn",
        "arnet-f8b32",
        "arnet-f8b32-hdn",
        "arnet-f8b64",
        "artcnn-c4f16",
        "artcnn-c4f16-dn",
        "artcnn-c4f16-ds",
        "artcnn-c4f32",
        "artcnn-c4f32-dn",
        "artcnn-c4f32-ds",
        "fsrcnnx-f8",
        "fsrcnnx-f8-distort-plus",
        "fsrcnnx-f16",
        "fsrcnnx-f16-distort-plus",
    };

    constexpr const char* ModelDescriptionList[] = {
        "Lightweight CNN, detail enhancement.", // acnet-classic-gan
        "Lightweight CNN, mild denoising.",     // acnet-classic-hdn0
        "Lightweight CNN, moderate denoising.", // acnet-classic-hdn1
        "Lightweight CNN, heavy denoising.",    // acnet-classic-hdn2
        "Lightweight CNN, extreme denoising.",  // acnet-classic-hdn3
        "Lightweight CNN, without denoising.",  // acnet-f8b8
        "Lightweight CNN, mild denoising.",     // acnet-f8b8-hdn
        "ResNet for real-time tasks, without denoising.", // arnet-f8b8
        "ResNet for real-time tasks, mild denoising.",    // acnet-f8b8-hdn
        "ResNet for real-time tasks, without denoising.", // arnet-f8b16
        "ResNet for real-time tasks, mild denoising.",    // acnet-f8b16-hdn
        "ResNet for real-time tasks, without denoising.", // arnet-f8b32
        "ResNet for real-time tasks, mild denoising.",    // arnet-f8b32-hdn
        "ResNet for real-time tasks, without denoising.", // arnet-f8b64
        "ArtCNN-C4F16 from Artoriuz, lightweight option for real-time tasks. (v1.6.2) (https://github.com/Artoriuz/ArtCNN)", // artcnn-c4f16
        "ArtCNN-C4F16-DN from Artoriuz, trained to denoise and soften. (v1.6.2) (https://github.com/Artoriuz/ArtCNN)", // artcnn-c4f16-dn
        "ArtCNN-C4F16-DS from Artoriuz, trained to denoise and sharpen. (v1.6.2) (https://github.com/Artoriuz/ArtCNN)", // artcnn-c4f16-ds
        "ArtCNN-C4F32 from Artoriuz, real-time tasks if hardware allows. (v1.6.2) (https://github.com/Artoriuz/ArtCNN)", // artcnn-c4f32
        "ArtCNN-C4F32-DN from Artoriuz, trained to denoise and soften. (v1.6.2) (https://github.com/Artoriuz/ArtCNN)", // artcnn-c4f32-dn
        "ArtCNN-C4F32-DS from Artoriuz, trained to denoise and sharpen. (v1.6.2) (https://github.com/Artoriuz/ArtCNN)", // artcnn-c4f32-ds
        "FSRCNNX-8-0-4-1 from igv. (https://github.com/igv/FSRCNN-TensorFlow)",  // fsrcnnx-f8
        "FSRCNNX-Distort-Plus-8-0-4-1 from nessotrin. (https://github.com/nessotrin/FSRCNN-TensorFlow)", // fsrcnnx-f8-distort-plus
        "FSRCNNX-16-0-4-1 from igv. (https://github.com/igv/FSRCNN-TensorFlow)", // fsrcnnx-f16
        "FSRCNNX-Distort-Plus-16-0-4-1 from nessotrin. (https://github.com/nessotrin/FSRCNN-TensorFlow)", // fsrcnnx-f16-distort-plus
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
