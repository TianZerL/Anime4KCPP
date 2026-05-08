#ifndef AC_SPECS_HPP
#define AC_SPECS_HPP

namespace ac::specs
{
    struct Model
    {
        const char* name;
        const char* description;
        int parameterCount;
        const char* version = nullptr;
        const char* author = nullptr;
        const char* homepage = nullptr;
    };

    struct Processor
    {
        const char* name;
        const char* description;
    };

    constexpr Model ModelList[] = {
        {
            "acnet-legacy-gan",
            "Lightweight CNN, detail enhancement.",
            4784
        },
        {
            "acnet-legacy-hdn0",
            "Lightweight CNN, mild denoising.",
            4784
        },
        {
            "acnet-legacy-hdn1",
            "Lightweight CNN, moderate denoising.",
            4784
        },
        {
            "acnet-legacy-hdn2",
            "Lightweight CNN, heavy denoising.",
            4784
        },
        {
            "acnet-legacy-hdn3",
            "Lightweight CNN, extreme denoising.",
            4784
        },
        {
            "acnet-f8b4",
            "Lightweight CNN, without denoising.",
            2748
        },
        {
            "acnet-f8b4-hdn",
            "Lightweight CNN, mild denoising.",
            2748
        },
        {
            "acnet-f8b4-box",
            "Lightweight CNN, without denoising.",
            2748
        },
        {
            "acnet-f8b4-box-hdn",
            "Lightweight CNN, mild denoising.",
            2748
        },
        {
            "acnet-f8b8",
            "Lightweight CNN, without denoising.",
            5116
        },
        {
            "acnet-f8b8-hdn",
            "Lightweight CNN, mild denoising.",
            5116
        },
        {
            "acnet-f8b8-box",
            "Lightweight CNN, without denoising.",
            5116
        },
        {
            "acnet-f8b8-box-hdn",
            "Lightweight CNN, mild denoising.",
            5116
        },
        {
            "acnet-f8b18",
            "Lightweight CNN, without denoising.",
            11036
        },
        {
            "acnet-f8b18-hdn",
            "Lightweight CNN, mild denoising.",
            11036
        },
        {
            "acnet-f8b18-box",
            "Lightweight CNN, without denoising.",
            11036
        },
        {
            "acnet-f8b18-box-hdn",
            "Lightweight CNN, mild denoising.",
            11036
        },
        {
            "arnet-f8b8",
            "ResNet for real-time tasks, without denoising.",
            9860
        },
        {
            "arnet-f8b8-hdn",
            "ResNet for real-time tasks, mild denoising.",
            9860
        },
        {
            "arnet-f8b8-box",
            "ResNet for real-time tasks, without denoising.",
            9860
        },
        {
            "arnet-f8b8-box-hdn",
            "ResNet for real-time tasks, mild denoising.",
            9860
        },
        {
            "arnet-f8b16",
            "ResNet for real-time tasks, without denoising.",
            19268
        },
        {
            "arnet-f8b16-hdn",
            "ResNet for real-time tasks, mild denoising.",
            19268
        },
        {
            "arnet-f8b16-box",
            "ResNet for real-time tasks, without denoising.",
            19268
        },
        {
            "arnet-f8b16-box-hdn",
            "ResNet for real-time tasks, mild denoising.",
            19268
        },
        {
            "arnet-f8b32",
            "ResNet for real-time tasks, without denoising.",
            38084
        },
        {
            "arnet-f8b32-hdn",
            "ResNet for real-time tasks, mild denoising.",
            38084
        },
        {
            "arnet-f8b32-box",
            "ResNet for real-time tasks, without denoising.",
            38084
        },
        {
            "arnet-f8b32-box-hdn",
            "ResNet for real-time tasks, mild denoising.",
            38084
        },
        {
            "arnet-f8b64",
            "ResNet for real-time tasks, without denoising.",
            75716
        },
        {
            "arnet-f8b64-hdn",
            "ResNet for real-time tasks, mild denoising.",
            75716
        },
        {
            "arnet-f8b64-box",
            "ResNet for real-time tasks, without denoising.",
            75716
        },
        {
            "arnet-f8b64-box-hdn",
            "ResNet for real-time tasks, mild denoising.",
            75716
        },
        {
            "artcnn-c4f16",
            "ArtCNN-C4F16 from Artoriuz, lightweight option for real-time tasks.",
            12340,
            "v1.6.2",
            "Artoriuz",
            "https://github.com/Artoriuz/ArtCNN"
        },
        {
            "artcnn-c4f16-dn",
            "ArtCNN-C4F16-DN from Artoriuz, trained to denoise and soften.",
            12340,
            "v1.6.2",
            "Artoriuz",
            "https://github.com/Artoriuz/ArtCNN"
        },
        {
            "artcnn-c4f16-ds",
            "ArtCNN-C4F16-DS from Artoriuz, trained to denoise and sharpen.",
            12340,
            "v1.6.2",
            "Artoriuz",
            "https://github.com/Artoriuz/ArtCNN"
        },
        {
            "artcnn-c4f32",
            "ArtCNN-C4F32 from Artoriuz, real-time tasks if hardware allows.",
            47716,
            "v1.6.2",
            "Artoriuz",
            "https://github.com/Artoriuz/ArtCNN"
        },
        {
            "artcnn-c4f32-dn",
            "ArtCNN-C4F32-DN from Artoriuz, trained to denoise and soften.",
            47716,
            "v1.6.2",
            "Artoriuz",
            "https://github.com/Artoriuz/ArtCNN"
        },
        {
            "artcnn-c4f32-ds",
            "ArtCNN-C4F32-DS from Artoriuz, trained to denoise and sharpen.",
            47716,
            "v1.6.2",
            "Artoriuz",
            "https://github.com/Artoriuz/ArtCNN"
        },
        {
            "fsrcnnx-f8b4",
            "FSRCNNX-8-0-4-1 from igv.",
            2948,
            "v1.1",
            "igv",
            "https://github.com/igv/FSRCNN-TensorFlow"
        },
        {
            "fsrcnnx-f8b4-distort-plus",
            "FSRCNNX-Distort-Plus-8-0-4-1 from nessotrin.",
            2948,
            "v1.3_distort_plus",
            "nessotrin",
            "https://github.com/nessotrin/FSRCNN-TensorFlow"
        },
        {
            "fsrcnnx-f16b4",
            "FSRCNNX-16-0-4-1 from igv.",
            10628,
            "v1.1",
            "igv",
            "https://github.com/igv/FSRCNN-TensorFlow"
        },
        {
            "fsrcnnx-f16b4-distort-plus",
            "FSRCNNX-Distort-Plus-16-0-4-1 from nessotrin.",
            10628,
            "v1.3_distort_plus",
            "nessotrin",
            "https://github.com/nessotrin/FSRCNN-TensorFlow"
        }
    };

    constexpr Processor ProcessorList[] = {
        {
            "auto",
            "Auto-detect reasonable processor and device."
        },
        {
            "cpu",
            "General-purpose CPU processing with optional SIMD acceleration."
        },
#   ifdef AC_CORE_WITH_OPENCL
        {
            "opencl",
            "Cross-platform acceleration requiring OpenCL 1.2+ compliant devices."
        },
#   endif
#   ifdef AC_CORE_WITH_CUDA
        {
            "cuda",
            "NVIDIA GPU acceleration requiring Compute Capability 5.0+."
        },
#   endif
    };
}

#endif
