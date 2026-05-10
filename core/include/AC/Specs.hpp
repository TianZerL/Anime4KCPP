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
            "Lightweight CNN, moderate denoising.",
            4784
        },
        {
            "acnet-legacy-hdn1",
            "Lightweight CNN, strong denoising.",
            4784
        },
        {
            "acnet-legacy-hdn2",
            "Lightweight CNN, aggressive denoising.",
            4784
        },
        {
            "acnet-legacy-hdn3",
            "Lightweight CNN, extreme denoising.",
            4784
        },
        {
            "acnet-f8b4",
            "Lightweight VGG-style network, trained to be neutral.",
            2748
        },
        {
            "acnet-f8b4-hdn",
            "Lightweight VGG-style network, trained to mildly denoise.",
            2748
        },
        {
            "acnet-f8b4-box",
            "Lightweight VGG-style network, trained to be neutral through box degradation which is better for line restoration, but it may appear slightly blurry.",
            2748
        },
        {
            "acnet-f8b4-box-hdn",
            "Lightweight VGG-style network, trained to mildly denoise based on the box variant.",
            2748
        },
        {
            "acnet-f8b8",
            "Lightweight VGG-style network, trained to be neutral.",
            5116
        },
        {
            "acnet-f8b8-hdn",
            "Lightweight VGG-style network, trained to mildly denoise.",
            5116
        },
        {
            "acnet-f8b8-box",
            "Lightweight VGG-style network, trained to be neutral through box degradation which is better for line restoration, but it may appear slightly blurry.",
            5116
        },
        {
            "acnet-f8b8-box-hdn",
            "Lightweight VGG-style network, trained to mildly denoise based on the box variant.",
            5116
        },
        {
            "acnet-f8b18",
            "Lightweight VGG-style network, trained to be neutral.",
            11036
        },
        {
            "acnet-f8b18-hdn",
            "Lightweight VGG-style network, trained to mildly denoise.",
            11036
        },
        {
            "acnet-f8b18-box",
            "Lightweight VGG-style network, trained to be neutral through box degradation which is better for line restoration, but it may appear slightly blurry.",
            11036
        },
        {
            "acnet-f8b18-box-hdn",
            "Lightweight VGG-style network, trained to mildly denoise based on the box variant.",
            11036
        },
        {
            "arnet-f8b8",
            "Lightweight ResNet-style network, trained to be neutral.",
            9860
        },
        {
            "arnet-f8b8-hdn",
            "Lightweight ResNet-style network, trained to mildly denoise.",
            9860
        },
        {
            "arnet-f8b8-box",
            "Lightweight ResNet-style network, trained to be neutral through box degradation which is better for line restoration, but it may appear slightly blurry.",
            9860
        },
        {
            "arnet-f8b8-box-hdn",
            "Lightweight ResNet-style network, trained to mildly denoise based on the box variant.",
            9860
        },
        {
            "arnet-f8b16",
            "Lightweight ResNet-style network, trained to be neutral.",
            19268
        },
        {
            "arnet-f8b16-hdn",
            "Lightweight ResNet-style network, trained to mildly denoise.",
            19268
        },
        {
            "arnet-f8b16-box",
            "Lightweight ResNet-style network, trained to be neutral through box degradation which is better for line restoration, but it may appear slightly blurry.",
            19268
        },
        {
            "arnet-f8b16-box-hdn",
            "Lightweight ResNet-style network, trained to mildly denoise based on the box variant.",
            19268
        },
        {
            "arnet-f8b32",
            "Lightweight ResNet-style network, trained to be neutral.",
            38084
        },
        {
            "arnet-f8b32-hdn",
            "Lightweight ResNet-style network, trained to mildly denoise.",
            38084
        },
        {
            "arnet-f8b32-box",
            "Lightweight ResNet-style network, trained to be neutral through box degradation which is better for line restoration, but it may appear slightly blurry.",
            38084
        },
        {
            "arnet-f8b32-box-hdn",
            "Lightweight ResNet-style network, trained to mildly denoise based on the box variant.",
            38084
        },
        {
            "arnet-f8b64",
            "Lightweight ResNet-style network, trained to be neutral.",
            75716
        },
        {
            "arnet-f8b64-hdn",
            "Lightweight ResNet-style network, trained to mildly denoise.",
            75716
        },
        {
            "arnet-f8b64-box",
            "Lightweight ResNet-style network, trained to be neutral through box degradation which is better for line restoration, but it may appear slightly blurry.",
            75716
        },
        {
            "arnet-f8b64-box-hdn",
            "Lightweight ResNet-style network, trained to mildly denoise based on the box variant.",
            75716
        },
        {
            "artcnn-c4f16",
            "Artoriuz's ArtCNN_C4F16 (integrated as-is), trained to be neutral.",
            12340,
            "v1.6.2",
            "Artoriuz",
            "https://github.com/Artoriuz/ArtCNN"
        },
        {
            "artcnn-c4f16-dn",
            "Artoriuz's ArtCNN_C4F16_DN (integrated as-is), trained to denoise and soften.",
            12340,
            "v1.6.2",
            "Artoriuz",
            "https://github.com/Artoriuz/ArtCNN"
        },
        {
            "artcnn-c4f16-ds",
            "Artoriuz's ArtCNN_C4F16_DS (integrated as-is), trained to denoise and sharpen.",
            12340,
            "v1.6.2",
            "Artoriuz",
            "https://github.com/Artoriuz/ArtCNN"
        },
        {
            "artcnn-c4f32",
            "Artoriuz's ArtCNN_C4F32 (integrated as-is), trained to be neutral.",
            47716,
            "v1.6.2",
            "Artoriuz",
            "https://github.com/Artoriuz/ArtCNN"
        },
        {
            "artcnn-c4f32-dn",
            "Artoriuz's ArtCNN_C4F32_DN (integrated as-is), trained to denoise and soften.",
            47716,
            "v1.6.2",
            "Artoriuz",
            "https://github.com/Artoriuz/ArtCNN"
        },
        {
            "artcnn-c4f32-ds",
            "Artoriuz's ArtCNN_C4F32_DS (integrated as-is), trained to denoise and sharpen.",
            47716,
            "v1.6.2",
            "Artoriuz",
            "https://github.com/Artoriuz/ArtCNN"
        },
        {
            "fsrcnnx-f8b4",
            "igv's FSRCNNX_x2_8-0-4-1 (integrated as-is), trained to slightly denoise.",
            2948,
            "v1.1",
            "igv",
            "https://github.com/igv/FSRCNN-TensorFlow"
        },
        {
            "fsrcnnx-f8b4-distort-plus",
            "nessotrin's FSRCNNX-x2-2-8-0-4-1.v1.fastv2 (integrated as-is), trained to strongly denoise.",
            2948,
            "v1.3_distort_plus",
            "nessotrin",
            "https://github.com/nessotrin/FSRCNN-TensorFlow"
        },
        {
            "fsrcnnx-f16b4",
            "igv's FSRCNNX_x2_16-0-4-1 (integrated as-is), trained to slightly denoise.",
            10628,
            "v1.1",
            "igv",
            "https://github.com/igv/FSRCNN-TensorFlow"
        },
        {
            "fsrcnnx-f16b4-distort-plus",
            "nessotrin's FSRCNNX-x2-2-16-0-4-1.v1.fastv2 (integrated as-is), trained to strongly denoise.",
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
