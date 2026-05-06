#ifndef AC_CORE_INTERNAL_PROCESSOR_CUDA_COMMON_HPP
#define AC_CORE_INTERNAL_PROCESSOR_CUDA_COMMON_HPP

#include <cstdint>

#include <cuda_runtime.h>

namespace ac::core::cuda
{
    class DeviceImage
    {
    public:
        /// Same rule as ac::core::Image.
        using ElementType = int;
        static constexpr int UInt8 = 0 << 8 | 1;
        static constexpr int UInt16 = 0 << 8 | 2;
        static constexpr int Float16 = 2 << 8 | 2;
        static constexpr int Float32 = 2 << 8 | 4;

    public:
        int width() const noexcept { return w; }
        int height() const noexcept { return h; }
        int channels() const noexcept { return c; }
        int stride() const noexcept { return pitch; }
        int size() const noexcept { return h * pitch; }
        int elementSize() const noexcept { return elementType & 0xff; }
        int pixelSize() const noexcept { return c * elementSize(); }
        ElementType type() const noexcept { return elementType; }
        std::uint8_t* data() const noexcept { return static_cast<std::uint8_t*>(pixels); }
        std::uint8_t* line(const int y) const noexcept { return data() + y * pitch; }
        std::uint8_t* pixel(const int x, const int y) const noexcept { return line(y) + x * pixelSize(); }
        void* ptr() const noexcept { return pixels; }
        void* ptr(const int y) const noexcept { return line(y); }
        void* ptr(const int x, const int y) const noexcept { return pixel(x, y); }
        bool empty() const noexcept { return pixels == nullptr; }

    protected:
        DeviceImage() noexcept = default;

    protected:
        int w = 0;
        int h = 0;
        int c = 0;
        ElementType elementType = Float32;
        int pitch = 0;
        void* pixels = nullptr;
    };

    template<int outType = DeviceImage::Float16>
    void conv3x3_1to8_relu_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept;
    template<int inoutType = DeviceImage::Float16>
    void conv3x3_8to8_relu_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept;
    template<int outType = DeviceImage::Float16>
    void conv3x3_8to8_relu_deconv2x2_8to1_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels1, const float* biases1,
        const float* kernels2,
        cudaStream_t stream
    ) noexcept;
    template<int outType = DeviceImage::Float16>
    void conv3x3_1to8_prelu_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases, const float* alphas,
        cudaStream_t stream
    ) noexcept;
    template<int outType = DeviceImage::Float16>
    void conv3x3_1to8_identity_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept;
    template<int inoutType = DeviceImage::Float16>
    void conv3x3_8to8_prelu_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases, const float* alphas,
        cudaStream_t stream
    ) noexcept;
    template<int inoutType = DeviceImage::Float16>
    void conv3x3_8to8_identity_residual_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        const DeviceImage& idt, const float scale,
        cudaStream_t stream
    ) noexcept;
    template<int inoutType = DeviceImage::Float16>
    void conv3x3_8to8_identity_residual_conv1x1_8to8_prelu_add_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels1, const float* biases1,
        const DeviceImage& idt, const float scale,
        const float* kernels2, const float* biases2, const float* alphas2,
        const DeviceImage& feat,
        cudaStream_t stream
    ) noexcept;
    template<int inType = DeviceImage::Float16>
    void conv3x3_8to4_identity_pixelshuffle_4to1_add_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        const DeviceImage& idt,
        cudaStream_t stream
    ) noexcept;
    template<int outType = DeviceImage::Float16>
    void conv3x3_1to16_identity_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept;
    template<int inoutType = DeviceImage::Float16>
    void conv3x3_16to16_relu_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept;
    template<int inoutType = DeviceImage::Float16>
    void conv3x3_16to16_identity_add_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        const DeviceImage& feat,
        cudaStream_t stream
    ) noexcept;
    template<int inType = DeviceImage::Float16>
    void conv3x3_16to4_identity_pixelshuffle_4to1_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept;
    template<int outType = DeviceImage::Float16>
    void conv3x3_1to32_identity_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept;
    template<int inoutType = DeviceImage::Float16>
    void conv3x3_32to32_relu_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept;
    template<int inoutType = DeviceImage::Float16>
    void conv3x3_32to32_identity_add_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        const DeviceImage& feat,
        cudaStream_t stream
    ) noexcept;
    template<int inType = DeviceImage::Float16>
    void conv3x3_32to4_identity_pixelshuffle_4to1_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept;
    template<int outType = DeviceImage::Float16>
    void conv5x5_1to8_identity_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept;
    template<int inoutType = DeviceImage::Float16>
    void conv3x3_8to8_prelu_conv1x1_8to8_add_prelu_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const DeviceImage& feat,
        cudaStream_t stream
    ) noexcept;
    template<int inType = DeviceImage::Float16>
    void conv3x3_8to4_identity_pixelshuffle_4to1_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept;
    template<int outType = DeviceImage::Float16>
    void conv5x5_1to16_identity_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept;
    template<int inoutType = DeviceImage::Float16>
    void conv3x3_16to16_prelu_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases, const float* alphas,
        cudaStream_t stream
    ) noexcept;
    template<int inoutType = DeviceImage::Float16>
    void conv3x3_16to16_prelu_conv1x1_16to16_add_prelu_cuda(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const DeviceImage& feat,
        cudaStream_t stream
    ) noexcept;
}

#endif
