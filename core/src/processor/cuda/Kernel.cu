#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda_fp16.h>

#include "AC/Core/Image.hpp"

namespace ac::core::cuda
{
    struct BlockSize
    {
        static constexpr int x = 16;
        static constexpr int y = 16;
    };

    struct ResidualArg
    {
        const void* ptr;
        int w;
        int h;
        int c;
        int pitch;
        float scale;
    };

    __device__ static float identity(float v) noexcept
    {
        return v;
    }
    __device__ static float relu(float v) noexcept
    {
        return fmaxf(v, 0.0f);
    }
    __device__ static float lrelu(float v, float n) noexcept
    {
        return fmaxf(v, v * n);
    }

    class Identity
    {
    public:
        Identity() noexcept = default;
        __device__ float operator() (const float v) const noexcept { return identity(v); }
    };
    class ReLU
    {
    public:
        ReLU() noexcept = default;
        __device__ float operator() (const float v) const noexcept { return relu(v); }
    };
    class LReLU
    {
    public:
        LReLU(const float negativeSlope) noexcept : negativeSlope(negativeSlope) {}
        __device__ float operator() (const float v) const noexcept { return lrelu(v, negativeSlope); }

    private:
        const float negativeSlope;
    };

    template<typename T, int N>
    __device__ static inline const T* getReadPtrFromShared(const T(* const sharedPtr)[N], const int c, const int tx, const int ty)
    {
        return sharedPtr[1 + ty] + (1 + tx) * c;
    }
    template<typename T>
    __device__ static inline const T* getReadPtr(const void* const ptr, const int w, const int h, const int c, const int pitch, const int x, const int y)
    {
        auto posX = min(max(x, 0), w - 1);
        auto posY = min(max(y, 0), h - 1);
        return reinterpret_cast<const T*>(static_cast<const ::cuda::std::uint8_t*>(ptr) + pitch * posY) + posX * c;
    }
    template<typename T>
    __device__ static inline T* getWritePtr(void* const ptr, const int w, const int h, const int c, const int pitch, const int x, const int y)
    {
        auto posX = min(max(x, 0), w - 1);
        auto posY = min(max(y, 0), h - 1);
        return reinterpret_cast<T*>(static_cast<::cuda::std::uint8_t*>(ptr) + pitch * posY) + posX * c;
    }

    template<typename Float, ::cuda::std::enable_if_t<::cuda::std::is_floating_point_v<Float>, bool> = true>
    __device__ static inline float toFloat(const Float v) noexcept
    {
        return static_cast<float>(v);
    }
    template<typename Unsigned, ::cuda::std::enable_if_t<::cuda::std::is_unsigned_v<Unsigned>, bool> = true>
    __device__ static inline float toFloat(const Unsigned v) noexcept
    {
        return static_cast<float>(v) / static_cast<float>(::cuda::std::numeric_limits<Unsigned>::max());
    }
    __device__ static inline float toFloat(const half v) noexcept
    {
        return __half2float(v);
    }

    template<typename T>
    __device__ static inline auto toHalf(const T v) noexcept
    {
        if constexpr (::cuda::std::is_same_v<T, half>)
            return v;
        else
            return __float2half(toFloat(v));
    }

    template<typename Float, ::cuda::std::enable_if_t<::cuda::std::is_floating_point_v<Float>, bool> = true>
    __device__ static inline Float fromFloat(const float v) noexcept
    {
        return __saturatef(v);
    }
    template<typename Unsigned, ::cuda::std::enable_if_t<::cuda::std::is_unsigned_v<Unsigned>, bool> = true>
    __device__ static inline Unsigned fromFloat(const float v) noexcept
    {
        return static_cast<Unsigned>(fromFloat<float>(v) * ::cuda::std::numeric_limits<Unsigned>::max() + 0.5f);
    }

    template<bool check, typename T>
    struct RestrictPointer
    {
        using Type = T* __restrict__;
    };

    template<typename T>
    struct RestrictPointer<false, T>
    {
        using Type = T*;
    };

    namespace kernel
    {
        template <typename IN, int cin, int cout, typename ActiveFunc, typename... ResidualArgs>
        __global__ void conv3x3_cuda(
            const void* const __restrict__ sptr,
            const int srcW, const int srcH, const int srcC, const int spitch,
            const typename RestrictPointer<!sizeof...(ResidualArgs), void>::Type dptr,
            const int dstW, const int dstH, const int dstC, const int dpitch,
            const float* const __restrict__ kernels,
            const float* const __restrict__ biases,
            ActiveFunc activeFunc, ResidualArgs ...residualArg)
        {
            [[maybe_unused]] const ::cuda::std::array<ResidualArg, sizeof...(ResidualArgs)> residualArgs{ residualArg... };

            auto bx = blockIdx.x * BlockSize::x;
            auto by = blockIdx.y * BlockSize::y;
            auto x = bx + threadIdx.x;
            auto y = by + threadIdx.y;
            auto tid = threadIdx.y * BlockSize::x + threadIdx.x;

            constexpr auto threads = BlockSize::x * BlockSize::y;
            constexpr auto knum = cout * 9 * cin;
            constexpr auto bnum = cout;
            constexpr auto rsizeY = BlockSize::y + 2;
            constexpr auto rsizeX = (BlockSize::x + 2) * cin;

            __shared__ float kptr[knum];
            if constexpr (threads < knum)
            {
                constexpr auto line = knum / threads;
                constexpr auto remain = knum % threads;
                for (int i = 0; i < line; i++)
                {
                    auto idx = tid + i * threads;
                    kptr[idx] = kernels[idx];
                }
                if (tid < remain)
                {
                    auto idx = tid + line * threads;
                    kptr[idx] = kernels[idx];
                }
            }
            else if (tid < knum) kptr[tid] = kernels[tid];

            __shared__ float bptr[bnum];
            if constexpr (threads < bnum)
            {
                constexpr auto line = bnum / threads;
                constexpr auto remain = bnum % threads;
                for (int i = 0; i < line; i++)
                {
                    auto idx = tid + i * threads;
                    bptr[idx] = biases[idx];
                }
                if (tid < remain)
                {
                    auto idx = tid + line * threads;
                    bptr[idx] = biases[idx];
                }
            }
            else if (tid < bnum) bptr[tid] = biases[tid];

            __shared__ IN iptr[rsizeY][rsizeX];
            for (int j = 0; j < rsizeY; j++)
            {
                if constexpr (threads < rsizeX)
                {
                    constexpr auto line = rsizeX / threads;
                    constexpr auto remain = rsizeX % threads;
                    for (int i = 0; i < line; i++)
                    {
                        auto idx = tid + i * threads;
                        iptr[j][idx] = getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, bx - 1 + idx / cin, by - 1 + j)[idx % cin];
                    }
                    if (tid < remain)
                    {
                        auto idx = tid + line * threads;
                        iptr[j][idx] = getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, bx - 1 + idx / cin, by - 1 + j)[idx % cin];
                    }
                }
                else if (tid < rsizeX) iptr[j][tid] = getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, bx - 1 + tid / cin, by - 1 + j)[tid % cin];
            }
            __syncthreads();

            if (x >= srcW || y >= srcH) return;

            auto out = getWritePtr<half>(dptr, dstW, dstH, dstC, dpitch, x, y);

            const IN* r[] = {
                getReadPtrFromShared(iptr, cin, threadIdx.x - 1, threadIdx.y - 1),
                getReadPtrFromShared(iptr, cin, threadIdx.x    , threadIdx.y - 1),
                getReadPtrFromShared(iptr, cin, threadIdx.x + 1, threadIdx.y - 1),
                getReadPtrFromShared(iptr, cin, threadIdx.x - 1, threadIdx.y    ),
                getReadPtrFromShared(iptr, cin, threadIdx.x    , threadIdx.y    ),
                getReadPtrFromShared(iptr, cin, threadIdx.x + 1, threadIdx.y    ),
                getReadPtrFromShared(iptr, cin, threadIdx.x - 1, threadIdx.y + 1),
                getReadPtrFromShared(iptr, cin, threadIdx.x    , threadIdx.y + 1),
                getReadPtrFromShared(iptr, cin, threadIdx.x + 1, threadIdx.y + 1)
            };

            for (int n = 0; n < cout; n++)
            {
                const float* k[] = {
                    kptr + n * cin * 9 + cin * 0,
                    kptr + n * cin * 9 + cin * 1,
                    kptr + n * cin * 9 + cin * 2,
                    kptr + n * cin * 9 + cin * 3,
                    kptr + n * cin * 9 + cin * 4,
                    kptr + n * cin * 9 + cin * 5,
                    kptr + n * cin * 9 + cin * 6,
                    kptr + n * cin * 9 + cin * 7,
                    kptr + n * cin * 9 + cin * 8
                };

                float sum = bptr[n];

                for (int idx = 0; idx < 9; idx++)
                    for (int c = 0; c < cin; c++)
                        sum += toFloat(r[idx][c]) * k[idx][c];

                if constexpr (sizeof...(ResidualArgs))
                    for (int idx = 0; idx < sizeof...(ResidualArgs); idx++)
                        sum = sum * residualArgs[idx].scale + toFloat(getReadPtr<half>(residualArgs[idx].ptr, residualArgs[idx].w, residualArgs[idx].h, residualArgs[idx].c, residualArgs[idx].pitch, x, y)[n]);

                out[n] = toHalf(activeFunc(sum));
            }
        }

        template<typename IN, typename OUT, int cin, int cout>
        __global__ void deconv2x2_cuda(
            const void* const __restrict__ sptr,
            const int srcW, const int srcH, const int srcC, const int spitch,
            void* const __restrict__ dptr,
            const int dstW, const int dstH, const int dstC, const int dpitch,
            const float* const __restrict__ kernels)
        {
            auto x = blockIdx.x * BlockSize::x + threadIdx.x;
            auto y = blockIdx.y * BlockSize::y + threadIdx.y;

            if (x >= dstW || y >= dstH) return;

            auto in = getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, x / 2, y / 2);
            auto out = getWritePtr<OUT>(dptr, dstW, dstH, dstC, dpitch, x, y);

            auto index = ((y & 1) << 1) + (x & 1);

            for (int n = 0; n < cout; n++)
            {
                auto k = kernels + n * cin * 4 + cin * index;
                float sum = 0.0f;
                for (int c = 0; c < cin; c++) sum += toFloat(in[c]) * k[c];
                out[n] = fromFloat<OUT>(sum);
            }
        }

        template <typename IN, typename OUT>
        __global__ void conv3x3_8to4_identity_pixelshuffle_4to1_cuda(
            const void* const __restrict__ sptr,
            const int srcW, const int srcH, const int srcC, const int spitch,
            void* const __restrict__ dptr,
            const int dstW, const int dstH, const int dstC, const int dpitch,
            const float* const __restrict__ kernels,
            const float* const __restrict__ biases)
        {
            auto bx = blockIdx.x * BlockSize::x;
            auto by = blockIdx.y * BlockSize::y;
            auto x = bx + threadIdx.x;
            auto y = by + threadIdx.y;
            auto tid = threadIdx.y * BlockSize::x + threadIdx.x;

            constexpr auto cin = 8;
            constexpr auto upscale = 2;
            constexpr auto threads = BlockSize::x * BlockSize::y;
            constexpr auto knum = 4 * 9 * 8;
            constexpr auto bnum = 4;
            constexpr auto rsizeY = BlockSize::y + 2;
            constexpr auto rsizeX = (BlockSize::x + 2) * cin;

            __shared__ float kptr[knum];
            if constexpr (threads < knum)
            {
                constexpr auto line = knum / threads;
                constexpr auto remain = knum % threads;
                for (int i = 0; i < line; i++)
                {
                    auto idx = tid + i * threads;
                    kptr[idx] = kernels[idx];
                }
                if (tid < remain)
                {
                    auto idx = tid + line * threads;
                    kptr[idx] = kernels[idx];
                }
            }
            else if (tid < knum) kptr[tid] = kernels[tid];

            __shared__ float bptr[bnum];
            if constexpr (threads < bnum)
            {
                constexpr auto line = bnum / threads;
                constexpr auto remain = bnum % threads;
                for (int i = 0; i < line; i++)
                {
                    auto idx = tid + i * threads;
                    bptr[idx] = biases[idx];
                }
                if (tid < remain)
                {
                    auto idx = tid + line * threads;
                    bptr[idx] = biases[idx];
                }
            }
            else if (tid < bnum) bptr[tid] = biases[tid];

            __shared__ IN iptr[rsizeY][rsizeX];
            for (int j = 0; j < rsizeY; j++)
            {
                if constexpr (threads < rsizeX)
                {
                    constexpr auto line = rsizeX / threads;
                    constexpr auto remain = rsizeX % threads;
                    for (int i = 0; i < line; i++)
                    {
                        auto idx = tid + i * threads;
                        iptr[j][idx] = getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, bx - 1 + idx / cin, by - 1 + j)[idx % cin];
                    }
                    if (tid < remain)
                    {
                        auto idx = tid + line * threads;
                        iptr[j][idx] = getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, bx - 1 + idx / cin, by - 1 + j)[idx % cin];
                    }
                }
                else if (tid < rsizeX) iptr[j][tid] = getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, bx - 1 + tid / cin, by - 1 + j)[tid % cin];
            }
            __syncthreads();

            if (x >= srcW || y >= srcH) return;

            const IN* r[] = {
                getReadPtrFromShared(iptr, cin, threadIdx.x - 1, threadIdx.y - 1),
                getReadPtrFromShared(iptr, cin, threadIdx.x    , threadIdx.y - 1),
                getReadPtrFromShared(iptr, cin, threadIdx.x + 1, threadIdx.y - 1),
                getReadPtrFromShared(iptr, cin, threadIdx.x - 1, threadIdx.y    ),
                getReadPtrFromShared(iptr, cin, threadIdx.x    , threadIdx.y    ),
                getReadPtrFromShared(iptr, cin, threadIdx.x + 1, threadIdx.y    ),
                getReadPtrFromShared(iptr, cin, threadIdx.x - 1, threadIdx.y + 1),
                getReadPtrFromShared(iptr, cin, threadIdx.x    , threadIdx.y + 1),
                getReadPtrFromShared(iptr, cin, threadIdx.x + 1, threadIdx.y + 1)
            };

            auto dstX = x * upscale;
            auto dstY = y * upscale;

            for (int n = 0; n < 4; n++)
            {
                const float* k[] = {
                    kptr + n * cin * 9 + cin * 0,
                    kptr + n * cin * 9 + cin * 1,
                    kptr + n * cin * 9 + cin * 2,
                    kptr + n * cin * 9 + cin * 3,
                    kptr + n * cin * 9 + cin * 4,
                    kptr + n * cin * 9 + cin * 5,
                    kptr + n * cin * 9 + cin * 6,
                    kptr + n * cin * 9 + cin * 7,
                    kptr + n * cin * 9 + cin * 8
                };

                float sum = bptr[n];

                for (int idx = 0; idx < 9; idx++)
                    for (int c = 0; c < cin; c++)
                        sum += toFloat(r[idx][c]) * k[idx][c];

                *getWritePtr<OUT>(dptr, dstW, dstH, dstC, dpitch, dstX + (n & 1), dstY + (n >> 1)) = fromFloat<OUT>(sum);
            }
        }
    }

    void conv3x3_1to8_relu_cuda(
        const void* sptr,
        int srcW, int srcH, int srcC, int spitch,
        void* dptr,
        int dstW, int dstH, int dstC, int dpitch,
        const float* kernels,
        const float* biases,
        Image::ElementType stype,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        switch (stype)
        {
        case Image::UInt8:
            kernel::conv3x3_cuda<::cuda::std::uint8_t, 1, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, ReLU());
            break;
        case Image::UInt16:
            kernel::conv3x3_cuda<::cuda::std::uint16_t, 1, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, ReLU());
            break;
        case Image::Float32:
            kernel::conv3x3_cuda<float, 1, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, ReLU());
            break;
        }
    }
    void conv3x3_8to8_relu_cuda(
        const void* sptr,
        int srcW, int srcH, int srcC, int spitch,
        void* dptr,
        int dstW, int dstH, int dstC, int dpitch,
        const float* kernels,
        const float* biases,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        kernel::conv3x3_cuda<half, 8, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, ReLU());
    }
    void deconv2x2_8to1_cuda(
        const void* sptr,
        int srcW, int srcH, int srcC, int spitch,
        void* dptr,
        int dstW, int dstH, int dstC, int dpitch,
        const float* kernels,
        Image::ElementType dtype,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y };
        switch (dtype)
        {
        case Image::UInt8:
            return kernel::deconv2x2_cuda<half, ::cuda::std::uint8_t, 8, 1> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels);
        case Image::UInt16:
            return kernel::deconv2x2_cuda<half, ::cuda::std::uint16_t, 8, 1> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels);
        case Image::Float32:
            return kernel::deconv2x2_cuda<half, float, 8, 1> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels);
        }
    }

    void conv3x3_1to8_identity_cuda(
        const void* sptr,
        int srcW, int srcH, int srcC, int spitch,
        void* dptr,
        int dstW, int dstH, int dstC, int dpitch,
        const float* kernels,
        const float* biases,
        Image::ElementType stype,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        switch (stype)
        {
        case Image::UInt8:
            kernel::conv3x3_cuda<::cuda::std::uint8_t, 1, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity());
            break;
        case Image::UInt16:
            kernel::conv3x3_cuda<::cuda::std::uint16_t, 1, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity());
            break;
        case Image::Float32:
            kernel::conv3x3_cuda<float, 1, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_8to8_lrelu_cuda(
        const void* sptr,
        int srcW, int srcH, int srcC, int spitch,
        void* dptr,
        int dstW, int dstH, int dstC, int dpitch,
        const float* kernels,
        const float* biases,
        const float negativeSlope,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        kernel::conv3x3_cuda<half, 8, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, LReLU(negativeSlope));
    }
    void conv3x3_8to8_residual_identity_cuda(
        const void* sptr,
        int srcW, int srcH, int srcC, int spitch,
        void* dptr,
        int dstW, int dstH, int dstC, int dpitch,
        const float* kernels,
        const float* biases,
        void* iptr,
        int idW, int idH, int idC, int ipitch,
        const float scale,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        kernel::conv3x3_cuda<half, 8, 8> <<< grid, block, 0, stream >>> (
            sptr, srcW, srcH, srcC, spitch,
            dptr, dstW, dstH, dstC, dpitch,
            kernels, biases,
            Identity(),
            ResidualArg{ iptr, idW, idH, idC, ipitch, scale });
    }
    void conv3x3_8to8_residual_identity_cuda(
        const void* sptr,
        int srcW, int srcH, int srcC, int spitch,
        void* dptr,
        int dstW, int dstH, int dstC, int dpitch,
        const float* kernels,
        const float* biases,
        void* iptr,
        int idW, int idH, int idC, int ipitch,
        const float scale,
        void* fptr,
        int featW, int featH, int featC, int fpitch,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        kernel::conv3x3_cuda<half, 8, 8> <<< grid, block, 0, stream >>> (
            sptr, srcW, srcH, srcC, spitch,
            dptr, dstW, dstH, dstC, dpitch,
            kernels, biases,
            Identity(),
            ResidualArg{ iptr, idW, idH, idC, ipitch, scale },
            ResidualArg{ fptr, featW, featH, featC, fpitch, 1.0f });
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_cuda(
        const void* sptr,
        int srcW, int srcH, int srcC, int spitch,
        void* dptr,
        int dstW, int dstH, int dstC, int dpitch,
        const float* kernels,
        const float* biases,
        Image::ElementType dtype,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        switch (dtype)
        {
        case Image::UInt8:
            return kernel::conv3x3_8to4_identity_pixelshuffle_4to1_cuda<half, ::cuda::std::uint8_t> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases);
        case Image::UInt16:
            return kernel::conv3x3_8to4_identity_pixelshuffle_4to1_cuda<half, ::cuda::std::uint16_t> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases);
        case Image::Float32:
            return kernel::conv3x3_8to4_identity_pixelshuffle_4to1_cuda<half, float> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases);
        }
    }
}
