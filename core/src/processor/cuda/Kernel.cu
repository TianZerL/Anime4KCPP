#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda_fp16.h>

#include "AC/Core/Internal/Processor/CUDA/Common.hpp"

namespace ac::core::cuda
{
    struct DataType
    {
        using UInt8 = ::cuda::std::uint8_t;
        using UInt16 = ::cuda::std::uint16_t;
        using Float16 = half;
        using Float32 = float;
    };

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

    class Identity
    {
    public:
        Identity() noexcept = default;
        __device__ float operator() (const float v, const int /*c*/) const noexcept { return v; }
    };
    class ReLU
    {
    public:
        ReLU() noexcept = default;
        __device__ float operator() (const float v, const int /*c*/) const noexcept { return fmaxf(v, 0.0f); }
    };
    class LReLU
    {
    public:
        LReLU(const float negativeSlope) noexcept : negativeSlope(negativeSlope) {}
        __device__ float operator() (const float v, const int /*c*/) const noexcept { return fmaxf(v, v * negativeSlope); }

    private:
        const float negativeSlope;
    };
    class PReLU
    {
    public:
        PReLU(const float* __restrict__ alphas) noexcept : alphas(alphas) {}
        __device__ float operator() (const float v, const int c) const noexcept { return fmaxf(v, 0.0f) + alphas[c] * fminf(v, 0.0f); }

    private:
        const float* __restrict__ alphas;
    };

    template<int padx, int pady, typename T, int N>
    __device__ static inline const T* getReadPtrFromShared(const T(* const sharedPtr)[N], const int c, const int tx, const int ty)
    {
        return sharedPtr[pady + ty] + (padx + tx) * c;
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
    __device__ static inline Float saturate(const Float v) noexcept
    {
        if constexpr (::cuda::std::is_same_v<Float, float>)
            return __saturatef(v);
        else if constexpr (::cuda::std::is_same_v<Float, half>)
            return __hmin(__hmax(v, 0.0f), 1.0f);
        else
            return min(max(v, 0.0f), 1.0f);
    }

    template<typename To, typename From>
    __device__ static inline To cast(const From v) noexcept
    {
        if constexpr (::cuda::std::is_same_v<To, From>)
            return v;
        else if constexpr (::cuda::std::is_unsigned_v<From> && (::cuda::std::is_floating_point_v<To> || ::cuda::std::is_same_v<To, half>))
            return static_cast<To>(v) / static_cast<To>(::cuda::std::numeric_limits<From>::max());
        else if constexpr ((::cuda::std::is_floating_point_v<From> || ::cuda::std::is_same_v<From, half>) && ::cuda::std::is_unsigned_v<To>)
            return static_cast<To>(v * static_cast<From>(::cuda::std::numeric_limits<To>::max()) + static_cast<From>(0.5f));
        else
            return static_cast<To>(v);
    }

    template <typename T, int cin, int padx, int pady, int limit = 8192>
    __device__ static inline auto imageBlockToSharedAdaptive(const void* const __restrict__ data, const int weight, const int height, const int channels, const int pitch) noexcept
    {
        constexpr auto imageBlockSizeY = BlockSize::y + pady * 2;
        constexpr auto imageBlockSizeX = (BlockSize::x + padx * 2) * cin;

        if constexpr (imageBlockSizeY * imageBlockSizeX * sizeof(T) <= limit)
        {
            auto bx = blockIdx.x * BlockSize::x;
            auto by = blockIdx.y * BlockSize::y;
            auto tid = threadIdx.y * BlockSize::x + threadIdx.x;

            constexpr auto threads = BlockSize::x * BlockSize::y;

            __shared__ T sptr[imageBlockSizeY][imageBlockSizeX];
            for (int j = 0; j < imageBlockSizeY; j++)
            {
                if constexpr (threads < imageBlockSizeX)
                {
                    constexpr auto line = imageBlockSizeX / threads;
                    constexpr auto remain = imageBlockSizeX % threads;
                    for (int i = 0; i < line; i++)
                    {
                        auto idx = tid + i * threads;
                        sptr[j][idx] = getReadPtr<T>(data, weight, height, channels, pitch, bx - padx + idx / cin, by - pady + j)[idx % cin];
                    }
                    if (remain > 0 && tid < remain)
                    {
                        auto idx = tid + line * threads;
                        sptr[j][idx] = getReadPtr<T>(data, weight, height, channels, pitch, bx - padx + idx / cin, by - pady + j)[idx % cin];
                    }
                }
                else if (tid < imageBlockSizeX) sptr[j][tid] = getReadPtr<T>(data, weight, height, channels, pitch, bx - padx + tid / cin, by - pady + j)[tid % cin];
            }

            return sptr;
        }
        else return data;
    }

    template <typename IN, int kw, int kh, int cin, int padx, int pady, typename T>
    __device__ static inline auto loadImageBlockAdaptive(const IN** rptr, const int x, const int y, const T data, const int weight, const int height, const int channels, const int pitch) noexcept
    {
        constexpr auto hkw = kw / 2;
        constexpr auto hkh = kh / 2;
        for (int ypos = -hkh; ypos <= hkh; ypos++)
            for (int xpos = -hkw; xpos <= hkw; xpos++)
                if constexpr (::cuda::std::is_same_v<T, const void*>)
                    rptr[(ypos + hkh) * kw + xpos + hkw] = getReadPtr<IN>(data, weight, height, channels, pitch, x + xpos, y + ypos);
                else
                    rptr[(ypos + hkh) * kw + xpos + hkw] = getReadPtrFromShared<padx, pady>(data, cin, threadIdx.x + xpos, threadIdx.y + ypos);
    }

    template <int size, typename T>
    __device__ static inline void copyToShared(T* const __restrict__ sptr, const T* const __restrict__ gptr) noexcept
    {
        constexpr auto threads = BlockSize::x * BlockSize::y;

        auto tid = threadIdx.y * BlockSize::x + threadIdx.x;

        if constexpr (threads < size)
        {
            constexpr auto line = size / threads;
            constexpr auto remain = size % threads;
            for (int i = 0; i < line; i++)
            {
                auto idx = tid + i * threads;
                sptr[idx] = gptr[idx];
            }
            if (remain > 0 && tid < remain)
            {
                auto idx = tid + line * threads;
                sptr[idx] = gptr[idx];
            }
        }
        else if (tid < size) sptr[tid] = gptr[tid];
    }

    template <int size, typename T>
    __device__ static inline T* copyToShared(const T* const __restrict__ gptr) noexcept
    {
        __shared__ T sptr[size];
        copyToShared<size>(sptr, gptr);
        return sptr;
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

    struct OpImplCUDA
    {
        template <int vsize, typename T1, typename T2>
        __device__ static float dot(const T1* const __restrict__ v1, const T2* const __restrict__ v2) noexcept
        {
            float sum = 0.0f;

            for (int i = 0; i < vsize; i++) sum += cast<float>(v1[i]) * cast<float>(v2[i]);

            return sum;
        }

        template <int cout, int cpos, typename T>
        __device__ static void conv_cin1(const T* const __restrict__ rptr, float* const __restrict__ out, const float* const __restrict__ kernels, const float* const __restrict__ biases) noexcept
        {
            for (int n = 0; n < cout; n++)
            {
                auto kptr = kernels + n * cpos;
                out[n] = biases[n];
                for (int i = 0; i < cpos; i++) out[n] += cast<float>(rptr[i]) * kptr[i];
            }
        }

        template <int cin, int cout, int cpos, typename T>
        __device__ static void conv(const T* const* const rptr, float* const __restrict__ out, const float* const __restrict__ kernels, const float* const __restrict__ biases) noexcept
        {
            for (int n = 0; n < cout; n++)
            {
                out[n] = biases[n];
                for (int p = 0; p < cpos; p++)
                    for (int c = 0; c < cin; c++)
                        out[n] += cast<float>(rptr[p][c]) * kernels[n * cin * cpos + cin * p + c];
            }
        }
    };

    namespace kernel
    {
        template <typename OpImpl, typename IN, typename OUT, int kw, int kh, int cout, typename ActiveFunc>
        __global__ void conv_cin1(
            const void* const __restrict__ sptr,
            const int srcW, const int srcH, const int srcC, const int spitch,
            void* const __restrict__ dptr,
            const int dstW, const int dstH, const int dstC, const int dpitch,
            const float* const __restrict__ kernels,
            const float* const __restrict__ biases,
            ActiveFunc activeFunc)
        {
            static_assert(kw % 2 == 1 && kh % 2 == 1, "kw and kh must be odd");

            auto x = blockIdx.x * BlockSize::x + threadIdx.x;
            auto y = blockIdx.y * BlockSize::y + threadIdx.y;

            constexpr auto hkw = kw / 2;
            constexpr auto hkh = kh / 2;

            auto kptr = copyToShared<cout * kw * kh * 1>(kernels);
            auto bptr = copyToShared<cout>(biases);

            __syncthreads();

            if (x >= srcW || y >= srcH) return;

            float rptr[kw * kh];
            for (int ypos = -hkh; ypos <= hkh; ypos++)
                for (int xpos = -hkw; xpos <= hkw; xpos++)
                    rptr[(ypos + hkh) * kw + xpos + hkw] = cast<float>(*getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, x + xpos, y + ypos));

            float sum[cout];

            OpImpl::template conv_cin1<cout, kw * kh>(rptr, sum, kptr, bptr);

            auto out = getWritePtr<OUT>(dptr, dstW, dstH, dstC, dpitch, x, y);
            for (int n = 0; n < cout; n++) out[n] = cast<OUT>(activeFunc(sum[n], n));
        }

        template <typename OpImpl, typename IN, typename OUT, int kw, int kh, int cin, int cout, bool postactive = false, typename ActiveFunc, typename... ResidualArgs>
        __global__ void conv(
            const void* const __restrict__ sptr,
            const int srcW, const int srcH, const int srcC, const int spitch,
            const typename RestrictPointer<!sizeof...(ResidualArgs), void>::Type dptr,
            const int dstW, const int dstH, const int dstC, const int dpitch,
            const float* const __restrict__ kernels,
            const float* const __restrict__ biases,
            ActiveFunc activeFunc, ResidualArgs ...residualArg)
        {
            static_assert(kw % 2 == 1 && kh % 2 == 1, "kw and kh must be odd");

            [[maybe_unused]] const ::cuda::std::array<ResidualArg, sizeof...(ResidualArgs)> residualArgs{ residualArg... };

            auto x = blockIdx.x * BlockSize::x + threadIdx.x;
            auto y = blockIdx.y * BlockSize::y + threadIdx.y;

            constexpr auto hkw = kw / 2;
            constexpr auto hkh = kh / 2;

            auto kptr = copyToShared<cout * kw * kh * cin>(kernels);
            auto bptr = copyToShared<cout>(biases);
            auto iptr = imageBlockToSharedAdaptive<IN, cin, hkw, hkh>(sptr, srcW, srcH, srcC, spitch);

            __syncthreads();

            if (x >= srcW || y >= srcH) return;

            const IN* rptr[kw * kh];
            loadImageBlockAdaptive<IN, kw, kh, cin, hkw, hkh>(rptr, x, y, iptr, srcW, srcH, srcC, spitch);

            float sum[cout];

            OpImpl::template conv<cin, cout, kw * kh>(rptr, sum, kptr, bptr);

            auto out = getWritePtr<OUT>(dptr, dstW, dstH, dstC, dpitch, x, y);

            for (int n = 0; n < cout; n++)
            {
                if constexpr (!postactive) sum[n] = activeFunc(sum[n], n);

                if constexpr (sizeof...(ResidualArgs))
                    for (int idx = 0; idx < sizeof...(ResidualArgs); idx++)
                        sum[n] = sum[n] * residualArgs[idx].scale + cast<float>(getReadPtr<IN>(residualArgs[idx].ptr, residualArgs[idx].w, residualArgs[idx].h, residualArgs[idx].c, residualArgs[idx].pitch, x, y)[n]);

                if constexpr (postactive) sum[n] = activeFunc(sum[n], n);

                out[n] = cast<OUT>(sum[n]);
            }
        }

        template <typename OpImpl, typename IN, typename OUT, int cin, int ctemp, int cout, bool postactive3x3 = false, bool postactive1x1 = false, typename ActiveFunc3x3, typename ResidualArgs3x3, typename ActiveFunc1x1, typename ResidualArgs1x1>
        __global__ void conv3x3_conv1x1(
            const void* const __restrict__ sptr,
            const int srcW, const int srcH, const int srcC, const int spitch,
            const typename RestrictPointer<::cuda::std::is_same_v<ResidualArgs3x3, ResidualArg>, void>::Type dptr,
            const int dstW, const int dstH, const int dstC, const int dpitch,
            const float* const __restrict__ kernels3x3, const float* const __restrict__ biases3x3, ActiveFunc3x3 activeFunc3x3, ResidualArgs3x3 residualArg3x3,
            const float* const __restrict__ kernels1x1, const float* const __restrict__ biases1x1, ActiveFunc1x1 activeFunc1x1, ResidualArgs1x1 residualArg1x1)
        {
            auto x = blockIdx.x * BlockSize::x + threadIdx.x;
            auto y = blockIdx.y * BlockSize::y + threadIdx.y;

            constexpr auto padx = 1;
            constexpr auto pady = 1;
            constexpr auto knum3x3 = ctemp * 9 * cin;
            constexpr auto bnum3x3 = ctemp;
            constexpr auto knum1x1 = cout * 1 * ctemp;
            constexpr auto bnum1x1 = cout;
            constexpr auto ksize = knum3x3 > knum1x1 ? knum3x3 : knum1x1;
            constexpr auto bsize = bnum3x3 > bnum1x1 ? bnum3x3 : bnum1x1;

            __shared__ float kptr[ksize];
            __shared__ float bptr[bsize];

            copyToShared<knum3x3>(kptr, kernels3x3);
            copyToShared<bnum3x3>(bptr, biases3x3);

            auto iptr = imageBlockToSharedAdaptive<IN, cin, padx, pady>(sptr, srcW, srcH, srcC, spitch);

            __syncthreads();

            float buffer[ctemp];

            // conv3x3
            if (x < srcW && y < srcH)
            {
                const IN* rptr3x3[9];
                loadImageBlockAdaptive<IN, 3, 3, cin, padx, pady>(rptr3x3, x, y, iptr, srcW, srcH, srcC, spitch);

                OpImpl::template conv<cin, ctemp, 3 * 3>(rptr3x3, buffer, kptr, bptr);

                for (int n = 0; n < ctemp; n++)
                {
                    if constexpr (!postactive3x3) buffer[n] = activeFunc3x3(buffer[n], n);

                    if constexpr (::cuda::std::is_same_v<ResidualArgs3x3, ResidualArg>)
                        buffer[n] = buffer[n] * residualArg3x3.scale + cast<float>(getReadPtr<IN>(residualArg3x3.ptr, residualArg3x3.w, residualArg3x3.h, residualArg3x3.c, residualArg3x3.pitch, x, y)[n]);

                    if constexpr (postactive3x3) buffer[n] = activeFunc3x3(buffer[n], n);
                }
            }

            copyToShared<knum1x1>(kptr, kernels1x1);
            copyToShared<bnum1x1>(bptr, biases1x1);

            __syncthreads();

            // conv1x1
            if (x < srcW && y < srcH)
            {
                float sum[cout];
                const float* rptr1x1[] = { buffer };
                OpImpl::template conv<ctemp, cout, 1 * 1>(rptr1x1, sum, kptr, bptr);

                auto out = getWritePtr<OUT>(dptr, dstW, dstH, dstC, dpitch, x, y);

                for (int n = 0; n < cout; n++)
                {
                    if constexpr (!postactive1x1) sum[n] = activeFunc1x1(sum[n], n);

                    if constexpr (::cuda::std::is_same_v<ResidualArgs1x1, ResidualArg>)
                        sum[n] = sum[n] * residualArg1x1.scale + cast<float>(getReadPtr<IN>(residualArg1x1.ptr, residualArg1x1.w, residualArg1x1.h, residualArg1x1.c, residualArg1x1.pitch, x, y)[n]);

                    if constexpr (postactive1x1) sum[n] = activeFunc1x1(sum[n], n);

                    out[n] = cast<OUT>(sum[n]);
                }
            }
        }
    
        template <typename OpImpl, typename IN, typename OUT, int kw, int kh, int cin, int upscale, typename ActiveFunc, typename NearestInterpolationArg>
        __global__ void conv_pixelshuffle(
            const void* const __restrict__ sptr,
            const int srcW, const int srcH, const int srcC, const int spitch,
            void* const __restrict__ dptr,
            const int dstW, const int dstH, const int dstC, const int dpitch,
            const float* const __restrict__ kernels,
            const float* const __restrict__ biases,
            ActiveFunc activeFunc,
            NearestInterpolationArg nearestInterpolationArg)
        {
            static_assert(kw % 2 == 1 && kh % 2 == 1, "kw and kh must be odd");

            auto x = blockIdx.x * BlockSize::x + threadIdx.x;
            auto y = blockIdx.y * BlockSize::y + threadIdx.y;

            constexpr auto cout = upscale * upscale;

            constexpr auto hkw = kw / 2;
            constexpr auto hkh = kh / 2;

            auto kptr = copyToShared<cout * kw * kh * cin>(kernels);
            auto bptr = copyToShared<cout>(biases);
            auto iptr = imageBlockToSharedAdaptive<IN, cin, hkw, hkh>(sptr, srcW, srcH, srcC, spitch);

            __syncthreads();

            if (x >= srcW || y >= srcH) return;

            const IN* rptr[kw * kh];
            loadImageBlockAdaptive<IN, kw, kh, cin, hkw, hkh>(rptr, x, y, iptr, srcW, srcH, srcC, spitch);

            float sum[cout];

            OpImpl::template conv<cin, cout, kw* kh>(rptr, sum, kptr, bptr);

            auto dstX = x * upscale;
            auto dstY = y * upscale;

            for (int n = 0; n < cout; n++)
            {
                sum[n] = activeFunc(sum[n], n);

                if constexpr (::cuda::std::is_same_v<NearestInterpolationArg, ResidualArg>)
                    sum[n] = sum[n] * nearestInterpolationArg.scale + cast<float>(*getReadPtr<OUT>(nearestInterpolationArg.ptr, nearestInterpolationArg.w, nearestInterpolationArg.h, nearestInterpolationArg.c, nearestInterpolationArg.pitch, x, y));

                *getWritePtr<OUT>(dptr, dstW, dstH, dstC, dpitch, dstX + (n & 1), dstY + (n >> 1)) = cast<OUT>(saturate(sum[n]));
            }
        }

        template <typename OpImpl, typename IN, typename OUT, int kw, int kh, int cin, int ctemp, int cout, typename ActiveFunc>
        __global__ void conv_deconv2x2(
            const void* const __restrict__ sptr,
            const int srcW, const int srcH, const int srcC, const int spitch,
            void* const __restrict__ dptr,
            const int dstW, const int dstH, const int dstC, const int dpitch,
            const float* const __restrict__ kernels1,
            const float* const __restrict__ biases1,
            const float* const __restrict__ kernels2,
            ActiveFunc activeFunc)
        {
            static_assert(kw % 2 == 1 && kh % 2 == 1, "kw and kh must be odd");

            auto x = blockIdx.x * BlockSize::x + threadIdx.x;
            auto y = blockIdx.y * BlockSize::y + threadIdx.y;

            constexpr auto upscale = 2;

            constexpr auto hkw = kw / 2;
            constexpr auto hkh = kh / 2;

            auto kptr1 = copyToShared<ctemp * kw * kh * cin>(kernels1);
            auto bptr1 = copyToShared<ctemp>(biases1);
            auto kptr2 = copyToShared<cout * 2 * 2 * ctemp>(kernels2);
            auto iptr = imageBlockToSharedAdaptive<IN, cin, hkw, hkh>(sptr, srcW, srcH, srcC, spitch);

            __syncthreads();

            if (x >= srcW || y >= srcH) return;

            const IN* rptr[kw * kh];
            loadImageBlockAdaptive<IN, kw, kh, cin, hkw, hkh>(rptr, x, y, iptr, srcW, srcH, srcC, spitch);

            float sum[ctemp];

            OpImpl::template conv<cin, ctemp, kw* kh>(rptr, sum, kptr1, bptr1);

            for (int n = 0; n < ctemp; n++) sum[n] = activeFunc(sum[n], n);

            auto dstX = x * upscale;
            auto dstY = y * upscale;
            for (int dy = 0; dy < upscale; dy++)
            {
                for (int dx = 0; dx < upscale; dx++)
                {
                    auto out = getWritePtr<OUT>(dptr, dstW, dstH, dstC, dpitch, dstX + dx, dstY + dy);
                    for (int n = 0; n < cout; n++) out[n] = cast<OUT>(saturate(OpImpl::template dot<ctemp>(sum, kptr2 + n * ctemp * 4 + ctemp * (dy * upscale + dx))));
                }
            }
        }
    }

    template<>
    void conv3x3_1to8_relu_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        switch (src.type())
        {
        case DeviceImage::UInt8:
            kernel::conv_cin1<OpImplCUDA, DataType::UInt8, DataType::Float16, 3, 3, 8> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, ReLU{});
            break;
        case DeviceImage::UInt16:
            kernel::conv_cin1<OpImplCUDA, DataType::UInt16, DataType::Float16, 3, 3, 8> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, ReLU{});
            break;
        case DeviceImage::Float16:
            kernel::conv_cin1<OpImplCUDA, DataType::Float16, DataType::Float16, 3, 3, 8> << < grid, block, 0, stream >> > (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, ReLU{});
            break;
        case DeviceImage::Float32:
            kernel::conv_cin1<OpImplCUDA, DataType::Float32, DataType::Float16, 3, 3, 8> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, ReLU{});
            break;
        }
    }
    template<>
    void conv3x3_8to8_relu_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        kernel::conv<OpImplCUDA, DataType::Float16, DataType::Float16, 3, 3, 8, 8> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, ReLU{});
    }
    template<>
    void conv3x3_8to8_relu_deconv2x2_8to1_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels1, const float* biases1,
        const float* kernels2,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        switch (dst.type())
        {
        case DeviceImage::UInt8:
            kernel::conv_deconv2x2<OpImplCUDA, DataType::Float16, DataType::UInt8, 3, 3, 8, 8, 1> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels1, biases1, kernels2, ReLU{});
            break;
        case DeviceImage::UInt16:
            kernel::conv_deconv2x2<OpImplCUDA, DataType::Float16, DataType::UInt16, 3, 3, 8, 8, 1> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels1, biases1, kernels2, ReLU{});
            break;
        case DeviceImage::Float16:
            kernel::conv_deconv2x2<OpImplCUDA, DataType::Float16, DataType::Float16, 3, 3, 8, 8, 1> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels1, biases1, kernels2, ReLU{});
            break;
        case DeviceImage::Float32:
            kernel::conv_deconv2x2<OpImplCUDA, DataType::Float16, DataType::Float32, 3, 3, 8, 8, 1> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels1, biases1, kernels2, ReLU{});
            break;
        }
    }

    template<>
    void conv3x3_1to8_prelu_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases, const float* alphas,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        switch (src.type())
        {
        case DeviceImage::UInt8:
            kernel::conv_cin1<OpImplCUDA, DataType::UInt8, DataType::Float16, 3, 3, 8> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, PReLU{ alphas });
            break;
        case DeviceImage::UInt16:
            kernel::conv_cin1<OpImplCUDA, DataType::UInt16, DataType::Float16, 3, 3, 8> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, PReLU{ alphas });
            break;
        case DeviceImage::Float16:
            kernel::conv_cin1<OpImplCUDA, DataType::Float16, DataType::Float16, 3, 3, 8> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, PReLU{ alphas });
            break;
        case DeviceImage::Float32:
            kernel::conv_cin1<OpImplCUDA, DataType::Float32, DataType::Float16, 3, 3, 8> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, PReLU{ alphas });
            break;
        }
    }

    template<>
    void conv3x3_1to8_identity_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        switch (src.type())
        {
        case DeviceImage::UInt8:
            kernel::conv_cin1<OpImplCUDA, DataType::UInt8, DataType::Float16, 3, 3, 8> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{});
            break;
        case DeviceImage::UInt16:
            kernel::conv_cin1<OpImplCUDA, DataType::UInt16, DataType::Float16, 3, 3, 8> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{});
            break;
        case DeviceImage::Float16:
            kernel::conv_cin1<OpImplCUDA, DataType::Float16, DataType::Float16, 3, 3, 8> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{});
            break;
        case DeviceImage::Float32:
            kernel::conv_cin1<OpImplCUDA, DataType::Float32, DataType::Float16, 3, 3, 8> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{});
            break;
        }
    }
    template<>
    void conv3x3_8to8_prelu_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases, const float* alphas,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        kernel::conv<OpImplCUDA, DataType::Float16, DataType::Float16, 3, 3, 8, 8> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, PReLU{ alphas });
    }
    template<>
    void conv3x3_8to8_identity_residual_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        const DeviceImage& idt, const float scale,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        kernel::conv<OpImplCUDA, DataType::Float16, DataType::Float16, 3, 3, 8, 8> <<< grid, block, 0, stream >>> (
            src.ptr(), src.width(), src.height(), src.channels(), src.stride(),
            dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(),
            kernels, biases,
            Identity{},
            ResidualArg{ idt.ptr(), idt.width(), idt.height(), idt.channels(), idt.stride(), scale });
    }
    template<>
    void conv3x3_8to8_identity_residual_conv1x1_8to8_prelu_add_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels1, const float* biases1,
        const DeviceImage& idt, const float scale,
        const float* kernels2, const float* biases2, const float* alphas2,
        const DeviceImage& feat,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        kernel::conv3x3_conv1x1<OpImplCUDA, DataType::Float16, DataType::Float16, 8, 8, 8, false, false> <<< grid, block, 0, stream >>> (
            src.ptr(), src.width(), src.height(), src.channels(), src.stride(),
            dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(),
            kernels1, biases1, Identity{}, ResidualArg{ idt.ptr(), idt.width(), idt.height(), idt.channels(), idt.stride(), scale },
            kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ feat.ptr(), feat.width(), feat.height(), feat.channels(), feat.stride(), 1.0f });
    }
    template<>
    void conv3x3_8to4_identity_pixelshuffle_4to1_add_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        const DeviceImage& idt,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        switch (dst.type())
        {
        case DeviceImage::UInt8:
            kernel::conv_pixelshuffle<OpImplCUDA, DataType::Float16, DataType::UInt8, 3, 3, 8, 2> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{}, ResidualArg{ idt.ptr(), idt.width(), idt.height(), idt.channels(), idt.stride(), 1.0f});
            break;
        case DeviceImage::UInt16:
            kernel::conv_pixelshuffle<OpImplCUDA, DataType::Float16, DataType::UInt16, 3, 3, 8, 2> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{}, ResidualArg{ idt.ptr(), idt.width(), idt.height(), idt.channels(), idt.stride(), 1.0f });
            break;
        case DeviceImage::Float16:
            kernel::conv_pixelshuffle<OpImplCUDA, DataType::Float16, DataType::Float16, 3, 3, 8, 2> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{}, ResidualArg{ idt.ptr(), idt.width(), idt.height(), idt.channels(), idt.stride(), 1.0f });
            break;
        case DeviceImage::Float32:
            kernel::conv_pixelshuffle<OpImplCUDA, DataType::Float16, DataType::Float32, 3, 3, 8, 2> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{}, ResidualArg{ idt.ptr(), idt.width(), idt.height(), idt.channels(), idt.stride(), 1.0f });
            break;
        }
    }

    template<>
    void conv3x3_1to16_identity_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        switch (src.type())
        {
        case DeviceImage::UInt8:
            kernel::conv_cin1<OpImplCUDA, DataType::UInt8, DataType::Float16, 3, 3, 16> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{});
            break;
        case DeviceImage::UInt16:
            kernel::conv_cin1<OpImplCUDA, DataType::UInt16, DataType::Float16, 3, 3, 16> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{});
            break;
        case DeviceImage::Float16:
            kernel::conv_cin1<OpImplCUDA, DataType::Float16, DataType::Float16, 3, 3, 16> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{});
            break;
        case DeviceImage::Float32:
            kernel::conv_cin1<OpImplCUDA, DataType::Float32, DataType::Float16, 3, 3, 16> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{});
            break;
        }
    }
    template<>
    void conv3x3_16to16_relu_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        kernel::conv<OpImplCUDA, DataType::Float16, DataType::Float16, 3, 3, 16, 16> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, ReLU{});
    }
    template<>
    void conv3x3_16to16_identity_add_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        const DeviceImage& feat,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        kernel::conv<OpImplCUDA, DataType::Float16, DataType::Float16, 3, 3, 16, 16> <<< grid, block, 0, stream >>> (
            src.ptr(), src.width(), src.height(), src.channels(), src.stride(),
            dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(),
            kernels, biases,
            Identity{},
            ResidualArg{ feat.ptr(), feat.width(), feat.height(), feat.channels(), feat.stride(), 1.0f });
    }
    template<>
    void conv3x3_16to4_identity_pixelshuffle_4to1_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        switch (dst.type())
        {
        case DeviceImage::UInt8:
            kernel::conv_pixelshuffle<OpImplCUDA, DataType::Float16, DataType::UInt8, 3, 3, 16, 2> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{}, nullptr);
            break;
        case DeviceImage::UInt16:
            kernel::conv_pixelshuffle<OpImplCUDA, DataType::Float16, DataType::UInt16, 3, 3, 16, 2> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{}, nullptr);
            break;
        case DeviceImage::Float16:
            kernel::conv_pixelshuffle<OpImplCUDA, DataType::Float16, DataType::Float16, 3, 3, 16, 2> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{}, nullptr);
            break;
        case DeviceImage::Float32:
            kernel::conv_pixelshuffle<OpImplCUDA, DataType::Float16, DataType::Float32, 3, 3, 16, 2> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{}, nullptr);
            break;
        }
    }

    template<>
    void conv3x3_1to32_identity_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        switch (src.type())
        {
        case DeviceImage::UInt8:
            kernel::conv_cin1<OpImplCUDA, DataType::UInt8, DataType::Float16, 3, 3, 32> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{});
            break;
        case DeviceImage::UInt16:
            kernel::conv_cin1<OpImplCUDA, DataType::UInt16, DataType::Float16, 3, 3, 32> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{});
            break;
        case DeviceImage::Float16:
            kernel::conv_cin1<OpImplCUDA, DataType::Float16, DataType::Float16, 3, 3, 32> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{});
            break;
        case DeviceImage::Float32:
            kernel::conv_cin1<OpImplCUDA, DataType::Float32, DataType::Float16, 3, 3, 32> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{});
            break;
        }
    }
    template<>
    void conv3x3_32to32_relu_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        kernel::conv<OpImplCUDA, DataType::Float16, DataType::Float16, 3, 3, 32, 32> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, ReLU{});
    }
    template<>
    void conv3x3_32to32_identity_add_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        const DeviceImage& feat,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        kernel::conv<OpImplCUDA, DataType::Float16, DataType::Float16, 3, 3, 32, 32> <<< grid, block, 0, stream >>> (
            src.ptr(), src.width(), src.height(), src.channels(), src.stride(),
            dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(),
            kernels, biases,
            Identity{},
            ResidualArg{ feat.ptr(), feat.width(), feat.height(), feat.channels(), feat.stride(), 1.0f });
    }
    template<>
    void conv3x3_32to4_identity_pixelshuffle_4to1_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        switch (dst.type())
        {
        case DeviceImage::UInt8:
            kernel::conv_pixelshuffle<OpImplCUDA, DataType::Float16, DataType::UInt8, 3, 3, 32, 2> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{}, nullptr);
            break;
        case DeviceImage::UInt16:
            kernel::conv_pixelshuffle<OpImplCUDA, DataType::Float16, DataType::UInt16, 3, 3, 32, 2> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{}, nullptr);
            break;
        case DeviceImage::Float16:
            kernel::conv_pixelshuffle<OpImplCUDA, DataType::Float16, DataType::Float16, 3, 3, 32, 2> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{}, nullptr);
            break;
        case DeviceImage::Float32:
            kernel::conv_pixelshuffle<OpImplCUDA, DataType::Float16, DataType::Float32, 3, 3, 32, 2> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{}, nullptr);
            break;
        }
    }

    template<>
    void conv5x5_1to8_identity_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        switch (src.type())
        {
        case DeviceImage::UInt8:
            kernel::conv_cin1<OpImplCUDA, DataType::UInt8, DataType::Float16, 5, 5, 8> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{});
            break;
        case DeviceImage::UInt16:
            kernel::conv_cin1<OpImplCUDA, DataType::UInt16, DataType::Float16, 5, 5, 8> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{});
            break;
        case DeviceImage::Float16:
            kernel::conv_cin1<OpImplCUDA, DataType::Float16, DataType::Float16, 5, 5, 8> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{});
            break;
        case DeviceImage::Float32:
            kernel::conv_cin1<OpImplCUDA, DataType::Float32, DataType::Float16, 5, 5, 8> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{});
            break;
        }
    }
    template<>
    void conv3x3_8to8_prelu_conv1x1_8to8_add_prelu_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const DeviceImage& feat,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        kernel::conv3x3_conv1x1<OpImplCUDA, DataType::Float16, DataType::Float16, 8, 8, 8, false, true> <<< grid, block, 0, stream >>> (
            src.ptr(), src.width(), src.height(), src.channels(), src.stride(),
            dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(),
            kernels1, biases1, PReLU(alphas1), nullptr,
            kernels2, biases2, PReLU(alphas2), ResidualArg{ feat.ptr(), feat.width(), feat.height(), feat.channels(), feat.stride(), 1.0f });
    }
    template<>
    void conv3x3_8to4_identity_pixelshuffle_4to1_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        switch (dst.type())
        {
        case DeviceImage::UInt8:
            return kernel::conv_pixelshuffle<OpImplCUDA, DataType::Float16, DataType::UInt8, 3, 3, 8, 2> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{}, nullptr);
        case DeviceImage::UInt16:
            return kernel::conv_pixelshuffle<OpImplCUDA, DataType::Float16, DataType::UInt16, 3, 3, 8, 2> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{}, nullptr);
        case DeviceImage::Float16:
            return kernel::conv_pixelshuffle<OpImplCUDA, DataType::Float16, DataType::Float16, 3, 3, 8, 2> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{}, nullptr);
        case DeviceImage::Float32:
            return kernel::conv_pixelshuffle<OpImplCUDA, DataType::Float16, DataType::Float32, 3, 3, 8, 2> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{}, nullptr);
        }
    }

    template<>
    void conv5x5_1to16_identity_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        switch (src.type())
        {
        case DeviceImage::UInt8:
            kernel::conv_cin1<OpImplCUDA, DataType::UInt8, DataType::Float16, 5, 5, 16> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{});
            break;
        case DeviceImage::UInt16:
            kernel::conv_cin1<OpImplCUDA, DataType::UInt16, DataType::Float16, 5, 5, 16> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{});
            break;
        case DeviceImage::Float16:
            kernel::conv_cin1<OpImplCUDA, DataType::Float16, DataType::Float16, 5, 5, 16> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{});
            break;
        case DeviceImage::Float32:
            kernel::conv_cin1<OpImplCUDA, DataType::Float32, DataType::Float16, 5, 5, 16> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, Identity{});
            break;
        }
    }
    template<>
    void conv3x3_16to16_prelu_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels, const float* biases, const float* alphas,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        kernel::conv<OpImplCUDA, DataType::Float16, DataType::Float16, 3, 3, 16, 16> <<< grid, block, 0, stream >>> (src.ptr(), src.width(), src.height(), src.channels(), src.stride(), dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(), kernels, biases, PReLU{ alphas });
    }
    template<>
    void conv3x3_16to16_prelu_conv1x1_16to16_add_prelu_cuda<DeviceImage::Float16>(
        const DeviceImage& src, DeviceImage& dst,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const DeviceImage& feat,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (src.width() + block.x - 1) / block.x, (src.height() + block.y - 1) / block.y };
        kernel::conv3x3_conv1x1<OpImplCUDA, DataType::Float16, DataType::Float16, 16, 16, 16, false, true> <<< grid, block, 0, stream >>> (
            src.ptr(), src.width(), src.height(), src.channels(), src.stride(),
            dst.ptr(), dst.width(), dst.height(), dst.channels(), dst.stride(),
            kernels1, biases1, PReLU(alphas1), nullptr,
            kernels2, biases2, PReLU(alphas2), ResidualArg{ feat.ptr(), feat.width(), feat.height(), feat.channels(), feat.stride(), 1.0f });
    }
}
