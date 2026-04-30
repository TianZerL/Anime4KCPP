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
    __device__ static float prelu(float v, float n) noexcept
    {
        return fmaxf(v, 0.0f) + n * fminf(v, 0.0f);
    }

    class Identity
    {
    public:
        Identity() noexcept = default;
        __device__ float operator() (const float v, const int /*c*/) const noexcept { return identity(v); }
    };
    class ReLU
    {
    public:
        ReLU() noexcept = default;
        __device__ float operator() (const float v, const int /*c*/) const noexcept { return relu(v); }
    };
    class LReLU
    {
    public:
        LReLU(const float negativeSlope) noexcept : negativeSlope(negativeSlope) {}
        __device__ float operator() (const float v, const int /*c*/) const noexcept { return lrelu(v, negativeSlope); }

    private:
        const float negativeSlope;
    };
    class PReLU
    {
    public:
        PReLU(const float* __restrict__ alphas) noexcept : alphas(alphas) {}
        __device__ float operator() (const float v, const int c) const noexcept { return prelu(v, alphas[c]); }

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
        __device__ static float dot(const T1* const v1, const T2* const v2) noexcept
        {
            float sum = 0.0f;

            for (int i = 0; i < vsize; i++) sum += toFloat(v1[i]) * toFloat(v2[i]);

            return sum;
        }

        template <int cout, int cpos, typename T>
        __device__ static void conv_cin1(const T* const rptr, float* const __restrict__ out, const float* const __restrict__ kernels, const float* const __restrict__ biases) noexcept
        {
            for (int n = 0; n < cout; n++)
            {
                auto kptr = kernels + n * cpos;
                out[n] = biases[n];
                for (int i = 0; i < cpos; i++) out[n] += toFloat(rptr[i]) * kptr[i];
            }
        }

        template <int cin, int cout, int cpos, typename T>
        __device__ static void conv(const T** const rptr, float* const __restrict__ out, const float* const __restrict__ kernels, const float* const __restrict__ biases) noexcept
        {
            for (int n = 0; n < cout; n++)
            {
                out[n] = biases[n];
                for (int p = 0; p < cpos; p++)
                    for (int c = 0; c < cin; c++)
                        out[n] += toFloat(rptr[p][c]) * kernels[n * cin * cpos + cin * p + c];
            }
        }
    };

    namespace kernel
    {
        template<typename OpImpl, typename IN, typename OUT, int cin, int cout>
        __global__ void deconv2x2_cuda(
            const void* const __restrict__ sptr,
            const int srcW, const int srcH, const int srcC, const int spitch,
            void* const __restrict__ dptr,
            const int dstW, const int dstH, const int dstC, const int dpitch,
            const float* const __restrict__ kernels)
        {
            constexpr int upscale = 2;

            auto x = blockIdx.x * BlockSize::x + threadIdx.x;
            auto y = blockIdx.y * BlockSize::y + threadIdx.y;

            if (x >= dstW || y >= dstH) return;

            auto in = getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, x / 2, y / 2);
            auto out = getWritePtr<OUT>(dptr, dstW, dstH, dstC, dpitch, x, y);

            auto index = (y % upscale) * upscale + (x % upscale);

            for (int n = 0; n < cout; n++) out[n] = fromFloat<OUT>(OpImpl::template dot<cin>(in, kernels + n * cin * 4 + cin * index));
        }

        template <typename OpImpl, typename IN, int kw, int kh, int cout, typename ActiveFunc>
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
                    rptr[(ypos + hkh) * kw + xpos + hkw] = toFloat(*getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, x + xpos, y + ypos));

            float sum[cout];

            OpImpl::template conv_cin1<cout, kw * kh>(rptr, sum, kptr, bptr);

            auto out = getWritePtr<half>(dptr, dstW, dstH, dstC, dpitch, x, y);
            for (int n = 0; n < cout; n++) out[n] = toHalf(activeFunc(sum[n], n));
        }

        template <typename OpImpl, typename IN, int kw, int kh, int cin, int cout, bool postactive = false, typename ActiveFunc, typename... ResidualArgs>
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

            auto out = getWritePtr<half>(dptr, dstW, dstH, dstC, dpitch, x, y);

            for (int n = 0; n < cout; n++)
            {
                if constexpr (!postactive) sum[n] = activeFunc(sum[n], n);

                if constexpr (sizeof...(ResidualArgs))
                    for (int idx = 0; idx < sizeof...(ResidualArgs); idx++)
                        sum[n] = sum[n] * residualArgs[idx].scale + toFloat(getReadPtr<IN>(residualArgs[idx].ptr, residualArgs[idx].w, residualArgs[idx].h, residualArgs[idx].c, residualArgs[idx].pitch, x, y)[n]);

                if constexpr (postactive) sum[n] = activeFunc(sum[n], n);

                out[n] = toHalf(sum[n]);
            }
        }

        template <typename OpImpl, typename IN, typename OUT, int kw, int kh, int cin, int upscale, typename NearestInterpolationArg>
        __global__ void conv_identity_pixelshuffle(
            const void* const __restrict__ sptr,
            const int srcW, const int srcH, const int srcC, const int spitch,
            void* const __restrict__ dptr,
            const int dstW, const int dstH, const int dstC, const int dpitch,
            const float* const __restrict__ kernels,
            const float* const __restrict__ biases,
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

            OpImpl::template conv<cin, cout, kw * kh>(rptr, sum, kptr, bptr);

            auto dstX = x * upscale;
            auto dstY = y * upscale;

            for (int n = 0; n < cout; n++)
            {
                if constexpr (::cuda::std::is_same_v<NearestInterpolationArg, ResidualArg>)
                    sum[n] = sum[n] * nearestInterpolationArg.scale + toFloat(*getReadPtr<OUT>(nearestInterpolationArg.ptr, nearestInterpolationArg.w, nearestInterpolationArg.h, nearestInterpolationArg.c, nearestInterpolationArg.pitch, x, y));

                *getWritePtr<OUT>(dptr, dstW, dstH, dstC, dpitch, dstX + (n & 1), dstY + (n >> 1)) = fromFloat<OUT>(sum[n]);
            }
        }

        template <typename OpImpl, typename IN, int cin, int ctemp, int cout, bool postactive3x3 = false, bool postactive1x1 = false, typename ActiveFunc3x3, typename ResidualArgs3x3, typename ActiveFunc1x1, typename ResidualArgs1x1>
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

            if (x >= srcW || y >= srcH) return;

            const IN* rptr3x3[9];
            loadImageBlockAdaptive<IN, 3, 3, cin, padx, pady>(rptr3x3, x, y, iptr, srcW, srcH, srcC, spitch);

            float buffer[ctemp];
            OpImpl::template conv<cin, ctemp, 3 * 3>(rptr3x3, buffer, kptr, bptr);

            for (int n = 0; n < ctemp; n++)
            {
                if constexpr (!postactive3x3) buffer[n] = activeFunc3x3(buffer[n], n);

                if constexpr (::cuda::std::is_same_v<ResidualArgs3x3, ResidualArg>)
                    buffer[n] = buffer[n] * residualArg3x3.scale + toFloat(getReadPtr<half>(residualArg3x3.ptr, residualArg3x3.w, residualArg3x3.h, residualArg3x3.c, residualArg3x3.pitch, x, y)[n]);

                if constexpr (postactive3x3) buffer[n] = activeFunc3x3(buffer[n], n);
            }

            copyToShared<knum1x1>(kptr, kernels1x1);
            copyToShared<bnum1x1>(bptr, biases1x1);

            __syncthreads();

            float sum[cout];
            const float* rptr1x1[] = { buffer };
            OpImpl::template conv<ctemp, cout, 1 * 1>(rptr1x1, sum, kptr, bptr);

            auto out = getWritePtr<IN>(dptr, dstW, dstH, dstC, dpitch, x, y);

            for (int n = 0; n < cout; n++)
            {
                if constexpr (!postactive1x1) sum[n] = activeFunc1x1(sum[n], n);

                if constexpr (::cuda::std::is_same_v<ResidualArgs1x1, ResidualArg>)
                    sum[n] = sum[n] * residualArg1x1.scale + toFloat(getReadPtr<half>(residualArg1x1.ptr, residualArg1x1.w, residualArg1x1.h, residualArg1x1.c, residualArg1x1.pitch, x, y)[n]);

                if constexpr (postactive1x1) sum[n] = activeFunc1x1(sum[n], n);

                out[n] = toHalf(sum[n]);
            }
        }
}

    void conv3x3_1to8_relu_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
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
            kernel::conv_cin1<OpImplCUDA, std::uint8_t, 3, 3, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, ReLU{});
            break;
        case Image::UInt16:
            kernel::conv_cin1<OpImplCUDA, std::uint16_t, 3, 3, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, ReLU{});
            break;
        case Image::Float32:
            kernel::conv_cin1<OpImplCUDA, float, 3, 3, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, ReLU{});
            break;
        }
    }
    void conv3x3_8to8_relu_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
        const float* kernels,
        const float* biases,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        kernel::conv<OpImplCUDA, half, 3, 3, 8, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, ReLU{});
    }
    void deconv2x2_8to1_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
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
            return kernel::deconv2x2_cuda<OpImplCUDA, half, std::uint8_t, 8, 1> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels);
        case Image::UInt16:
            return kernel::deconv2x2_cuda<OpImplCUDA, half, std::uint16_t, 8, 1> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels);
        case Image::Float32:
            return kernel::deconv2x2_cuda<OpImplCUDA, half, float, 8, 1> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels);
        }
    }

    void conv3x3_1to8_prelu_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
        const float* kernels,
        const float* biases,
        const float* alphas,
        Image::ElementType stype,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        switch (stype)
        {
        case Image::UInt8:
            kernel::conv_cin1<OpImplCUDA, std::uint8_t, 3, 3, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, PReLU{ alphas });
            break;
        case Image::UInt16:
            kernel::conv_cin1<OpImplCUDA, std::uint16_t, 3, 3, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, PReLU{ alphas });
            break;
        case Image::Float32:
            kernel::conv_cin1<OpImplCUDA, float, 3, 3, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, PReLU{ alphas });
            break;
        }
    }

    void conv3x3_1to8_identity_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
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
            kernel::conv_cin1<OpImplCUDA, std::uint8_t, 3, 3, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            kernel::conv_cin1<OpImplCUDA, std::uint16_t, 3, 3, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity{});
            break;
        case Image::Float32:
            kernel::conv_cin1<OpImplCUDA, float, 3, 3, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_8to8_prelu_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
        const float* kernels,
        const float* biases,
        const float* alphas,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        kernel::conv<OpImplCUDA, half, 3, 3, 8, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, PReLU{ alphas });
    }
    void conv3x3_8to8_identity_residual_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
        const float* kernels,
        const float* biases,
        const void* iptr, int idW, int idH, int idC, int ipitch,
        const float scale,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        kernel::conv<OpImplCUDA, half, 3, 3, 8, 8> <<< grid, block, 0, stream >>> (
            sptr, srcW, srcH, srcC, spitch,
            dptr, dstW, dstH, dstC, dpitch,
            kernels, biases,
            Identity{},
            ResidualArg{ iptr, idW, idH, idC, ipitch, scale }
        );
    }
    void conv3x3_8to8_identity_residual_conv1x1_8to8_prelu_add_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
        const float* kernels1, const float* biases1,
        const void* iptr, int idW, int idH, int idC, int ipitch, const float scale,
        const float* kernels2, const float* biases2, const float* alphas2,
        const void* fptr, int featW, int featH, int featC, int fpitch,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        kernel::conv3x3_conv1x1<OpImplCUDA, half, 8, 8, 8, false, false> <<< grid, block, 0, stream >>> (
            sptr, srcW, srcH, srcC, spitch,
            dptr, dstW, dstH, dstC, dpitch,
            kernels1, biases1, Identity{}, ResidualArg{iptr, idW, idH, idC, ipitch, scale },
            kernels2, biases2, PReLU{ alphas2 }, ResidualArg{ fptr, featW, featH, featC, fpitch, 1.0f }
        );
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_add_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
        const float* kernels,
        const float* biases,
        const void* iptr, int idW, int idH, int idC, int ipitch,
        Image::ElementType dtype,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        switch (dtype)
        {
        case Image::UInt8:
            return kernel::conv_identity_pixelshuffle<OpImplCUDA, half, std::uint8_t, 3, 3, 8, 2> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, ResidualArg{ iptr, idW, idH, idC, ipitch, 1.0f });
        case Image::UInt16:
            return kernel::conv_identity_pixelshuffle<OpImplCUDA, half, std::uint16_t, 3, 3, 8, 2> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, ResidualArg{ iptr, idW, idH, idC, ipitch, 1.0f });
        case Image::Float32:
            return kernel::conv_identity_pixelshuffle<OpImplCUDA, half, float, 3, 3, 8, 2> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, ResidualArg{ iptr, idW, idH, idC, ipitch, 1.0f });
        }
    }

    void conv3x3_1to16_identity_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
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
            kernel::conv_cin1<OpImplCUDA, std::uint8_t, 3, 3, 16> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            kernel::conv_cin1<OpImplCUDA, std::uint16_t, 3, 3, 16> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity{});
            break;
        case Image::Float32:
            kernel::conv_cin1<OpImplCUDA, float, 3, 3, 16> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_16to16_relu_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
        const float* kernels,
        const float* biases,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        kernel::conv<OpImplCUDA, half, 3, 3, 16, 16> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, ReLU{});
    }
    void conv3x3_16to16_identity_add_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
        const float* kernels,
        const float* biases,
        const void* fptr, int featW, int featH, int featC, int fpitch,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        kernel::conv<OpImplCUDA, half, 3, 3, 16, 16> <<< grid, block, 0, stream >>> (
            sptr, srcW, srcH, srcC, spitch,
            dptr, dstW, dstH, dstC, dpitch,
            kernels, biases,
            Identity{},
            ResidualArg{ fptr, featW, featH, featC, fpitch, 1.0f });
    }
    void conv3x3_16to4_identity_pixelshuffle_4to1_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
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
            return kernel::conv_identity_pixelshuffle<OpImplCUDA, half, std::uint8_t, 3, 3, 16, 2> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, nullptr);
        case Image::UInt16:
            return kernel::conv_identity_pixelshuffle<OpImplCUDA, half, std::uint16_t, 3, 3, 16, 2> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, nullptr);
        case Image::Float32:
            return kernel::conv_identity_pixelshuffle<OpImplCUDA, half, float, 3, 3, 16, 2> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, nullptr);
        }
    }

    void conv3x3_1to32_identity_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
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
            kernel::conv_cin1<OpImplCUDA, std::uint8_t, 3, 3, 32> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            kernel::conv_cin1<OpImplCUDA, std::uint16_t, 3, 3, 32> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity{});
            break;
        case Image::Float32:
            kernel::conv_cin1<OpImplCUDA, float, 3, 3, 32> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_32to32_relu_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
        const float* kernels,
        const float* biases,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        kernel::conv<OpImplCUDA, half, 3, 3, 32, 32> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, ReLU{});
    }
    void conv3x3_32to32_identity_add_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
        const float* kernels,
        const float* biases,
        const void* fptr, int featW, int featH, int featC, int fpitch,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        kernel::conv<OpImplCUDA, half, 3, 3, 32, 32> <<< grid, block, 0, stream >>> (
            sptr, srcW, srcH, srcC, spitch,
            dptr, dstW, dstH, dstC, dpitch,
            kernels, biases,
            Identity{},
            ResidualArg{ fptr, featW, featH, featC, fpitch, 1.0f });
    }
    void conv3x3_32to4_identity_pixelshuffle_4to1_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
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
            return kernel::conv_identity_pixelshuffle<OpImplCUDA, half, std::uint8_t, 3, 3, 32, 2> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, nullptr);
        case Image::UInt16:
            return kernel::conv_identity_pixelshuffle<OpImplCUDA, half, std::uint16_t, 3, 3, 32, 2> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, nullptr);
        case Image::Float32:
            return kernel::conv_identity_pixelshuffle<OpImplCUDA, half, float, 3, 3, 32, 2> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, nullptr);
        }
    }

    void conv5x5_1to8_identity_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
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
            kernel::conv_cin1<OpImplCUDA, std::uint8_t, 5, 5, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            kernel::conv_cin1<OpImplCUDA, std::uint16_t, 5, 5, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity{});
            break;
        case Image::Float32:
            kernel::conv_cin1<OpImplCUDA, float, 5, 5, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_8to8_prelu_conv1x1_8to8_add_prelu_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const void* fptr, int featW, int featH, int featC, int fpitch,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        kernel::conv3x3_conv1x1<OpImplCUDA, half, 8, 8, 8, false, true> <<< grid, block, 0, stream >>> (
            sptr, srcW, srcH, srcC, spitch,
            dptr, dstW, dstH, dstC, dpitch,
            kernels1, biases1, PReLU(alphas1), nullptr,
            kernels2, biases2, PReLU(alphas2), ResidualArg{ fptr, featW, featH, featC, fpitch, 1.0f }
        );
    }
    void conv3x3_8to4_identity_pixelshuffle_4to1_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
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
            return kernel::conv_identity_pixelshuffle<OpImplCUDA, half, std::uint8_t, 3, 3, 8, 2> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, nullptr);
        case Image::UInt16:
            return kernel::conv_identity_pixelshuffle<OpImplCUDA, half, std::uint16_t, 3, 3, 8, 2> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, nullptr);
        case Image::Float32:
            return kernel::conv_identity_pixelshuffle<OpImplCUDA, half, float, 3, 3, 8, 2> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, nullptr);
        }
    }

    void conv5x5_1to16_identity_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
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
            kernel::conv_cin1<OpImplCUDA, std::uint8_t, 5, 5, 16> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity{});
            break;
        case Image::UInt16:
            kernel::conv_cin1<OpImplCUDA, std::uint16_t, 5, 5, 16> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity{});
            break;
        case Image::Float32:
            kernel::conv_cin1<OpImplCUDA, float, 5, 5, 16> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity{});
            break;
        }
    }
    void conv3x3_16to16_prelu_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
        const float* kernels,
        const float* biases,
        const float* alphas,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        kernel::conv<OpImplCUDA, half, 3, 3, 16, 16> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, PReLU{ alphas });
    }
    void conv3x3_16to16_prelu_conv1x1_16to16_add_prelu_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        const void* fptr, int featW, int featH, int featC, int fpitch,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        kernel::conv3x3_conv1x1<OpImplCUDA, half, 16, 16, 16, false, true> <<< grid, block, 0, stream >>> (
            sptr, srcW, srcH, srcC, spitch,
            dptr, dstW, dstH, dstC, dpitch,
            kernels1, biases1, PReLU(alphas1), nullptr,
            kernels2, biases2, PReLU(alphas2), ResidualArg{ fptr, featW, featH, featC, fpitch, 1.0f }
        );
    }
}
