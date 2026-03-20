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

    template<int pad, typename T, int N>
    __device__ static inline const T* getReadPtrFromShared(const T(* const sharedPtr)[N], const int c, const int tx, const int ty)
    {
        return sharedPtr[pad + ty] + (pad + tx) * c;
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
        template <typename IN, int cin, int cout, bool postactive = false, typename ActiveFunc, typename... ResidualArgs>
        __global__ void conv1x1_cuda(
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
            constexpr auto knum = cout * 1 * cin;
            constexpr auto bnum = cout;

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
                if (remain > 0 && tid < remain)
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
                if (remain > 0 && tid < remain)
                {
                    auto idx = tid + line * threads;
                    bptr[idx] = biases[idx];
                }
            }
            else if (tid < bnum) bptr[tid] = biases[tid];

            __syncthreads();

            if (x >= srcW || y >= srcH) return;

            auto out = getWritePtr<half>(dptr, dstW, dstH, dstC, dpitch, x, y);

            const IN* r = getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, x, y);

            for (int n = 0; n < cout; n++)
            {
                const float* k = kptr + n * cin;

                float sum = bptr[n];

                for (int c = 0; c < cin; c++) sum += toFloat(r[c]) * k[c];

                if constexpr (!postactive) sum = activeFunc(sum, n);

                if constexpr (sizeof...(ResidualArgs))
                    for (int idx = 0; idx < sizeof...(ResidualArgs); idx++)
                        sum = sum * residualArgs[idx].scale + toFloat(getReadPtr<half>(residualArgs[idx].ptr, residualArgs[idx].w, residualArgs[idx].h, residualArgs[idx].c, residualArgs[idx].pitch, x, y)[n]);

                if constexpr (postactive) sum = activeFunc(sum, n);

                out[n] = toHalf(sum);
            }
        }
        template <typename IN, int cin, int cout, bool postactive = false, typename ActiveFunc, typename... ResidualArgs>
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
            constexpr auto pad = 1;
            constexpr auto rsizeY = BlockSize::y + pad * 2;
            constexpr auto rsizeX = (BlockSize::x + pad * 2) * cin;

            constexpr bool imageToShared = rsizeY * rsizeX * sizeof(IN) <= 8192;

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
                if (remain > 0 && tid < remain)
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
                if (remain > 0 && tid < remain)
                {
                    auto idx = tid + line * threads;
                    bptr[idx] = biases[idx];
                }
            }
            else if (tid < bnum) bptr[tid] = biases[tid];

            ::cuda::std::conditional_t<imageToShared, const IN(*)[rsizeX], const void* __restrict__> iptr{};
            if constexpr (imageToShared)
            {
                __shared__ IN ibuffer[rsizeY][rsizeX];
                for (int j = 0; j < rsizeY; j++)
                {
                    if constexpr (threads < rsizeX)
                    {
                        constexpr auto line = rsizeX / threads;
                        constexpr auto remain = rsizeX % threads;
                        for (int i = 0; i < line; i++)
                        {
                            auto idx = tid + i * threads;
                            ibuffer[j][idx] = getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, bx - pad + idx / cin, by - pad + j)[idx % cin];
                        }
                        if (remain > 0 && tid < remain)
                        {
                            auto idx = tid + line * threads;
                            ibuffer[j][idx] = getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, bx - pad + idx / cin, by - pad + j)[idx % cin];
                        }
                    }
                    else if (tid < rsizeX) ibuffer[j][tid] = getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, bx - pad + tid / cin, by - pad + j)[tid % cin];
                }
                iptr = ibuffer;
            }
            else iptr = sptr;

            __syncthreads();

            if (x >= srcW || y >= srcH) return;

            auto out = getWritePtr<half>(dptr, dstW, dstH, dstC, dpitch, x, y);

            const IN* r[9]{};
            if constexpr (imageToShared)
            {
                r[0] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x - 1, threadIdx.y - 1);
                r[1] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x    , threadIdx.y - 1);
                r[2] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x + 1, threadIdx.y - 1);
                r[3] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x - 1, threadIdx.y    );
                r[4] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x    , threadIdx.y    );
                r[5] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x + 1, threadIdx.y    );
                r[6] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x - 1, threadIdx.y + 1);
                r[7] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x    , threadIdx.y + 1);
                r[8] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x + 1, threadIdx.y + 1);
            }
            else
            {
                r[0] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x - 1, y - 1);
                r[1] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x    , y - 1);
                r[2] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x + 1, y - 1);
                r[3] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x - 1, y    );
                r[4] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x    , y    );
                r[5] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x + 1, y    );
                r[6] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x - 1, y + 1);
                r[7] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x    , y + 1);
                r[8] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x + 1, y + 1);
            }

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

                if constexpr (!postactive) sum = activeFunc(sum, n);

                if constexpr (sizeof...(ResidualArgs))
                    for (int idx = 0; idx < sizeof...(ResidualArgs); idx++)
                        sum = sum * residualArgs[idx].scale + toFloat(getReadPtr<half>(residualArgs[idx].ptr, residualArgs[idx].w, residualArgs[idx].h, residualArgs[idx].c, residualArgs[idx].pitch, x, y)[n]);

                if constexpr (postactive) sum = activeFunc(sum, n);

                out[n] = toHalf(sum);
            }
        }
        template <typename IN, int cin, int cout, bool postactive = false, typename ActiveFunc, typename... ResidualArgs>
        __global__ void conv5x5_cuda(
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
            constexpr auto knum = cout * 25 * cin;
            constexpr auto bnum = cout;
            constexpr auto pad = 2;
            constexpr auto rsizeY = BlockSize::y + pad * 2;
            constexpr auto rsizeX = (BlockSize::x + pad * 2) * cin;

            constexpr bool imageToShared = rsizeY * rsizeX * sizeof(IN) <= 8192;

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
                if (remain > 0 && tid < remain)
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
                if (remain > 0 && tid < remain)
                {
                    auto idx = tid + line * threads;
                    bptr[idx] = biases[idx];
                }
            }
            else if (tid < bnum) bptr[tid] = biases[tid];

            ::cuda::std::conditional_t<imageToShared, const IN(*)[rsizeX], const void* __restrict__> iptr{};
            if constexpr (imageToShared)
            {
                __shared__ IN ibuffer[rsizeY][rsizeX];
                for (int j = 0; j < rsizeY; j++)
                {
                    if constexpr (threads < rsizeX)
                    {
                        constexpr auto line = rsizeX / threads;
                        constexpr auto remain = rsizeX % threads;
                        for (int i = 0; i < line; i++)
                        {
                            auto idx = tid + i * threads;
                            ibuffer[j][idx] = getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, bx - pad + idx / cin, by - pad + j)[idx % cin];
                        }
                        if (remain > 0 && tid < remain)
                        {
                            auto idx = tid + line * threads;
                            ibuffer[j][idx] = getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, bx - pad + idx / cin, by - pad + j)[idx % cin];
                        }
                    }
                    else if (tid < rsizeX) ibuffer[j][tid] = getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, bx - pad + tid / cin, by - pad + j)[tid % cin];
                }
                iptr = ibuffer;
            }
            else iptr = sptr;

            __syncthreads();

            if (x >= srcW || y >= srcH) return;

            auto out = getWritePtr<half>(dptr, dstW, dstH, dstC, dpitch, x, y);

            for (int n = 0; n < cout; n++)
            {
                float sum = bptr[n];

                for (int in = 0; in < 5; in++)
                {
                    for (int jn = 0; jn < 5; jn++)
                    {
                        const IN* r{};
                        if constexpr (imageToShared)
                            r = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x + jn - 2, threadIdx.y + in - 2);
                        else
                            r = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x + jn - 2, y + in - 2);

                        auto k = kptr + n * cin * 25 + cin * (in * 5 + jn);

                        for (int c = 0; c < cin; c++) sum += toFloat(r[c]) * k[c];

                    }
                }

                if constexpr (!postactive) sum = activeFunc(sum, n);

                if constexpr (sizeof...(ResidualArgs))
                    for (int idx = 0; idx < sizeof...(ResidualArgs); idx++)
                        sum = sum * residualArgs[idx].scale + toFloat(getReadPtr<half>(residualArgs[idx].ptr, residualArgs[idx].w, residualArgs[idx].h, residualArgs[idx].c, residualArgs[idx].pitch, x, y)[n]);

                if constexpr (postactive) sum = activeFunc(sum, n);

                out[n] = toHalf(sum);
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

        template <typename IN, typename OUT, int cin, int upscale>
        __global__ void conv3x3_identity_pixelshuffle_cuda(
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

            constexpr auto cout = upscale * upscale;
            constexpr auto threads = BlockSize::x * BlockSize::y;
            constexpr auto knum = cout * 9 * cin;
            constexpr auto bnum = cout;
            constexpr auto pad = 1;
            constexpr auto rsizeY = BlockSize::y + pad * 2;
            constexpr auto rsizeX = (BlockSize::x + pad * 2) * cin;

            constexpr bool imageToShared = rsizeY * rsizeX * sizeof(IN) <= 8192;

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
                if (remain > 0 && tid < remain)
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
                if (remain > 0 && tid < remain)
                {
                    auto idx = tid + line * threads;
                    bptr[idx] = biases[idx];
                }
            }
            else if (tid < bnum) bptr[tid] = biases[tid];

            ::cuda::std::conditional_t<imageToShared, const IN(*)[rsizeX], const void* __restrict__> iptr{};
            if constexpr (imageToShared)
            {
                __shared__ IN ibuffer[rsizeY][rsizeX];
                for (int j = 0; j < rsizeY; j++)
                {
                    if constexpr (threads < rsizeX)
                    {
                        constexpr auto line = rsizeX / threads;
                        constexpr auto remain = rsizeX % threads;
                        for (int i = 0; i < line; i++)
                        {
                            auto idx = tid + i * threads;
                            ibuffer[j][idx] = getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, bx - pad + idx / cin, by - pad + j)[idx % cin];
                        }
                        if (remain > 0 && tid < remain)
                        {
                            auto idx = tid + line * threads;
                            ibuffer[j][idx] = getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, bx - pad + idx / cin, by - pad + j)[idx % cin];
                        }
                    }
                    else if (tid < rsizeX) ibuffer[j][tid] = getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, bx - pad + tid / cin, by - pad + j)[tid % cin];
                }
                iptr = ibuffer;
            }
            else iptr = sptr;

            __syncthreads();

            if (x >= srcW || y >= srcH) return;

            auto out = getWritePtr<half>(dptr, dstW, dstH, dstC, dpitch, x, y);

            const IN* r[9]{};
            if constexpr (imageToShared)
            {
                r[0] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x - 1, threadIdx.y - 1);
                r[1] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x    , threadIdx.y - 1);
                r[2] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x + 1, threadIdx.y - 1);
                r[3] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x - 1, threadIdx.y    );
                r[4] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x    , threadIdx.y    );
                r[5] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x + 1, threadIdx.y    );
                r[6] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x - 1, threadIdx.y + 1);
                r[7] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x    , threadIdx.y + 1);
                r[8] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x + 1, threadIdx.y + 1);
            }
            else
            {
                r[0] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x - 1, y - 1);
                r[1] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x    , y - 1);
                r[2] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x + 1, y - 1);
                r[3] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x - 1, y    );
                r[4] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x    , y    );
                r[5] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x + 1, y    );
                r[6] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x - 1, y + 1);
                r[7] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x    , y + 1);
                r[8] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x + 1, y + 1);
            }

            auto dstX = x * upscale;
            auto dstY = y * upscale;

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

                *getWritePtr<OUT>(dptr, dstW, dstH, dstC, dpitch, dstX + (n & 1), dstY + (n >> 1)) = fromFloat<OUT>(sum);
            }
        }
    
        template <typename IN, int cin, int ctemp, int cout, bool postactive3x3 = false, bool postactive1x1 = false, typename ActiveFunc3x3, typename ResidualArgs3x3, typename ActiveFunc1x1, typename ResidualArgs1x1>
        __global__ void conv3x3_conv1x1_cuda(
            const void* const __restrict__ sptr,
            const int srcW, const int srcH, const int srcC, const int spitch,
            const typename RestrictPointer<::cuda::std::is_same_v<ResidualArgs3x3, ResidualArg>, void>::Type dptr,
            const int dstW, const int dstH, const int dstC, const int dpitch,
            const float* const __restrict__ kernels3x3, const float* const __restrict__ biases3x3, ActiveFunc3x3 activeFunc3x3, ResidualArgs3x3 residualArg3x3,
            const float* const __restrict__ kernels1x1, const float* const __restrict__ biases1x1, ActiveFunc1x1 activeFunc1x1, ResidualArgs1x1 residualArg1x1)
        {
            auto bx = blockIdx.x * BlockSize::x;
            auto by = blockIdx.y * BlockSize::y;
            auto x = bx + threadIdx.x;
            auto y = by + threadIdx.y;
            auto tid = threadIdx.y * BlockSize::x + threadIdx.x;

            constexpr auto threads = BlockSize::x * BlockSize::y;
            constexpr auto knum3x3 = ctemp * 9 * cin;
            constexpr auto bnum3x3 = ctemp;
            constexpr auto knum1x1 = cout * 1 * ctemp;
            constexpr auto bnum1x1 = cout;
            constexpr auto ksize = knum3x3 > knum1x1 ? knum3x3 : knum1x1;
            constexpr auto bsize = bnum3x3 > bnum1x1 ? bnum3x3 : bnum1x1;
            constexpr auto pad = 1;
            constexpr auto rsizeY = BlockSize::y + pad * 2;
            constexpr auto rsizeX = (BlockSize::x + pad * 2) * cin;

            constexpr bool imageToShared = rsizeY * rsizeX * sizeof(IN) <= 8192;

            __shared__ float kptr[ksize];
            if constexpr (threads < knum3x3)
            {
                constexpr auto line = knum3x3 / threads;
                constexpr auto remain = knum3x3 % threads;
                for (int i = 0; i < line; i++)
                {
                    auto idx = tid + i * threads;
                    kptr[idx] = kernels3x3[idx];
                }
                if (remain > 0 && tid < remain)
                {
                    auto idx = tid + line * threads;
                    kptr[idx] = kernels3x3[idx];
                }
            }
            else if (tid < knum3x3) kptr[tid] = kernels3x3[tid];

            __shared__ float bptr[bsize];
            if constexpr (threads < bnum3x3)
            {
                constexpr auto line = bnum3x3 / threads;
                constexpr auto remain = bnum3x3 % threads;
                for (int i = 0; i < line; i++)
                {
                    auto idx = tid + i * threads;
                    bptr[idx] = biases3x3[idx];
                }
                if (remain > 0 && tid < remain)
                {
                    auto idx = tid + line * threads;
                    bptr[idx] = biases3x3[idx];
                }
            }
            else if (tid < bnum3x3) bptr[tid] = biases3x3[tid];

            ::cuda::std::conditional_t<imageToShared, const IN(*)[rsizeX], const void* __restrict__> iptr{};
            if constexpr (imageToShared)
            {
                __shared__ IN ibuffer[rsizeY][rsizeX];
                for (int j = 0; j < rsizeY; j++)
                {
                    if constexpr (threads < rsizeX)
                    {
                        constexpr auto line = rsizeX / threads;
                        constexpr auto remain = rsizeX % threads;
                        for (int i = 0; i < line; i++)
                        {
                            auto idx = tid + i * threads;
                            ibuffer[j][idx] = getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, bx - pad + idx / cin, by - pad + j)[idx % cin];
                        }
                        if (remain > 0 && tid < remain)
                        {
                            auto idx = tid + line * threads;
                            ibuffer[j][idx] = getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, bx - pad + idx / cin, by - pad + j)[idx % cin];
                        }
                    }
                    else if (tid < rsizeX) ibuffer[j][tid] = getReadPtr<IN>(sptr, srcW, srcH, srcC, spitch, bx - pad + tid / cin, by - pad + j)[tid % cin];
                }
                iptr = ibuffer;
            }
            else iptr = sptr;

            __syncthreads();

            if (x >= srcW || y >= srcH) return;

            auto out = getWritePtr<half>(dptr, dstW, dstH, dstC, dpitch, x, y);

            half buffer[ctemp]{};

            const IN* r[9]{};
            if constexpr (imageToShared)
            {
                r[0] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x - 1, threadIdx.y - 1);
                r[1] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x    , threadIdx.y - 1);
                r[2] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x + 1, threadIdx.y - 1);
                r[3] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x - 1, threadIdx.y    );
                r[4] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x    , threadIdx.y    );
                r[5] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x + 1, threadIdx.y    );
                r[6] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x - 1, threadIdx.y + 1);
                r[7] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x    , threadIdx.y + 1);
                r[8] = getReadPtrFromShared<pad>(iptr, cin, threadIdx.x + 1, threadIdx.y + 1);
            }
            else
            {
                r[0] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x - 1, y - 1);
                r[1] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x    , y - 1);
                r[2] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x + 1, y - 1);
                r[3] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x - 1, y    );
                r[4] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x    , y    );
                r[5] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x + 1, y    );
                r[6] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x - 1, y + 1);
                r[7] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x    , y + 1);
                r[8] = getReadPtr<IN>(iptr, srcW, srcH, srcC, spitch, x + 1, y + 1);
            }

            for (int n = 0; n < ctemp; n++)
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

                if constexpr (!postactive3x3) sum = activeFunc3x3(sum, n);

                if constexpr (::cuda::std::is_same_v<ResidualArgs3x3, ResidualArg>)
                    sum = sum * residualArg3x3.scale + toFloat(getReadPtr<half>(residualArg3x3.ptr, residualArg3x3.w, residualArg3x3.h, residualArg3x3.c, residualArg3x3.pitch, x, y)[n]);

                if constexpr (postactive3x3) sum = activeFunc3x3(sum, n);

                buffer[n] = toHalf(sum);
            }

            if constexpr (threads < knum1x1)
            {
                constexpr auto line = knum1x1 / threads;
                constexpr auto remain = knum1x1 % threads;
                for (int i = 0; i < line; i++)
                {
                    auto idx = tid + i * threads;
                    kptr[idx] = kernels1x1[idx];
                }
                if (remain > 0 && tid < remain)
                {
                    auto idx = tid + line * threads;
                    kptr[idx] = kernels1x1[idx];
                }
            }
            else if (tid < knum1x1) kptr[tid] = kernels1x1[tid];

            if constexpr (threads < bnum1x1)
            {
                constexpr auto line = bnum1x1 / threads;
                constexpr auto remain = bnum1x1 % threads;
                for (int i = 0; i < line; i++)
                {
                    auto idx = tid + i * threads;
                    bptr[idx] = biases1x1[idx];
                }
                if (remain > 0 && tid < remain)
                {
                    auto idx = tid + line * threads;
                    bptr[idx] = biases1x1[idx];
                }
            }
            else if (tid < bnum1x1) bptr[tid] = biases1x1[tid];

            __syncthreads();

            for (int n = 0; n < cout; n++)
            {
                const float* k = kptr + n * ctemp;

                float sum = bptr[n];

                for (int c = 0; c < ctemp; c++) sum += toFloat(buffer[c]) * k[c];

                if constexpr (!postactive1x1) sum = activeFunc1x1(sum, n);

                if constexpr (::cuda::std::is_same_v<ResidualArgs1x1, ResidualArg>)
                    sum = sum * residualArg1x1.scale + toFloat(getReadPtr<half>(residualArg1x1.ptr, residualArg1x1.w, residualArg1x1.h, residualArg1x1.c, residualArg1x1.pitch, x, y)[n]);

                if constexpr (postactive1x1) sum = activeFunc1x1(sum, n);

                out[n] = toHalf(sum);
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
    void conv3x3_8to8_identity_residual_cuda(
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
    void conv3x3_8to8_identity_residual_add_cuda(
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
            return kernel::conv3x3_identity_pixelshuffle_cuda<half, ::cuda::std::uint8_t, 8, 2> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases);
        case Image::UInt16:
            return kernel::conv3x3_identity_pixelshuffle_cuda<half, ::cuda::std::uint16_t, 8, 2> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases);
        case Image::Float32:
            return kernel::conv3x3_identity_pixelshuffle_cuda<half, float, 8, 2> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases);
        }
    }

    void conv3x3_1to16_identity_cuda(
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
            kernel::conv3x3_cuda<::cuda::std::uint8_t, 1, 16> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity());
            break;
        case Image::UInt16:
            kernel::conv3x3_cuda<::cuda::std::uint16_t, 1, 16> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity());
            break;
        case Image::Float32:
            kernel::conv3x3_cuda<float, 1, 16> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_16to16_relu_cuda(
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
        kernel::conv3x3_cuda<half, 16, 16> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, ReLU());
    }
    void conv3x3_16to16_identity_add_cuda(
        const void* sptr,
        int srcW, int srcH, int srcC, int spitch,
        void* dptr,
        int dstW, int dstH, int dstC, int dpitch,
        const float* kernels,
        const float* biases,
        void* fptr,
        int featW, int featH, int featC, int fpitch,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        kernel::conv3x3_cuda<half, 16, 16> <<< grid, block, 0, stream >>> (
            sptr, srcW, srcH, srcC, spitch,
            dptr, dstW, dstH, dstC, dpitch,
            kernels, biases,
            Identity(),
            ResidualArg{ fptr, featW, featH, featC, fpitch, 1.0f });
    }
    void conv3x3_16to4_identity_pixelshuffle_4to1_cuda(
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
            return kernel::conv3x3_identity_pixelshuffle_cuda<half, ::cuda::std::uint8_t, 16, 2> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases);
        case Image::UInt16:
            return kernel::conv3x3_identity_pixelshuffle_cuda<half, ::cuda::std::uint16_t, 16, 2> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases);
        case Image::Float32:
            return kernel::conv3x3_identity_pixelshuffle_cuda<half, float, 16, 2> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases);
        }
    }

    void conv3x3_1to32_identity_cuda(
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
            kernel::conv3x3_cuda<::cuda::std::uint8_t, 1, 32> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity());
            break;
        case Image::UInt16:
            kernel::conv3x3_cuda<::cuda::std::uint16_t, 1, 32> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity());
            break;
        case Image::Float32:
            kernel::conv3x3_cuda<float, 1, 32> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_32to32_relu_cuda(
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
        kernel::conv3x3_cuda<half, 32, 32> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, ReLU());
    }
    void conv3x3_32to32_identity_add_cuda(
        const void* sptr,
        int srcW, int srcH, int srcC, int spitch,
        void* dptr,
        int dstW, int dstH, int dstC, int dpitch,
        const float* kernels,
        const float* biases,
        void* fptr,
        int featW, int featH, int featC, int fpitch,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        kernel::conv3x3_cuda<half, 32, 32> <<< grid, block, 0, stream >>> (
            sptr, srcW, srcH, srcC, spitch,
            dptr, dstW, dstH, dstC, dpitch,
            kernels, biases,
            Identity(),
            ResidualArg{ fptr, featW, featH, featC, fpitch, 1.0f });
    }
    void conv3x3_32to4_identity_pixelshuffle_4to1_cuda(
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
            return kernel::conv3x3_identity_pixelshuffle_cuda<half, ::cuda::std::uint8_t, 32, 2> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases);
        case Image::UInt16:
            return kernel::conv3x3_identity_pixelshuffle_cuda<half, ::cuda::std::uint16_t, 32, 2> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases);
        case Image::Float32:
            return kernel::conv3x3_identity_pixelshuffle_cuda<half, float, 32, 2> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases);
        }
    }

    void conv5x5_1to8_identity_cuda(
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
            kernel::conv5x5_cuda<::cuda::std::uint8_t, 1, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity());
            break;
        case Image::UInt16:
            kernel::conv5x5_cuda<::cuda::std::uint16_t, 1, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity());
            break;
        case Image::Float32:
            kernel::conv5x5_cuda<float, 1, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_8to8_prelu_cuda(
        const void* sptr,
        int srcW, int srcH, int srcC, int spitch,
        void* dptr,
        int dstW, int dstH, int dstC, int dpitch,
        const float* kernels,
        const float* biases,
        const float* alphas,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        kernel::conv3x3_cuda<half, 8, 8> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, PReLU(alphas));
    }
    void conv3x3_8to8_prelu_conv1x1_8to8_add_prelu_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        void* fptr, int featW, int featH, int featC, int fpitch,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        kernel::conv3x3_conv1x1_cuda<half, 8, 8, 8, false, true> <<< grid, block, 0, stream >>> (
            sptr, srcW, srcH, srcC, spitch,
            dptr, dstW, dstH, dstC, dpitch,
            kernels1, biases1, PReLU(alphas1), nullptr,
            kernels2, biases2, PReLU(alphas2), ResidualArg{ fptr, featW, featH, featC, fpitch, 1.0f }
        );
    }

    void conv5x5_1to16_identity_cuda(
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
            kernel::conv5x5_cuda<::cuda::std::uint8_t, 1, 16> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity());
            break;
        case Image::UInt16:
            kernel::conv5x5_cuda<::cuda::std::uint16_t, 1, 16> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity());
            break;
        case Image::Float32:
            kernel::conv5x5_cuda<float, 1, 16> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, Identity());
            break;
        }
    }
    void conv3x3_16to16_prelu_cuda(
        const void* sptr,
        int srcW, int srcH, int srcC, int spitch,
        void* dptr,
        int dstW, int dstH, int dstC, int dpitch,
        const float* kernels,
        const float* biases,
        const float* alphas,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        kernel::conv3x3_cuda<half, 16, 16> <<< grid, block, 0, stream >>> (sptr, srcW, srcH, srcC, spitch, dptr, dstW, dstH, dstC, dpitch, kernels, biases, PReLU(alphas));
    }
    void conv3x3_16to16_prelu_conv1x1_16to16_add_prelu_cuda(
        const void* sptr, int srcW, int srcH, int srcC, int spitch,
        void* dptr, int dstW, int dstH, int dstC, int dpitch,
        const float* kernels1, const float* biases1, const float* alphas1,
        const float* kernels2, const float* biases2, const float* alphas2,
        void* fptr, int featW, int featH, int featC, int fpitch,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ BlockSize::x, BlockSize::y };
        dim3 grid{ (srcW + block.x - 1) / block.x, (srcH + block.y - 1) / block.y };
        kernel::conv3x3_conv1x1_cuda<half, 16, 16, 16, false, true> <<< grid, block, 0, stream >>> (
            sptr, srcW, srcH, srcC, spitch,
            dptr, dstW, dstH, dstC, dpitch,
            kernels1, biases1, PReLU(alphas1), nullptr,
            kernels2, biases2, PReLU(alphas2), ResidualArg{ fptr, featW, featH, featC, fpitch, 1.0f }
        );
    }
}
