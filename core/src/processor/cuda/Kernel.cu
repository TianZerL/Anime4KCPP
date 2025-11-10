#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda_fp16.h>

#include "AC/Core/Image.hpp"

namespace ac::core::cuda
{
    template<typename Float, ::cuda::std::enable_if_t<::cuda::std::is_floating_point_v<Float>, bool> = true>
    static __inline__ __device__ Float fromFloat(const float v) noexcept
    {
        return __saturatef(v);
    }
    template<typename Unsigned, ::cuda::std::enable_if_t<::cuda::std::is_unsigned_v<Unsigned>, bool> = true>
    static __inline__ __device__ Unsigned fromFloat(const float v) noexcept
    {
        return static_cast<Unsigned>(fromFloat<float>(v) * ::cuda::std::numeric_limits<Unsigned>::max() + 0.5f);
    }
    template<typename T>
    static __inline__ __device__ void writeImage(const T val, const cudaSurfaceObject_t dst, const int x, const int y) noexcept
    {
        surf2Dwrite<T>(val, dst, sizeof(T) * x, y, cudaBoundaryModeZero);
    }
    static __inline__ __device__ float dot(const float4 a, const float4 b) noexcept
    {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    static __inline__ __device__ float dot(const float4 a, const float* const __restrict__ b) noexcept
    {
        return dot(a, make_float4(b[0], b[1], b[2], b[3]));
    }
    static __inline__ __device__ ushort4 readImageh(const cudaSurfaceObject_t src, const int x, const int y, const int layer) noexcept
    {
        return surf2DLayeredread<ushort4>(src, sizeof(ushort4) * x, y, layer, cudaBoundaryModeZero);
    }
    static __inline__ __device__ void writeImageh(const ushort4 val, const cudaSurfaceObject_t dst, const int x, const int y, const int layer) noexcept
    {
        surf2DLayeredwrite<ushort4>(val, dst, sizeof(ushort4) * x, y, layer, cudaBoundaryModeZero);
    }
    static __inline__ __device__ float4 half2Float(const ushort4 a) noexcept
    {
        return make_float4(__ushort_as_half(a.x), __ushort_as_half(a.y), __ushort_as_half(a.z), __ushort_as_half(a.w));
    }
    static __inline__ __device__ float4 readImagef(const cudaSurfaceObject_t src, const int x, const int y, const int layer) noexcept
    {
        return half2Float(readImageh(src, x, y, layer));
    }
#if __CUDA_ARCH__ < 700
    static __inline__ __device__ ushort4 float2Half(const float4 a) noexcept
    {
        return make_ushort4(__half_as_ushort(a.x), __half_as_ushort(a.y), __half_as_ushort(a.z), __half_as_ushort(a.w));
    }
    static __inline__ __device__ void writeImagef(const float4 val, const cudaSurfaceObject_t dst, const int x, const int y, const int layer) noexcept
    {
        writeImageh(float2Half(val), dst, x, y, layer);
    }
    static __inline__ __device__ float4& operator+=(float4& a, const float4& b) noexcept
    {
        a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
        return a;
    }
    static __inline__ __device__ float relu(const float a) noexcept
    {
        return fmaxf(a, 0.0f);
    }
    static __inline__ __device__ float4 relu(const float4 a) noexcept
    {
        return make_float4(relu(a.x), relu(a.y), relu(a.z), relu(a.w));
    }
#else
    struct __align__(8) Half4
    {
        half2 v0, v1;

        Half4() noexcept = default;
        __inline__ __device__ Half4(const half d) noexcept
        {
            make(d);
        }
        __inline__ __device__ Half4(const half d0, const half d1, const half d2, const half d3) noexcept
        {
            make(d0, d1, d2, d3);
        }
        __inline__ __device__ Half4(const half2 d0, const half2 d1) noexcept
        {
            make(d0, d1);
        }
        __inline__ __device__ Half4(const ushort4 d) noexcept
        {
            make(d);
        }
        __inline__ __device__ Half4(const float4 d) noexcept
        {
            make(d);
        }

        __inline__ __device__ void make(const half d) noexcept
        {
            v0 = make_half2(d, d);
            v1 = make_half2(d, d);
        }
        __inline__ __device__ void make(const half d0, const half d1, const half d2, const half d3) noexcept
        {
            v0 = make_half2(d0, d1);
            v1 = make_half2(d2, d3);
        }
        __inline__ __device__ void make(const half2 d0, const half2 d1) noexcept
        {
            v0 = d0;
            v1 = d1;
        }
        __inline__ __device__ void make(const ushort4 d) noexcept
        {
            v0 = make_half2(__ushort_as_half(d.x), __ushort_as_half(d.y));
            v1 = make_half2(__ushort_as_half(d.z), __ushort_as_half(d.w));
        }
        __inline__ __device__ void make(const float4 d) noexcept
        {
            v0 = __floats2half2_rn(d.x, d.y);
            v1 = __floats2half2_rn(d.z, d.w);
        }
        __inline__ __device__ Half4& operator=(const ushort4 d) noexcept
        {
            make(d);
            return *this;
        }
        __inline__ __device__ Half4& operator+=(const Half4& d) noexcept
        {
            v0 = __hadd2(v0, d.v0);
            v1 = __hadd2(v1, d.v1);
            return *this;
        }
        __inline__ __device__ operator ushort4() const noexcept
        {
            return make_ushort4(__half_as_ushort(v0.x), __half_as_ushort(v0.y), __half_as_ushort(v1.x), __half_as_ushort(v1.y));
        }
        __inline__ __device__ half dot(const half* const __restrict__ d) const noexcept
        {
            return dot(make_half2(d[0], d[1]), make_half2(d[2], d[3]));
        }
        __inline__ __device__ half dot(const half2* const __restrict__ d) const noexcept
        {
            return dot(d[0], d[1]);
        }
        __inline__ __device__ half dot(const Half4& d) const noexcept
        {
            return dot(d.v0, d.v1);
        }
        __inline__ __device__ half dot(const half2 d0, const half2 d1) const noexcept
        {
            auto s = __hfma2(v0, d0, __hmul2(v1, d1));
            return s.x + s.y;
        }
    };

    static __inline__ __device__ half2 relu(const half2 a) noexcept
    {
        return __hmax2(a, make_half2(0, 0));
    }
    static __inline__ __device__ Half4 relu(const Half4& a) noexcept
    {
        return Half4{ relu(a.v0), relu(a.v1) };
    }
#endif

    template<int cout,
        ::cuda::std::enable_if_t<cout % 4 == 0, bool> = true>
    __global__ void conv3x3_cuda_cin1_float(
        cudaTextureObject_t src,
        cudaSurfaceObject_t dst,
        const unsigned int width,
        const unsigned int height,
        const float* const __restrict__ kernels,
        const float* const __restrict__ biases
    )
    {
#   if __CUDA_ARCH__ < 700
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;
        auto tid = threadIdx.y * blockDim.x + threadIdx.x;
        auto threads = blockDim.x * blockDim.y;

        constexpr auto knum = cout * 9;
        constexpr auto bnum = cout;
        __shared__ __align__(16) float kptr[knum];
        if (threads < knum)
        {
            auto line = knum / threads;
            auto remain = knum % threads;
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
        __shared__ __align__(16) float bptr[bnum];
        if (threads < bnum)
        {
            auto line = bnum / threads;
            auto remain = bnum % threads;
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
        __syncthreads();

        if (x >= width || y >= height) return;

        constexpr int lout = cout / 4;

        const float r[] = {
            tex2D<float>(src, x - 1, y - 1),
            tex2D<float>(src, x    , y - 1),
            tex2D<float>(src, x + 1, y - 1),
            tex2D<float>(src, x - 1, y    ),
            tex2D<float>(src, x    , y    ),
            tex2D<float>(src, x + 1, y    ),
            tex2D<float>(src, x - 1, y + 1),
            tex2D<float>(src, x    , y + 1),
            tex2D<float>(src, x + 1, y + 1)
        };

        for (int nidx = 0; nidx < lout; nidx++)
        {
            auto npos = nidx * 4;
            float4 s = make_float4(bptr[npos + 0], bptr[npos + 1], bptr[npos + 2], bptr[npos + 3]);
            for (int i = 0; i < 9; i++)
            {
                s.x += r[i] * kptr[(npos + 0) * 9 + i];
                s.y += r[i] * kptr[(npos + 1) * 9 + i];
                s.z += r[i] * kptr[(npos + 2) * 9 + i];
                s.w += r[i] * kptr[(npos + 3) * 9 + i];
            }

            writeImagef(relu(s), dst, x, y, nidx);
        }
#   endif
    }

    template<int cin, int cout, bool residual = false,
        ::cuda::std::enable_if_t<(cin % 4 == 0) && (cout % 4 == 0), bool> = true>
    __global__ void conv3x3_cuda_float(
        cudaSurfaceObject_t src,
        cudaSurfaceObject_t dst,
        const unsigned int width,
        const unsigned int height,
        const float* const __restrict__ kernels,
        const float* const __restrict__ biases
    )
    {
#   if __CUDA_ARCH__ < 700
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;
        auto tid = threadIdx.y * blockDim.x + threadIdx.x;
        auto threads = blockDim.x * blockDim.y;

        constexpr auto knum = cout * 9 * cin;
        constexpr auto bnum = cout;
        __shared__ __align__(16) float kptr[knum];
        if (threads < knum)
        {
            auto line = knum / threads;
            auto remain = knum % threads;
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
        __shared__ __align__(16) float bptr[bnum];
        if (threads < bnum)
        {
            auto line = bnum / threads;
            auto remain = bnum % threads;
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
        __syncthreads();

        if (x >= width || y >= height) return;

        constexpr int lin = cin / 4;
        constexpr int lout = cout / 4;

        float4 r[9][lin]{};

        for (int cidx = 0; cidx < lin; cidx++)
        {
            r[0][cidx] = readImagef(src, x - 1, y - 1, cidx);
            r[1][cidx] = readImagef(src, x    , y - 1, cidx);
            r[2][cidx] = readImagef(src, x + 1, y - 1, cidx);
            r[3][cidx] = readImagef(src, x - 1, y    , cidx);
            r[4][cidx] = readImagef(src, x    , y    , cidx);
            r[5][cidx] = readImagef(src, x + 1, y    , cidx);
            r[6][cidx] = readImagef(src, x - 1, y + 1, cidx);
            r[7][cidx] = readImagef(src, x    , y + 1, cidx);
            r[8][cidx] = readImagef(src, x + 1, y + 1, cidx);
        };

        for (int nidx = 0; nidx < lout; nidx++)
        {
            auto npos = nidx * 4;
            float4 s = make_float4(bptr[npos + 0], bptr[npos + 1], bptr[npos + 2], bptr[npos + 3]);
            if constexpr (residual) s += readImagef(dst, x, y, nidx);

            for (int i = 0; i < 9; i++)
                for (int cidx = 0; cidx < lin; cidx++)
                {
                    s.x += dot(r[i][cidx], kptr + (npos + 0) * 9 * cin + i * cin + cidx * 4);
                    s.y += dot(r[i][cidx], kptr + (npos + 1) * 9 * cin + i * cin + cidx * 4);
                    s.z += dot(r[i][cidx], kptr + (npos + 2) * 9 * cin + i * cin + cidx * 4);
                    s.w += dot(r[i][cidx], kptr + (npos + 3) * 9 * cin + i * cin + cidx * 4);
                }

            writeImagef(relu(s), dst, x, y, nidx);
        }
#   endif
    }

    template<int cout,
        ::cuda::std::enable_if_t<cout % 4 == 0, bool> = true>
    __global__ void conv3x3_cuda_cin1_half(
        cudaTextureObject_t src,
        cudaSurfaceObject_t dst,
        const unsigned int width,
        const unsigned int height,
        const float* const __restrict__ kernels,
        const float* const __restrict__ biases
    )
    {
#   if __CUDA_ARCH__ >= 700
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;
        auto tid = threadIdx.y * blockDim.x + threadIdx.x;
        auto threads = blockDim.x * blockDim.y;

        constexpr auto knum = cout * 9 ;
        constexpr auto bnum = cout;
        __shared__ __align__(8) half kptr[knum];
        if (threads < knum)
        {
            auto line = knum / threads;
            auto remain = knum % threads;
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
        __shared__ __align__(8) half bptr[bnum];
        if (threads < bnum)
        {
            auto line = bnum / threads;
            auto remain = bnum % threads;
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
        __syncthreads();

        if (x >= width || y >= height) return;

        constexpr int lout = cout / 4;

        const half r[] = {
            tex2D<float>(src, x - 1, y - 1),
            tex2D<float>(src, x    , y - 1),
            tex2D<float>(src, x + 1, y - 1),
            tex2D<float>(src, x - 1, y    ),
            tex2D<float>(src, x    , y    ),
            tex2D<float>(src, x + 1, y    ),
            tex2D<float>(src, x - 1, y + 1),
            tex2D<float>(src, x    , y + 1),
            tex2D<float>(src, x + 1, y + 1)
        };

        for (int nidx = 0; nidx < lout; nidx++)
        {
            auto npos = nidx * 4;
            Half4 s{ bptr[npos + 0], bptr[npos + 1], bptr[npos + 2], bptr[npos + 3] };
            for (int i = 0; i < 9; i++)
            {
                s.v0.x += r[i] * kptr[(npos + 0) * 9 + i];
                s.v0.y += r[i] * kptr[(npos + 1) * 9 + i];
                s.v1.x += r[i] * kptr[(npos + 2) * 9 + i];
                s.v1.y += r[i] * kptr[(npos + 3) * 9 + i];
            }

            writeImageh(relu(s), dst, x, y, nidx);
        }
#   endif
    }

    template<int cin, int cout, bool residual = false,
        ::cuda::std::enable_if_t<(cin % 4 == 0) && (cout % 4 == 0), bool> = true>
    __global__ void conv3x3_cuda_half(
        cudaSurfaceObject_t src,
        cudaSurfaceObject_t dst,
        const unsigned int width,
        const unsigned int height,
        const float* const __restrict__ kernels,
        const float* const __restrict__ biases
    )
    {
#   if __CUDA_ARCH__ >= 700
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;
        auto tid = threadIdx.y * blockDim.x + threadIdx.x;
        auto threads = blockDim.x * blockDim.y;

        constexpr auto knum = cout * 9 * cin / 2;
        constexpr auto bnum = cout / 2;
        __shared__ __align__(8) half2 kptr[knum];
        if (threads < knum)
        {
            auto line = knum / threads;
            auto remain = knum % threads;
            for (int i = 0; i < line; i++)
            {
                auto idx = tid + i * threads;
                kptr[idx] = __float22half2_rn(reinterpret_cast<const float2*>(kernels)[idx]);
            }
            if (tid < remain)
            {
                auto idx = tid + line * threads;
                kptr[idx] = __float22half2_rn(reinterpret_cast<const float2*>(kernels)[idx]);
            }
        }
        else if (tid < knum) kptr[tid] = __float22half2_rn(reinterpret_cast<const float2*>(kernels)[tid]);
        __shared__ __align__(8) half2 bptr[bnum];
        if (threads < bnum)
        {
            auto line = bnum / threads;
            auto remain = bnum % threads;
            for (int i = 0; i < line; i++)
            {
                auto idx = tid + i * threads;
                bptr[idx] = __float22half2_rn(reinterpret_cast<const float2*>(biases)[idx]);
            }
            if (tid < remain)
            {
                auto idx = tid + line * threads;
                bptr[idx] = __float22half2_rn(reinterpret_cast<const float2*>(biases)[idx]);
            }
        }
        else if (tid < bnum) bptr[tid] = __float22half2_rn(reinterpret_cast<const float2*>(biases)[tid]);
        __syncthreads();

        if (x >= width || y >= height) return;

        constexpr int lin = cin / 4;
        constexpr int lout = cout / 4;

        Half4 r[9][lin]{};

        for (int cidx = 0; cidx < lin; cidx++)
        {
            r[0][cidx] = readImageh(src, x - 1, y - 1, cidx);
            r[1][cidx] = readImageh(src, x    , y - 1, cidx);
            r[2][cidx] = readImageh(src, x + 1, y - 1, cidx);
            r[3][cidx] = readImageh(src, x - 1, y    , cidx);
            r[4][cidx] = readImageh(src, x    , y    , cidx);
            r[5][cidx] = readImageh(src, x + 1, y    , cidx);
            r[6][cidx] = readImageh(src, x - 1, y + 1, cidx);
            r[7][cidx] = readImageh(src, x    , y + 1, cidx);
            r[8][cidx] = readImageh(src, x + 1, y + 1, cidx);
        };

        for (int nidx = 0; nidx < lout; nidx++)
        {
            Half4 s { bptr[nidx * 2 + 0], bptr[nidx * 2 + 1] };
            if constexpr (residual) s += readImageh(dst, x, y, nidx);

            for (int i = 0; i < 9; i++)
                for (int cidx = 0; cidx < lin; cidx++)
                {
                    s.v0.x += r[i][cidx].dot(kptr + ((nidx * 4 + 0) * 9 * cin + i * cin + cidx * 4) / 2);
                    s.v0.y += r[i][cidx].dot(kptr + ((nidx * 4 + 1) * 9 * cin + i * cin + cidx * 4) / 2);
                    s.v1.x += r[i][cidx].dot(kptr + ((nidx * 4 + 2) * 9 * cin + i * cin + cidx * 4) / 2);
                    s.v1.y += r[i][cidx].dot(kptr + ((nidx * 4 + 3) * 9 * cin + i * cin + cidx * 4) / 2);
                }

            writeImageh(relu(s), dst, x, y, nidx);
        }
#   endif
    }

    template<typename OUT, int cin,
        ::cuda::std::enable_if_t<cin % 4 == 0, bool> = true>
    __global__ void deconv2x2_cuda_cout1(
        cudaSurfaceObject_t src,
        cudaSurfaceObject_t dst,
        const unsigned int width,
        const unsigned int height,
        const float* const __restrict__ kernels
    )
    {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) return;

        constexpr int lin = cin / 4;

        const unsigned int index = ((y & 1) << 1) + (x & 1);

        float sum = 0.0f;
        for (int cidx = 0; cidx < lin; cidx++) sum += dot(readImagef(src, x / 2, y / 2, cidx), kernels + cin * index + cidx * 4);
        writeImage(fromFloat<OUT>(sum), dst, x, y);
    }

    void conv3x3_1to8_cuda(
        cudaTextureObject_t src,
        cudaSurfaceObject_t dst,
        unsigned int width,
        unsigned int height,
        const float* kernels,
        const float* biases,
        int computeCapability,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ 16, 8 };
        dim3 grid{ (width + block.x - 1) / block.x, (height + block.y - 1) / block.y };
        if (computeCapability >= 70)
            conv3x3_cuda_cin1_half<8> <<< grid, block, 0, stream >>> (src, dst, width, height, kernels, biases);
        else
            conv3x3_cuda_cin1_float<8> <<< grid, block, 0, stream >>> (src, dst, width, height, kernels, biases);
    }

    void conv3x3_8to8_cuda(
        cudaSurfaceObject_t src,
        cudaSurfaceObject_t dst,
        unsigned int width,
        unsigned int height,
        const float* kernels,
        const float* biases,
        int computeCapability,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ 16, 8 };
        dim3 grid{ (width + block.x - 1) / block.x, (height + block.y - 1) / block.y };
        if (computeCapability >= 70)
            conv3x3_cuda_half<8, 8> <<< grid, block, 0, stream >>> (src, dst, width, height, kernels, biases);
        else
            conv3x3_cuda_float<8, 8> <<< grid, block, 0, stream >>> (src, dst, width, height, kernels, biases);
    }

    void conv3x3_residual_8to8_cuda(
        cudaSurfaceObject_t src,
        cudaSurfaceObject_t dst,
        unsigned int width,
        unsigned int height,
        const float* kernels,
        const float* biases,
        int computeCapability,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ 16, 8 };
        dim3 grid{ (width + block.x - 1) / block.x, (height + block.y - 1) / block.y };
        if (computeCapability >= 70)
            conv3x3_cuda_half<8, 8, true> <<< grid, block, 0, stream >>> (src, dst, width, height, kernels, biases);
        else
            conv3x3_cuda_float<8, 8, true> <<< grid, block, 0, stream >>> (src, dst, width, height, kernels, biases);
    }

    void deconv2x2_8to1_cuda(
        cudaSurfaceObject_t src,
        cudaSurfaceObject_t dst,
        unsigned int width,
        unsigned int height,
        const float* kernels,
        Image::ElementType type,
        [[maybe_unused]] int computeCapability,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ 16, 8 };
        dim3 grid{ (width + block.x - 1) / block.x, (height + block.y - 1) / block.y };
        switch (type)
        {
        case Image::UInt8:
            return deconv2x2_cuda_cout1<std::uint8_t, 8> <<< grid, block, 0, stream >>> (src, dst, width, height, kernels);
        case Image::UInt16:
            return deconv2x2_cuda_cout1<std::uint16_t, 8> <<< grid, block, 0, stream >>> (src, dst, width, height, kernels);
        case Image::Float32:
            return deconv2x2_cuda_cout1<float, 8> <<< grid, block, 0, stream >>> (src, dst, width, height, kernels);
        }
    }
}
