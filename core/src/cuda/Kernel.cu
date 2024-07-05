#include <cuda_fp16.h>

#include <cuda/std/type_traits>
#include <cuda/std/limits>

#include "AC/Core/Image.hpp"

namespace ac::core::cuda
{
    template<typename Float, ::cuda::std::enable_if_t<::cuda::std::is_floating_point_v<Float>, bool> = true>
    __device__ inline Float fromFloat(const float v)
    {
        return fminf(fmaxf(v, 0.0f), 1.0f);
    }
    template<typename Unsigned, ::cuda::std::enable_if_t<::cuda::std::is_unsigned_v<Unsigned>, bool> = true>
    __device__ inline Unsigned fromFloat(const float v)
    {
        return static_cast<Unsigned>(fromFloat<float>(v) * ::cuda::std::numeric_limits<Unsigned>::max() + 0.5f);
    }

    __device__ inline static float dot(const float4 a, const float* const __restrict__ b)
    {
        return a.x * b[0] + a.y * b[1] + a.z * b[2] + a.w * b[3];
    }
    __device__ inline static float dot(const float4 a, const float4 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    template<int cout,
        ::cuda::std::enable_if_t<cout % 4 == 0 && (cout * 9 < 128 * 4), bool> = true>
    __global__ void conv3x3_cuda_cin1(
        cudaTextureObject_t src,
        cudaSurfaceObject_t dst,
        const unsigned int width,
        const unsigned int height,
        const float* const __restrict__ kernels,
        const float* const __restrict__ biases
    )
    {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;
        auto tid = threadIdx.y * blockDim.x + threadIdx.x;

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

        __shared__ float kptr[cout * 9];
        if (tid * 4 < cout * 9)
        {
            kptr[tid * 4 + 0] = kernels[tid * 4 + 0];
            kptr[tid * 4 + 1] = kernels[tid * 4 + 1];
            kptr[tid * 4 + 2] = kernels[tid * 4 + 2];
            kptr[tid * 4 + 3] = kernels[tid * 4 + 3];
        }
        __syncthreads();

        for (int nidx = 0; nidx < lout; nidx++)
        {
            auto npos = nidx * 4;

            auto offset0 = (npos + 0) * 9;
            auto offset1 = (npos + 1) * 9;
            auto offset2 = (npos + 2) * 9;
            auto offset3 = (npos + 3) * 9;

            auto layer = make_ushort4(
                __half_as_ushort(__float2half(fmaxf(
                    r[0] * kptr[offset0 + 0] +
                    r[1] * kptr[offset0 + 1] +
                    r[2] * kptr[offset0 + 2] +
                    r[3] * kptr[offset0 + 3] +
                    r[4] * kptr[offset0 + 4] +
                    r[5] * kptr[offset0 + 5] +
                    r[6] * kptr[offset0 + 6] +
                    r[7] * kptr[offset0 + 7] +
                    r[8] * kptr[offset0 + 8] + biases[npos + 0], 0.0f
                ))),
                __half_as_ushort(__float2half(fmaxf(
                    r[0] * kptr[offset1 + 0] +
                    r[1] * kptr[offset1 + 1] +
                    r[2] * kptr[offset1 + 2] +
                    r[3] * kptr[offset1 + 3] +
                    r[4] * kptr[offset1 + 4] +
                    r[5] * kptr[offset1 + 5] +
                    r[6] * kptr[offset1 + 6] +
                    r[7] * kptr[offset1 + 7] +
                    r[8] * kptr[offset1 + 8] + biases[npos + 1], 0.0f
                ))),
                __half_as_ushort(__float2half(fmaxf(
                    r[0] * kptr[offset2 + 0] +
                    r[1] * kptr[offset2 + 1] +
                    r[2] * kptr[offset2 + 2] +
                    r[3] * kptr[offset2 + 3] +
                    r[4] * kptr[offset2 + 4] +
                    r[5] * kptr[offset2 + 5] +
                    r[6] * kptr[offset2 + 6] +
                    r[7] * kptr[offset2 + 7] +
                    r[8] * kptr[offset2 + 8] + biases[npos + 2], 0.0f
                ))),
                __half_as_ushort(__float2half(fmaxf(
                    r[0] * kptr[offset3 + 0] +
                    r[1] * kptr[offset3 + 1] +
                    r[2] * kptr[offset3 + 2] +
                    r[3] * kptr[offset3 + 3] +
                    r[4] * kptr[offset3 + 4] +
                    r[5] * kptr[offset3 + 5] +
                    r[6] * kptr[offset3 + 6] +
                    r[7] * kptr[offset3 + 7] +
                    r[8] * kptr[offset3 + 8] + biases[npos + 3], 0.0f
                ))));
            surf2DLayeredwrite(layer, dst, sizeof(layer) * x, y, nidx, cudaBoundaryModeZero);
        }
    }

    template<int cin, int cout,
        ::cuda::std::enable_if_t<(cin % 4 == 0) && (cout % 4 == 0) && (cout * 9 * cin < 128 * 8), bool> = true>
    __global__ void conv3x3_cuda(
        cudaTextureObject_t src,
        cudaSurfaceObject_t dst,
        const unsigned int width,
        const unsigned int height,
        const float* const __restrict__ kernels,
        const float* const __restrict__ biases
    )
    {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;
        auto tid = threadIdx.y * blockDim.x + threadIdx.x;

        if (x >= width || y >= height) return;

        constexpr int lin = cin / 4;
        constexpr int lout = cout / 4;

        float4 r0[lin] = {};
        float4 r1[lin] = {};
        float4 r2[lin] = {};
        float4 r3[lin] = {};
        float4 r4[lin] = {};
        float4 r5[lin] = {};
        float4 r6[lin] = {};
        float4 r7[lin] = {};
        float4 r8[lin] = {};

        for (int cidx = 0; cidx < lin; cidx++)
        {
            r0[cidx] = tex2DLayered<float4>(src, x - 1, y - 1, cidx);
            r1[cidx] = tex2DLayered<float4>(src, x    , y - 1, cidx);
            r2[cidx] = tex2DLayered<float4>(src, x + 1, y - 1, cidx);
            r3[cidx] = tex2DLayered<float4>(src, x - 1, y    , cidx);
            r4[cidx] = tex2DLayered<float4>(src, x    , y    , cidx);
            r5[cidx] = tex2DLayered<float4>(src, x + 1, y    , cidx);
            r6[cidx] = tex2DLayered<float4>(src, x - 1, y + 1, cidx);
            r7[cidx] = tex2DLayered<float4>(src, x    , y + 1, cidx);
            r8[cidx] = tex2DLayered<float4>(src, x + 1, y + 1, cidx);
        };

        __shared__ float kptr[cout * 9 * cin];
        if (tid * 8 < cout * 9 * cin)
        {
            kptr[tid * 8 + 0] = kernels[tid * 8 + 0];
            kptr[tid * 8 + 1] = kernels[tid * 8 + 1];
            kptr[tid * 8 + 2] = kernels[tid * 8 + 2];
            kptr[tid * 8 + 3] = kernels[tid * 8 + 3];
            kptr[tid * 8 + 4] = kernels[tid * 8 + 4];
            kptr[tid * 8 + 5] = kernels[tid * 8 + 5];
            kptr[tid * 8 + 6] = kernels[tid * 8 + 6];
            kptr[tid * 8 + 7] = kernels[tid * 8 + 7];
        }
        __syncthreads();

        for (int nidx = 0; nidx < lout; nidx++)
        {
            auto npos = nidx * 4;
            float sum[4] = {};
            for (int i = 0; i < 4; i++)
            {
                auto offset0 = (npos + i) * 9 * cin + 0 * cin;
                auto offset1 = (npos + i) * 9 * cin + 1 * cin;
                auto offset2 = (npos + i) * 9 * cin + 2 * cin;
                auto offset3 = (npos + i) * 9 * cin + 3 * cin;
                auto offset4 = (npos + i) * 9 * cin + 4 * cin;
                auto offset5 = (npos + i) * 9 * cin + 5 * cin;
                auto offset6 = (npos + i) * 9 * cin + 6 * cin;
                auto offset7 = (npos + i) * 9 * cin + 7 * cin;
                auto offset8 = (npos + i) * 9 * cin + 8 * cin;

                for (int cidx = 0; cidx < lin; cidx++)
                {
                    auto cpos = cidx * 4;
                    sum[i] +=
                        dot(r0[cidx], kptr + offset0 + cpos) +
                        dot(r1[cidx], kptr + offset1 + cpos) +
                        dot(r2[cidx], kptr + offset2 + cpos) +
                        dot(r3[cidx], kptr + offset3 + cpos) +
                        dot(r4[cidx], kptr + offset4 + cpos) +
                        dot(r5[cidx], kptr + offset5 + cpos) +
                        dot(r6[cidx], kptr + offset6 + cpos) +
                        dot(r7[cidx], kptr + offset7 + cpos) +
                        dot(r8[cidx], kptr + offset8 + cpos);
                }

                sum[i] += biases[npos + i];
            }

            auto layer = make_ushort4(
                __half_as_ushort(__float2half(fmaxf(sum[0], 0.0f))),
                __half_as_ushort(__float2half(fmaxf(sum[1], 0.0f))),
                __half_as_ushort(__float2half(fmaxf(sum[2], 0.0f))),
                __half_as_ushort(__float2half(fmaxf(sum[3], 0.0f))));

            surf2DLayeredwrite(layer, dst, sizeof(layer) * x, y, nidx, cudaBoundaryModeZero);
        }
    }

    template<typename OUT, int cin,
        ::cuda::std::enable_if_t<cin % 4 == 0, bool> = true>
    __global__ void deconv2x2_cuda_cout1(
        cudaTextureObject_t src,
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
        for (int cidx = 0; cidx < lin; cidx++)
        {
            auto offset = cidx * 4 * 4 + index;
            sum += dot(tex2DLayered<float4>(src, x / 2, y / 2, cidx), make_float4(
                kernels[offset + 0],
                kernels[offset + 4],
                kernels[offset + 8],
                kernels[offset + 12]));
        }
        surf2Dwrite(fromFloat<OUT>(sum), dst, sizeof(OUT) * x, y, cudaBoundaryModeZero);
    }

    void conv3x3_1to8_cuda(
        cudaTextureObject_t src,
        cudaSurfaceObject_t dst,
        unsigned int width,
        unsigned int height,
        const float* kernels,
        const float* biases,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ 16, 8 };
        dim3 grid{ (width + block.x - 1) / block.x, (height + block.y - 1) / block.y };
        conv3x3_cuda_cin1<8> <<< grid, block, 0, stream >>> (src, dst, width, height, kernels, biases);
    }

    void conv3x3_8to8_cuda(
        cudaTextureObject_t src,
        cudaSurfaceObject_t dst,
        unsigned int width,
        unsigned int height,
        const float* kernels,
        const float* biases,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ 16, 8 };
        dim3 grid{ (width + block.x - 1) / block.x, (height + block.y - 1) / block.y };
        conv3x3_cuda<8, 8> <<< grid, block, 0, stream >>> (src, dst, width, height, kernels, biases);
    }

    void deconv2x2_8to1_cuda(
        cudaTextureObject_t src,
        cudaSurfaceObject_t dst,
        unsigned int width,
        unsigned int height,
        const float* kernels,
        Image::ElementType type,
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
