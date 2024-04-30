#include <cuda_fp16.h>

#include <cuda/std/type_traits>
#include <cuda/std/limits>

#include "AC/Core/Image.hpp"

namespace ac::core::cuda
{
    template<typename Float, ::cuda::std::enable_if_t<::cuda::std::is_floating_point_v<Float>, bool> = true>
    __device__ inline static Float fromFloat(const float v)
    {
        return fminf(fmaxf(v, 0.0f), 1.0f);
    }
    template<typename Unsigned, ::cuda::std::enable_if_t<::cuda::std::is_unsigned_v<Unsigned>, bool> = true>
    __device__ inline static Unsigned fromFloat(const float v)
    {
        return static_cast<Unsigned>(rintf(fromFloat<float>(v) * ::cuda::std::numeric_limits<Unsigned>::max()));
    }

    __device__ inline static float dot(float4 a, const float* __restrict__ b)
    {
        return a.x * b[0] + a.y * b[1] + a.z * b[2] + a.w * b[3];
    }
    __device__ inline static float dot(float4 a, float4 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    template<int cout,
        ::cuda::std::enable_if_t<cout % 4 == 0, bool> = true>
    __global__ static void conv3x3_cuda_cin1(
        cudaTextureObject_t src,
        cudaSurfaceObject_t dst,
        const unsigned int width,
        const unsigned int height,
        const float* __restrict__ kernels,
        const float* __restrict__ biases
    )
    {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

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

            const float* __restrict__ k0 = kernels + (npos + 0) * 9;
            const float* __restrict__ k1 = kernels + (npos + 1) * 9;
            const float* __restrict__ k2 = kernels + (npos + 2) * 9;
            const float* __restrict__ k3 = kernels + (npos + 3) * 9;

            auto layer = make_ushort4(
                __half_as_ushort(__float2half(fmaxf(
                    r[0] * k0[0] +
                    r[1] * k0[1] +
                    r[2] * k0[2] +
                    r[3] * k0[3] +
                    r[4] * k0[4] +
                    r[5] * k0[5] +
                    r[6] * k0[6] +
                    r[7] * k0[7] +
                    r[8] * k0[8] + biases[npos + 0], 0.0f
                ))),
                __half_as_ushort(__float2half(fmaxf(
                    r[0] * k1[0] +
                    r[1] * k1[1] +
                    r[2] * k1[2] +
                    r[3] * k1[3] +
                    r[4] * k1[4] +
                    r[5] * k1[5] +
                    r[6] * k1[6] +
                    r[7] * k1[7] +
                    r[8] * k1[8] + biases[npos + 1], 0.0f
                ))),
                __half_as_ushort(__float2half(fmaxf(
                    r[0] * k2[0] +
                    r[1] * k2[1] +
                    r[2] * k2[2] +
                    r[3] * k2[3] +
                    r[4] * k2[4] +
                    r[5] * k2[5] +
                    r[6] * k2[6] +
                    r[7] * k2[7] +
                    r[8] * k2[8] + biases[npos + 2], 0.0f
                ))),
                __half_as_ushort(__float2half(fmaxf(
                    r[0] * k3[0] +
                    r[1] * k3[1] +
                    r[2] * k3[2] +
                    r[3] * k3[3] +
                    r[4] * k3[4] +
                    r[5] * k3[5] +
                    r[6] * k3[6] +
                    r[7] * k3[7] +
                    r[8] * k3[8] + biases[npos + 3], 0.0f
                ))));
            surf2DLayeredwrite(layer, dst, sizeof(layer) * x, y, nidx, cudaBoundaryModeZero);
        }
    }

    template<int cin, int cout,
        ::cuda::std::enable_if_t<(cin % 4 == 0) && (cout % 4 == 0), bool> = true>
    __global__ static void conv3x3_cuda(
        cudaTextureObject_t src,
        cudaSurfaceObject_t dst,
        const unsigned int width,
        const unsigned int height,
        const float* __restrict__ kernels,
        const float* __restrict__ biases
    )
    {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

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

        for (int nidx = 0; nidx < lout; nidx++)
        {
            auto npos = nidx * 4;
            float sum[4] = {};
            for (int i = 0; i < 4; i++)
            {
                const float* __restrict__ k0 = kernels + (npos + i) * 9 * cin + 0 * cin;
                const float* __restrict__ k1 = kernels + (npos + i) * 9 * cin + 1 * cin;
                const float* __restrict__ k2 = kernels + (npos + i) * 9 * cin + 2 * cin;
                const float* __restrict__ k3 = kernels + (npos + i) * 9 * cin + 3 * cin;
                const float* __restrict__ k4 = kernels + (npos + i) * 9 * cin + 4 * cin;
                const float* __restrict__ k5 = kernels + (npos + i) * 9 * cin + 5 * cin;
                const float* __restrict__ k6 = kernels + (npos + i) * 9 * cin + 6 * cin;
                const float* __restrict__ k7 = kernels + (npos + i) * 9 * cin + 7 * cin;
                const float* __restrict__ k8 = kernels + (npos + i) * 9 * cin + 8 * cin;

                for (int cidx = 0; cidx < lin; cidx++)
                {
                    auto cpos = cidx * 4;
                    sum[i] +=
                        dot(r0[cidx], k0 + cpos) +
                        dot(r1[cidx], k1 + cpos) +
                        dot(r2[cidx], k2 + cpos) +
                        dot(r3[cidx], k3 + cpos) +
                        dot(r4[cidx], k4 + cpos) +
                        dot(r5[cidx], k5 + cpos) +
                        dot(r6[cidx], k6 + cpos) +
                        dot(r7[cidx], k7 + cpos) +
                        dot(r8[cidx], k8 + cpos);
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
    __global__ static void deconv2x2_cuda_cout1(
        cudaTextureObject_t src,
        cudaSurfaceObject_t dst,
        const unsigned int width,
        const unsigned int height,
        const float* __restrict__ kernels
    )
    {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) return;

        const unsigned int index = ((y & 1) << 1) + (x & 1);

        constexpr int lin = cin / 4;

        float sum = 0.0f;
        for (int cidx = 0; cidx < lin; cidx++)
        {
            const float* __restrict__ k = kernels + cidx * 4 * 4 + index;
            sum += dot(tex2DLayered<float4>(src, x / 2, y / 2, cidx), make_float4(k[0], k[4], k[8], k[12]));
        }
        surf2Dwrite(fromFloat<OUT>(sum), dst, sizeof(OUT) * x, y, cudaBoundaryModeZero);
    }

    void conv3x3_1to8_cuda(
        cudaTextureObject_t src,
        cudaSurfaceObject_t dst,
        const unsigned int width,
        const unsigned int height,
        const float* kernels,
        const float* biases,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ 8, 32 };
        dim3 grid{ (width + block.x - 1) / block.x, (height + block.y - 1) / block.y };
        conv3x3_cuda_cin1<8> <<< grid, block, 0, stream >>> (src, dst, width, height, kernels, biases);
    }

    void conv3x3_8to8_cuda(
        cudaTextureObject_t src,
        cudaSurfaceObject_t dst,
        const unsigned int width,
        const unsigned int height,
        const float* kernels,
        const float* biases,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ 8, 32 };
        dim3 grid{ (width + block.x - 1) / block.x, (height + block.y - 1) / block.y };
        conv3x3_cuda<8, 8> <<< grid, block, 0, stream >>> (src, dst, width, height, kernels, biases);
    }

    void deconv2x2_8to1_cuda(
        cudaTextureObject_t src,
        cudaSurfaceObject_t dst,
        const unsigned int width,
        const unsigned int height,
        const float* kernels,
        Image::ElementType type,
        cudaStream_t stream
    ) noexcept
    {
        dim3 block{ 8, 32 };
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
