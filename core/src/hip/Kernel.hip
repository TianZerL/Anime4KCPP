#include <cstddef>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <type_traits>
#include <limits>

#include "AC/Core/Image.hpp"

namespace ac::core::hip
{
    template<typename Float, std::enable_if_t<std::is_floating_point_v<Float>, bool> = true>
    __device__ inline Float fromFloat(const float v)
    {
        return fminf(fmaxf(v, 0.0f), 1.0f);
    }
    template<typename Unsigned, std::enable_if_t<std::is_unsigned_v<Unsigned>, bool> = true>
    __device__ inline Unsigned fromFloat(const float v)
    {
        return static_cast<Unsigned>(fromFloat<float>(v) * std::numeric_limits<Unsigned>::max() + 0.5f);
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
        std::enable_if_t<cout % 4 == 0 && (cout * 9 < 128 * 4), bool> = true>
    __global__ void conv3x3_hip_cin1(
        hipTextureObject_t src,
        void* const __restrict__ dst,
        const std::size_t dstPitch,
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

            auto dstBuffer = (unsigned short*)((char*)dst + y * dstPitch) + cout * x + 4 * nidx;
           
            dstBuffer[0] = __half_as_ushort(__float2half(fmaxf(
                    r[0] * kptr[offset0 + 0] +
                    r[1] * kptr[offset0 + 1] +
                    r[2] * kptr[offset0 + 2] +
                    r[3] * kptr[offset0 + 3] +
                    r[4] * kptr[offset0 + 4] +
                    r[5] * kptr[offset0 + 5] +
                    r[6] * kptr[offset0 + 6] +
                    r[7] * kptr[offset0 + 7] +
                    r[8] * kptr[offset0 + 8] + biases[npos + 0], 0.0f
                )));
            dstBuffer[1] = __half_as_ushort(__float2half(fmaxf(
                    r[0] * kptr[offset1 + 0] +
                    r[1] * kptr[offset1 + 1] +
                    r[2] * kptr[offset1 + 2] +
                    r[3] * kptr[offset1 + 3] +
                    r[4] * kptr[offset1 + 4] +
                    r[5] * kptr[offset1 + 5] +
                    r[6] * kptr[offset1 + 6] +
                    r[7] * kptr[offset1 + 7] +
                    r[8] * kptr[offset1 + 8] + biases[npos + 1], 0.0f
                )));
            dstBuffer[2] = __half_as_ushort(__float2half(fmaxf(
                    r[0] * kptr[offset2 + 0] +
                    r[1] * kptr[offset2 + 1] +
                    r[2] * kptr[offset2 + 2] +
                    r[3] * kptr[offset2 + 3] +
                    r[4] * kptr[offset2 + 4] +
                    r[5] * kptr[offset2 + 5] +
                    r[6] * kptr[offset2 + 6] +
                    r[7] * kptr[offset2 + 7] +
                    r[8] * kptr[offset2 + 8] + biases[npos + 2], 0.0f
                )));
            dstBuffer[3] = __half_as_ushort(__float2half(fmaxf(
                    r[0] * kptr[offset3 + 0] +
                    r[1] * kptr[offset3 + 1] +
                    r[2] * kptr[offset3 + 2] +
                    r[3] * kptr[offset3 + 3] +
                    r[4] * kptr[offset3 + 4] +
                    r[5] * kptr[offset3 + 5] +
                    r[6] * kptr[offset3 + 6] +
                    r[7] * kptr[offset3 + 7] +
                    r[8] * kptr[offset3 + 8] + biases[npos + 3], 0.0f
                )));
        }
    }

    template<int cin, int cout,
        std::enable_if_t<(cin % 4 == 0) && (cout % 4 == 0) && (cout * 9 * cin < 128 * 8), bool> = true>
    __global__ void conv3x3_hip(
        const void* const __restrict__ src,
        const std::size_t srcPitch,
        void* const __restrict__ dst,
        const std::size_t dstPitch,
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

        auto sp = y < height - 1 ? +srcPitch : 0;
        auto sn = y > 0 ? -srcPitch : 0;
        auto cp = x < width - 1 ? +cin * sizeof(unsigned short) : 0;
        auto cn = x > 0 ? -cin * sizeof(unsigned short) : 0;

        float4 r0[lin] = {};
        float4 r1[lin] = {};
        float4 r2[lin] = {};
        float4 r3[lin] = {};
        float4 r4[lin] = {};
        float4 r5[lin] = {};
        float4 r6[lin] = {};
        float4 r7[lin] = {};
        float4 r8[lin] = {};

        auto srcBuffer = ((char*)src + y * srcPitch) + cin * x * sizeof(unsigned short);

        auto tl = (unsigned short*)(srcBuffer + sn + cn), tc = (unsigned short*)(srcBuffer + sn), tr = (unsigned short*)(srcBuffer + sn + cp);
        auto ml = (unsigned short*)(srcBuffer +      cn), mc = (unsigned short*)(srcBuffer     ), mr = (unsigned short*)(srcBuffer +      cp);
        auto bl = (unsigned short*)(srcBuffer + sp + cn), bc = (unsigned short*)(srcBuffer + sp), br = (unsigned short*)(srcBuffer + sp + cp);

        for (int cidx = 0; cidx < lin; cidx++)
        {
            r0[cidx].x =  __half2float(__ushort_as_half(tl[cidx * 4 + 0]));
            r0[cidx].y =  __half2float(__ushort_as_half(tl[cidx * 4 + 1]));
            r0[cidx].z =  __half2float(__ushort_as_half(tl[cidx * 4 + 2]));
            r0[cidx].w =  __half2float(__ushort_as_half(tl[cidx * 4 + 3]));
            
            r1[cidx].x =  __half2float(__ushort_as_half(tc[cidx * 4 + 0]));
            r1[cidx].y =  __half2float(__ushort_as_half(tc[cidx * 4 + 1]));
            r1[cidx].z =  __half2float(__ushort_as_half(tc[cidx * 4 + 2]));
            r1[cidx].w =  __half2float(__ushort_as_half(tc[cidx * 4 + 3]));

            r2[cidx].x =  __half2float(__ushort_as_half(tr[cidx * 4 + 0]));
            r2[cidx].y =  __half2float(__ushort_as_half(tr[cidx * 4 + 1]));
            r2[cidx].z =  __half2float(__ushort_as_half(tr[cidx * 4 + 2]));
            r2[cidx].w =  __half2float(__ushort_as_half(tr[cidx * 4 + 3]));

            r3[cidx].x =  __half2float(__ushort_as_half(ml[cidx * 4 + 0]));
            r3[cidx].y =  __half2float(__ushort_as_half(ml[cidx * 4 + 1]));
            r3[cidx].z =  __half2float(__ushort_as_half(ml[cidx * 4 + 2]));
            r3[cidx].w =  __half2float(__ushort_as_half(ml[cidx * 4 + 3]));

            r4[cidx].x =  __half2float(__ushort_as_half(mc[cidx * 4 + 0]));
            r4[cidx].y =  __half2float(__ushort_as_half(mc[cidx * 4 + 1]));
            r4[cidx].z =  __half2float(__ushort_as_half(mc[cidx * 4 + 2]));
            r4[cidx].w =  __half2float(__ushort_as_half(mc[cidx * 4 + 3]));
            
            r5[cidx].x =  __half2float(__ushort_as_half(mr[cidx * 4 + 0]));
            r5[cidx].y =  __half2float(__ushort_as_half(mr[cidx * 4 + 1]));
            r5[cidx].z =  __half2float(__ushort_as_half(mr[cidx * 4 + 2]));
            r5[cidx].w =  __half2float(__ushort_as_half(mr[cidx * 4 + 3]));

            r6[cidx].x =  __half2float(__ushort_as_half(bl[cidx * 4 + 0]));
            r6[cidx].y =  __half2float(__ushort_as_half(bl[cidx * 4 + 1]));
            r6[cidx].z =  __half2float(__ushort_as_half(bl[cidx * 4 + 2]));
            r6[cidx].w =  __half2float(__ushort_as_half(bl[cidx * 4 + 3]));

            r7[cidx].x =  __half2float(__ushort_as_half(bc[cidx * 4 + 0]));
            r7[cidx].y =  __half2float(__ushort_as_half(bc[cidx * 4 + 1]));
            r7[cidx].z =  __half2float(__ushort_as_half(bc[cidx * 4 + 2]));
            r7[cidx].w =  __half2float(__ushort_as_half(bc[cidx * 4 + 3]));

            r8[cidx].x =  __half2float(__ushort_as_half(br[cidx * 4 + 0]));
            r8[cidx].y =  __half2float(__ushort_as_half(br[cidx * 4 + 1]));
            r8[cidx].z =  __half2float(__ushort_as_half(br[cidx * 4 + 2]));
            r8[cidx].w =  __half2float(__ushort_as_half(br[cidx * 4 + 3]));
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

            auto dstBuffer = (unsigned short*)((char*)dst + y * dstPitch) + cout * x + 4 * nidx;

            dstBuffer[0] = __half_as_ushort(__float2half(fmaxf(sum[0], 0.0f))),
            dstBuffer[1] = __half_as_ushort(__float2half(fmaxf(sum[1], 0.0f))),
            dstBuffer[2] = __half_as_ushort(__float2half(fmaxf(sum[2], 0.0f))),
            dstBuffer[3] = __half_as_ushort(__float2half(fmaxf(sum[3], 0.0f)));
        }
    }

    template<typename OUT, int cin,
        std::enable_if_t<cin % 4 == 0, bool> = true>
    __global__ void deconv2x2_hip_cout1(
        const void* const __restrict__ src,
        const std::size_t srcPitch,
        hipSurfaceObject_t dst,
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
            float4 r{};
            auto srcBuffer = (unsigned short*)((char*)src + (y / 2) * srcPitch) + cin * (x / 2);
            r.x = __half2float(__ushort_as_half(srcBuffer[cidx * 4 + 0]));
            r.y = __half2float(__ushort_as_half(srcBuffer[cidx * 4 + 1]));
            r.z = __half2float(__ushort_as_half(srcBuffer[cidx * 4 + 2]));
            r.w = __half2float(__ushort_as_half(srcBuffer[cidx * 4 + 3]));
            sum += dot(r, make_float4(
                kernels[offset + 0],
                kernels[offset + 4],
                kernels[offset + 8],
                kernels[offset + 12]));
        }
        surf2Dwrite(fromFloat<OUT>(sum), dst, sizeof(OUT) * x, y);
    }

    void conv3x3_1to8_hip(
        hipTextureObject_t src,
        void* dst,
        std::size_t dstPitch,
        unsigned int width,
        unsigned int height,
        const float* kernels,
        const float* biases,
        hipStream_t stream
    ) noexcept
    {
        dim3 block{ 16, 8 };
        dim3 grid{ (width + block.x - 1) / block.x, (height + block.y - 1) / block.y };
        conv3x3_hip_cin1<8> <<< grid, block, 0, stream >>> (src, dst, dstPitch, width, height, kernels, biases);
    }

    void conv3x3_8to8_hip(
        const void* src,
        std::size_t srcPitch,
        void* dst,
        std::size_t dstPitch,
        unsigned int width,
        unsigned int height,
        const float* kernels,
        const float* biases,
        hipStream_t stream
    ) noexcept
    {
        dim3 block{ 16, 8 };
        dim3 grid{ (width + block.x - 1) / block.x, (height + block.y - 1) / block.y };
        conv3x3_hip<8, 8> <<< grid, block, 0, stream >>> (src, srcPitch, dst, dstPitch, width, height, kernels, biases);
    }

    void deconv2x2_8to1_hip(
        const void* src,
        std::size_t srcPitch,
        hipSurfaceObject_t dst,
        unsigned int width,
        unsigned int height,
        const float* kernels,
        Image::ElementType type,
        hipStream_t stream
    ) noexcept
    {
        dim3 block{ 16, 8 };
        dim3 grid{ (width + block.x - 1) / block.x, (height + block.y - 1) / block.y };
        switch (type)
        {
        case Image::UInt8:
            return deconv2x2_hip_cout1<std::uint8_t, 8> <<< grid, block, 0, stream >>> (src, srcPitch, dst, width, height, kernels);
        case Image::UInt16:
            return deconv2x2_hip_cout1<std::uint16_t, 8> <<< grid, block, 0, stream >>> (src, srcPitch, dst, width, height, kernels);
        case Image::Float32:
            return deconv2x2_hip_cout1<float, 8> <<< grid, block, 0, stream >>> (src, srcPitch, dst, width, height, kernels);
        }
    }
}
