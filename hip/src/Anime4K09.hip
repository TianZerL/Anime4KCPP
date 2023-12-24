#include "hip/hip_runtime.h"
#include "HipHelper.hip"
#include "HipInterface.hpp"
#include <hip/amd_detail/amd_surface_functions.h>

#define MAX3(a, b, c) fmaxf(fmaxf(a, b), c)
#define MIN3(a, b, c) fminf(fminf(a, b), c)

template <typename T>
__inline__ __device__ static void getLightest(
    typename Vec4<T>::type &mc, typename Vec4<T>::type &a, typename Vec4<T>::type &b, typename Vec4<T>::type &c, float strength)
{
    constexpr float offset = std::is_floating_point<T>::value ? 0.0f : 0.5f;
    mc = makeVec4<T>(
        mc.x + strength * (__fdividef(a.x + b.x + c.x, 3.0f) - mc.x) + offset,
        mc.y + strength * (__fdividef(a.y + b.y + c.y, 3.0f) - mc.y) + offset,
        mc.z + strength * (__fdividef(a.z + b.z + c.z, 3.0f) - mc.z) + offset,
        mc.w + strength * (__fdividef(a.w + b.w + c.w, 3.0f) - mc.w) + offset);
}

template <typename T>
__inline__ __device__ static void getAVerage(
    typename Vec4<T>::type &mc, typename Vec4<T>::type &a, typename Vec4<T>::type &b, typename Vec4<T>::type &c, float strength)
{
    constexpr float offset = std::is_floating_point<T>::value ? 0.0f : 0.5f;
    mc = makeVec4<T>(
        mc.x + strength * (__fdividef(a.x + b.x + c.x, 3.0f) - mc.x) + offset,
        mc.y + strength * (__fdividef(a.y + b.y + c.y, 3.0f) - mc.y) + offset,
        mc.z + strength * (__fdividef(a.z + b.z + c.z, 3.0f) - mc.z) + offset,
        0.299f * mc.z + 0.587f * mc.y + 0.114f * mc.x + offset);
}

template <typename T>
__global__ static void getGray(
    hipTextureObject_t srcImg, hipSurfaceObject_t dstImg,
    unsigned int W, unsigned int H)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H)
        return;

    const float u = __fdividef(x + 0.5f, W);
    const float v = __fdividef(y + 0.5f, H);
    constexpr float scale = PixelValue<T>::max();
    constexpr float offset = std::is_floating_point<T>::value ? 0.0f : 0.5f;

    float4 fmc = tex2D<float4>(srcImg, u, v);

    auto mc = makeVec4<T>(
        fmc.x * scale + offset, fmc.y * scale + offset, fmc.z * scale + offset, fmc.w * scale + offset);
    mc.w = 0.299f * mc.z + 0.587f * mc.y + 0.114f * mc.x + offset;

    surf2Dwrite(mc, dstImg, sizeof(mc) * x, y);
}

template <typename T>
__global__ static void pushColor(
    hipSurfaceObject_t srcImg, hipSurfaceObject_t dstImg,
    unsigned int W, unsigned int H, const float strength)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H)
        return;

    typename Vec4<T>::type tl, tc, tr, ml, mc, mr, bl, bc, br;
    surf2Dread(&tl, srcImg, sizeof(mc) * (x - 1), y - 1);
    surf2Dread(&tc, srcImg, sizeof(mc) * x, y - 1);
    surf2Dread(&tr, srcImg, sizeof(mc) * (x + 1), y - 1);
    surf2Dread(&ml, srcImg, sizeof(mc) * (x - 1), y);
    surf2Dread(&mc, srcImg, sizeof(mc) * x, y);
    surf2Dread(&mr, srcImg, sizeof(mc) * (x + 1), y);
    surf2Dread(&bl, srcImg, sizeof(mc) * (x - 1), y + 1);
    surf2Dread(&bc, srcImg, sizeof(mc) * x, y + 1);
    surf2Dread(&br, srcImg, sizeof(mc) * (x + 1), y + 1);

    T maxD, minL;

    //top and bottom
    maxD = MAX3(bl.w, bc.w, br.w);
    minL = MIN3(tl.w, tc.w, tr.w);
    if (minL > mc.w && mc.w > maxD)
        getLightest<T>(mc, tl, tc, tr, strength);
    else
    {
        maxD = MAX3(tl.w, tc.w, tr.w);
        minL = MIN3(bl.w, bc.w, br.w);
        if (minL > mc.w && mc.w > maxD)
            getLightest<T>(mc, bl, bc, br, strength);
    }

    //sundiagonal
    maxD = MAX3(ml.w, mc.w, bc.w);
    minL = MIN3(tc.w, tr.w, mr.w);
    if (minL > maxD)
        getLightest<T>(mc, tc, tr, mr, strength);
    else
    {
        maxD = MAX3(tc.w, mc.w, mr.w);
        minL = MIN3(ml.w, bl.w, bc.w);
        if (minL > maxD)
            getLightest<T>(mc, ml, bl, bc, strength);
    }

    //left and right
    maxD = MAX3(tl.w, ml.w, bl.w);
    minL = MIN3(tr.w, mr.w, br.w);
    if (minL > mc.w && mc.w > maxD)
        getLightest<T>(mc, tr, mr, br, strength);
    else
    {
        maxD = MAX3(tr.w, mr.w, br.w);
        minL = MIN3(tl.w, ml.w, bl.w);
        if (minL > mc.w && mc.w > maxD)
            getLightest<T>(mc, tl, ml, bl, strength);
    }

    //diagonal
    maxD = MAX3(tc.w, mc.w, ml.w);
    minL = MIN3(mr.w, br.w, bc.w);
    if (minL > maxD)
        getLightest<T>(mc, mr, br, bc, strength);
    else
    {
        maxD = MAX3(bc.w, mc.w, mr.w);
        minL = MIN3(ml.w, tl.w, tc.w);
        if (minL > maxD)
            getLightest<T>(mc, ml, tl, tc, strength);
    }

    surf2Dwrite(mc, dstImg, sizeof(mc) * x, y);
}

template <typename T>
__global__ static void getGradient(
    hipSurfaceObject_t srcImg, hipSurfaceObject_t dstImg,
    unsigned int W, unsigned int H)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H)
        return;

    typename Vec4<T>::type tl, tc, tr, ml, mc, mr, bl, bc, br;
    surf2Dread(&tl, srcImg, sizeof(mc) * (x - 1), y - 1);
    surf2Dread(&tc, srcImg, sizeof(mc) * x, y - 1);
    surf2Dread(&tr, srcImg, sizeof(mc) * (x + 1), y - 1);
    surf2Dread(&ml, srcImg, sizeof(mc) * (x - 1), y);
    surf2Dread(&mc, srcImg, sizeof(mc) * x, y);
    surf2Dread(&mr, srcImg, sizeof(mc) * (x + 1), y);
    surf2Dread(&bl, srcImg, sizeof(mc) * (x - 1), y + 1);
    surf2Dread(&bc, srcImg, sizeof(mc) * x, y + 1);
    surf2Dread(&br, srcImg, sizeof(mc) * (x + 1), y + 1);

    const float gradX = tr.w + mr.w + mr.w + br.w - tl.w - ml.w - ml.w - bl.w;
    const float gradY = tl.w + tc.w + tc.w + tr.w - bl.w - bc.w - bc.w - br.w;

    mc.w = PixelValue<T>::max() - clamp(sqrtf(gradX * gradX + gradY * gradY), PixelValue<T>::min(), PixelValue<T>::max());

    surf2Dwrite(mc, dstImg, sizeof(mc) * x, y);
}

template <typename T>
__global__ static void pushGradient(
    hipSurfaceObject_t srcImg, hipSurfaceObject_t dstImg,
    unsigned int W, unsigned int H, const float strength)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H)
        return;

    typename Vec4<T>::type tl, tc, tr, ml, mc, mr, bl, bc, br;
    surf2Dread(&tl, srcImg, sizeof(mc) * (x - 1), y - 1);
    surf2Dread(&tc, srcImg, sizeof(mc) * x, y - 1);
    surf2Dread(&tr, srcImg, sizeof(mc) * (x + 1), y - 1);
    surf2Dread(&ml, srcImg, sizeof(mc) * (x - 1), y);
    surf2Dread(&mc, srcImg, sizeof(mc) * x, y);
    surf2Dread(&mr, srcImg, sizeof(mc) * (x + 1), y);
    surf2Dread(&bl, srcImg, sizeof(mc) * (x - 1), y + 1);
    surf2Dread(&bc, srcImg, sizeof(mc) * x, y + 1);
    surf2Dread(&br, srcImg, sizeof(mc) * (x + 1), y + 1);

    T maxD, minL;

    //top and bottom
    maxD = MAX3(bl.w, bc.w, br.w);
    minL = MIN3(tl.w, tc.w, tr.w);
    if (minL > mc.w && mc.w > maxD)
    {
        getAVerage<T>(mc, tl, tc, tr, strength);
        surf2Dwrite(mc, dstImg, sizeof(mc) * x, y);
        return;
    }

    maxD = MAX3(tl.w, tc.w, tr.w);
    minL = MIN3(bl.w, bc.w, br.w);
    if (minL > mc.w && mc.w > maxD)
    {
        getAVerage<T>(mc, bl, bc, br, strength);
        surf2Dwrite(mc, dstImg, sizeof(mc) * x, y);
        return;
    }

    //sundiagonal
    maxD = MAX3(ml.w, mc.w, bc.w);
    minL = MIN3(tc.w, tr.w, mr.w);
    if (minL > maxD)
    {
        getAVerage<T>(mc, tc, tr, mr, strength);
        surf2Dwrite(mc, dstImg, sizeof(mc) * x, y);
        return;
    }

    maxD = MAX3(tc.w, mc.w, mr.w);
    minL = MIN3(ml.w, bl.w, bc.w);
    if (minL > maxD)
    {
        getAVerage<T>(mc, ml, bl, bc, strength);
        surf2Dwrite(mc, dstImg, sizeof(mc) * x, y);
        return;
    }

    //left and right
    maxD = MAX3(tl.w, ml.w, bl.w);
    minL = MIN3(tr.w, mr.w, br.w);
    if (minL > mc.w && mc.w > maxD)
    {
        getAVerage<T>(mc, tr, mr, br, strength);
        surf2Dwrite(mc, dstImg, sizeof(mc) * x, y);
        return;
    }

    maxD = MAX3(tr.w, mr.w, br.w);
    minL = MIN3(tl.w, ml.w, bl.w);
    if (minL > mc.w && mc.w > maxD)
    {
        getAVerage<T>(mc, tl, ml, bl, strength);
        surf2Dwrite(mc, dstImg, sizeof(mc) * x, y);
        return;
    }

    //diagonal
    maxD = MAX3(tc.w, mc.w, ml.w);
    minL = MIN3(mr.w, br.w, bc.w);
    if (minL > maxD)
    {
        getAVerage<T>(mc, mr, br, bc, strength);
        surf2Dwrite(mc, dstImg, sizeof(mc) * x, y);
        return;
    }
    maxD = MAX3(bc.w, mc.w, mr.w);
    minL = MIN3(ml.w, tl.w, tc.w);
    if (minL > maxD)
    {
        getAVerage<T>(mc, ml, tl, tc, strength);
        surf2Dwrite(mc, dstImg, sizeof(mc) * x, y);
        return;
    }

    mc.w = 0.299f * mc.z + 0.587f * mc.y + 0.114f * mc.x;
    surf2Dwrite(mc, dstImg, sizeof(mc) * x, y);
}

template <typename T>
static void cuRunKernelAnime4K09Impl(const T *inputData, T *outputData, Anime4KCPP::Hip::ACHipParamAnime4K09 *param, hipStream_t& stream)
{
    hipError_t err = hipSuccess;
    if (currHipDeviceID)
    {
        err = hipSetDevice(currHipDeviceID);
        CheckHipErr(err);
    }

    // hipStream_t stream;
    // hipStreamCreate(&stream);

    hipChannelFormatDesc channelDesc = hipCreateChannelDesc<typename Vec4<T>::type>();

    hipArray_t cuArray0;
    err = hipMallocArray(&cuArray0, &channelDesc, param->orgW, param->orgH);
    CheckHipErr(err);

    hipArray_t cuArray1;
    err = hipMallocArray(&cuArray1, &channelDesc, param->W, param->H, hipArrayDefault /* hipArraySurfaceLoadStore */);
    CheckHipErr(err);

    hipArray_t cuArray2;
    err = hipMallocArray(&cuArray2, &channelDesc, param->W, param->H, hipArrayDefault /* hipArraySurfaceLoadStore */);
    CheckHipErr(err);

    hipArray_t cuArray3;
    err = hipMallocArray(&cuArray3, &channelDesc, param->W, param->H, hipArrayDefault /* hipArraySurfaceLoadStore */);
    CheckHipErr(err);

    struct hipResourceDesc resDesc;
    struct hipTextureDesc texDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    memset(&texDesc, 0, sizeof(texDesc));

    texDesc.addressMode[0] = hipAddressModeBorder;
    texDesc.addressMode[1] = hipAddressModeBorder;
    texDesc.filterMode = hipFilterModeLinear;
    texDesc.readMode = std::is_floating_point<T>::value ? hipReadModeElementType : hipReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    resDesc.resType = hipResourceTypeArray;

    resDesc.res.array.array = cuArray0;
    hipTextureObject_t tex = 0;
    err = hipCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
    CheckHipErr(err);

    resDesc.res.array.array = cuArray1;
    hipSurfaceObject_t surf1 = 0;
    err = hipCreateSurfaceObject(&surf1, &resDesc);
    CheckHipErr(err);

    resDesc.res.array.array = cuArray2;
    hipSurfaceObject_t surf2 = 0;
    err = hipCreateSurfaceObject(&surf2, &resDesc);
    CheckHipErr(err);

    resDesc.res.array.array = cuArray3;
    hipSurfaceObject_t surf3 = 0;
    err = hipCreateSurfaceObject(&surf3, &resDesc);
    CheckHipErr(err);

    err = hipMemcpy2DToArrayAsync(cuArray0, 0, 0, inputData,
                                   param->stride, sizeof(typename Vec4<T>::type) * param->orgW, param->orgH,
                                   hipMemcpyHostToDevice, stream);
    CheckHipErr(err);

    dim3 dimBlock(16, 16);
    dim3 dimGrid(
        (param->W + dimBlock.x - 1) / dimBlock.x,
        (param->H + dimBlock.y - 1) / dimBlock.y);

    {
        int i;
        getGray<T><<<dimGrid, dimBlock, 0, stream>>>(tex, surf1, param->W, param->H);
        for (i = 0; i < param->passes && i < param->pushColorCount; i++)
        {
            pushColor<T><<<dimGrid, dimBlock, 0, stream>>>(surf1, surf2, param->W, param->H, param->strengthColor);
            getGradient<T><<<dimGrid, dimBlock, 0, stream>>>(surf2, surf3, param->W, param->H);
            pushGradient<T><<<dimGrid, dimBlock, 0, stream>>>(surf3, surf1, param->W, param->H, param->strengthGradient);
        }
        while (i++ < param->passes)
        {
            getGradient<T><<<dimGrid, dimBlock, 0, stream>>>(surf1, surf2, param->W, param->H);
            pushGradient<T><<<dimGrid, dimBlock, 0, stream>>>(surf2, surf1, param->W, param->H, param->strengthGradient);
        }
    }

    err = hipHostRegister(outputData, sizeof(typename Vec4<T>::type) * param->W * param->H, hipHostRegisterDefault);
    CheckHipErr(err);

    err = hipMemcpy2DFromArrayAsync(outputData, sizeof(typename Vec4<T>::type) * param->W, cuArray1, 0, 0,
                                     sizeof(typename Vec4<T>::type) * param->W, param->H,
                                     hipMemcpyDeviceToHost, stream);
    CheckHipErr(err);

    err = hipStreamSynchronize(stream);
    CheckHipErr(err);

    err = hipHostUnregister(outputData);
    CheckHipErr(err);

    err = hipGetLastError();
    CheckHipErr(err);

    hipDestroyTextureObject(tex);
    hipDestroySurfaceObject(surf1);
    hipDestroySurfaceObject(surf2);
    hipDestroySurfaceObject(surf3);

    hipFreeArray(cuArray0);
    hipFreeArray(cuArray1);
    hipFreeArray(cuArray2);
    hipFreeArray(cuArray3);

    // hipStreamDestroy(stream);
}

void Anime4KCPP::Hip::cuRunKernelAnime4K09(const void* inputData, void* outputData, ACHipDataType type, ACHipParamAnime4K09* param, hipStream_t& stream)
{
    switch (type)
    {
    case ACHipDataType::AC_8U:
        cuRunKernelAnime4K09Impl<uchar>(reinterpret_cast<const uchar *>(inputData), reinterpret_cast<uchar *>(outputData), param, stream);
        break;
    case ACHipDataType::AC_16U:
        cuRunKernelAnime4K09Impl<ushort>(reinterpret_cast<const ushort *>(inputData), reinterpret_cast<ushort *>(outputData), param, stream);
        break;
    case ACHipDataType::AC_32F:
        cuRunKernelAnime4K09Impl<float>(reinterpret_cast<const float *>(inputData), reinterpret_cast<float *>(outputData), param, stream);
        break;
    }
}
