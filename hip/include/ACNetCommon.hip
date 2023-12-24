#include "hip/hip_runtime.h"
#ifndef ANIME4KCPP_HIP_ACNET_COMMON_HIP
#define ANIME4KCPP_HIP_ACNET_COMMON_HIP

#define RELU(x) fmaxf(x, 0.0f)

#define CHANNEL1TO8(n) \
    tl *kernelsL1[n * 9 + 0] + tc *kernelsL1[n * 9 + 1] + tr *kernelsL1[n * 9 + 2] + \
    ml *kernelsL1[n * 9 + 3] + mc *kernelsL1[n * 9 + 4] + mr *kernelsL1[n * 9 + 5] + \
    bl *kernelsL1[n * 9 + 6] + bc *kernelsL1[n * 9 + 7] + br *kernelsL1[n * 9 + 8] + biasL1[n]

#define CHANNEL8TO8(n) \
    tl1.x *kernelsL[L][n * 72 + 0 * 9 + 0] + tc1.x *kernelsL[L][n * 72 + 0 * 9 + 1] + tr1.x *kernelsL[L][n * 72 + 0 * 9 + 2] + \
    ml1.x *kernelsL[L][n * 72 + 0 * 9 + 3] + mc1.x *kernelsL[L][n * 72 + 0 * 9 + 4] + mr1.x *kernelsL[L][n * 72 + 0 * 9 + 5] + \
    bl1.x *kernelsL[L][n * 72 + 0 * 9 + 6] + bc1.x *kernelsL[L][n * 72 + 0 * 9 + 7] + br1.x *kernelsL[L][n * 72 + 0 * 9 + 8] + \
    tl1.y *kernelsL[L][n * 72 + 1 * 9 + 0] + tc1.y *kernelsL[L][n * 72 + 1 * 9 + 1] + tr1.y *kernelsL[L][n * 72 + 1 * 9 + 2] + \
    ml1.y *kernelsL[L][n * 72 + 1 * 9 + 3] + mc1.y *kernelsL[L][n * 72 + 1 * 9 + 4] + mr1.y *kernelsL[L][n * 72 + 1 * 9 + 5] + \
    bl1.y *kernelsL[L][n * 72 + 1 * 9 + 6] + bc1.y *kernelsL[L][n * 72 + 1 * 9 + 7] + br1.y *kernelsL[L][n * 72 + 1 * 9 + 8] + \
    tl1.z *kernelsL[L][n * 72 + 2 * 9 + 0] + tc1.z *kernelsL[L][n * 72 + 2 * 9 + 1] + tr1.z *kernelsL[L][n * 72 + 2 * 9 + 2] + \
    ml1.z *kernelsL[L][n * 72 + 2 * 9 + 3] + mc1.z *kernelsL[L][n * 72 + 2 * 9 + 4] + mr1.z *kernelsL[L][n * 72 + 2 * 9 + 5] + \
    bl1.z *kernelsL[L][n * 72 + 2 * 9 + 6] + bc1.z *kernelsL[L][n * 72 + 2 * 9 + 7] + br1.z *kernelsL[L][n * 72 + 2 * 9 + 8] + \
    tl1.w *kernelsL[L][n * 72 + 3 * 9 + 0] + tc1.w *kernelsL[L][n * 72 + 3 * 9 + 1] + tr1.w *kernelsL[L][n * 72 + 3 * 9 + 2] + \
    ml1.w *kernelsL[L][n * 72 + 3 * 9 + 3] + mc1.w *kernelsL[L][n * 72 + 3 * 9 + 4] + mr1.w *kernelsL[L][n * 72 + 3 * 9 + 5] + \
    bl1.w *kernelsL[L][n * 72 + 3 * 9 + 6] + bc1.w *kernelsL[L][n * 72 + 3 * 9 + 7] + br1.w *kernelsL[L][n * 72 + 3 * 9 + 8] + \
    tl2.x *kernelsL[L][n * 72 + 4 * 9 + 0] + tc2.x *kernelsL[L][n * 72 + 4 * 9 + 1] + tr2.x *kernelsL[L][n * 72 + 4 * 9 + 2] + \
    ml2.x *kernelsL[L][n * 72 + 4 * 9 + 3] + mc2.x *kernelsL[L][n * 72 + 4 * 9 + 4] + mr2.x *kernelsL[L][n * 72 + 4 * 9 + 5] + \
    bl2.x *kernelsL[L][n * 72 + 4 * 9 + 6] + bc2.x *kernelsL[L][n * 72 + 4 * 9 + 7] + br2.x *kernelsL[L][n * 72 + 4 * 9 + 8] + \
    tl2.y *kernelsL[L][n * 72 + 5 * 9 + 0] + tc2.y *kernelsL[L][n * 72 + 5 * 9 + 1] + tr2.y *kernelsL[L][n * 72 + 5 * 9 + 2] + \
    ml2.y *kernelsL[L][n * 72 + 5 * 9 + 3] + mc2.y *kernelsL[L][n * 72 + 5 * 9 + 4] + mr2.y *kernelsL[L][n * 72 + 5 * 9 + 5] + \
    bl2.y *kernelsL[L][n * 72 + 5 * 9 + 6] + bc2.y *kernelsL[L][n * 72 + 5 * 9 + 7] + br2.y *kernelsL[L][n * 72 + 5 * 9 + 8] + \
    tl2.z *kernelsL[L][n * 72 + 6 * 9 + 0] + tc2.z *kernelsL[L][n * 72 + 6 * 9 + 1] + tr2.z *kernelsL[L][n * 72 + 6 * 9 + 2] + \
    ml2.z *kernelsL[L][n * 72 + 6 * 9 + 3] + mc2.z *kernelsL[L][n * 72 + 6 * 9 + 4] + mr2.z *kernelsL[L][n * 72 + 6 * 9 + 5] + \
    bl2.z *kernelsL[L][n * 72 + 6 * 9 + 6] + bc2.z *kernelsL[L][n * 72 + 6 * 9 + 7] + br2.z *kernelsL[L][n * 72 + 6 * 9 + 8] + \
    tl2.w *kernelsL[L][n * 72 + 7 * 9 + 0] + tc2.w *kernelsL[L][n * 72 + 7 * 9 + 1] + tr2.w *kernelsL[L][n * 72 + 7 * 9 + 2] + \
    ml2.w *kernelsL[L][n * 72 + 7 * 9 + 3] + mc2.w *kernelsL[L][n * 72 + 7 * 9 + 4] + mr2.w *kernelsL[L][n * 72 + 7 * 9 + 5] + \
    bl2.w *kernelsL[L][n * 72 + 7 * 9 + 6] + bc2.w *kernelsL[L][n * 72 + 7 * 9 + 7] + br2.w *kernelsL[L][n * 72 + 7 * 9 + 8] + biasL[L][n]

#define DECLARE_ACNET_HDN_INTERFACE_FUNCTION(level)                                                                                          \
    void Anime4KCPP::Hip::cuRunKernelACNetHDN##level(const void *inputData, void *outputData, ACHipDataType type, ACHipParamACNet *param) \
    {                                                                                                                                        \
        switch (type)                                                                                                                        \
        {                                                                                                                                    \
        case ACHipDataType::AC_8U:                                                                                                          \
            cuRunKernelACNetImpl<uchar>(reinterpret_cast<const uchar *>(inputData), reinterpret_cast<uchar *>(outputData), param);           \
            break;                                                                                                                           \
        case ACHipDataType::AC_16U:                                                                                                         \
            cuRunKernelACNetImpl<ushort>(reinterpret_cast<const ushort *>(inputData), reinterpret_cast<ushort *>(outputData), param);        \
            break;                                                                                                                           \
        case ACHipDataType::AC_32F:                                                                                                         \
            cuRunKernelACNetImpl<float>(reinterpret_cast<const float *>(inputData), reinterpret_cast<float *>(outputData), param);           \
            break;                                                                                                                           \
        }                                                                                                                                    \
    }

static constexpr int L2 = 0, L3 = 1, L4 = 2, L5 = 3, L6 = 4, L7 = 5, L8 = 6, L9 = 7;

__global__ static void conv1To8(
    hipTextureObject_t srcImg, hipSurfaceObject_t dstImg,
    unsigned int W, unsigned int H)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H)
        return;

    const float tl = tex2D<float>(srcImg, x - 1, y - 1);
    const float tc = tex2D<float>(srcImg, x, y - 1);
    const float tr = tex2D<float>(srcImg, x + 1, y - 1);
    const float ml = tex2D<float>(srcImg, x - 1, y);
    const float mc = tex2D<float>(srcImg, x, y);
    const float mr = tex2D<float>(srcImg, x + 1, y);
    const float bl = tex2D<float>(srcImg, x - 1, y + 1);
    const float bc = tex2D<float>(srcImg, x, y + 1);
    const float br = tex2D<float>(srcImg, x + 1, y + 1);

    const float4 fc1234 = make_float4(
        RELU(CHANNEL1TO8(0)),
        RELU(CHANNEL1TO8(1)),
        RELU(CHANNEL1TO8(2)),
        RELU(CHANNEL1TO8(3)));
    const float4 fc5678 = make_float4(
        RELU(CHANNEL1TO8(4)),
        RELU(CHANNEL1TO8(5)),
        RELU(CHANNEL1TO8(6)),
        RELU(CHANNEL1TO8(7)));

    half c1234[4] = {__float2half(fc1234.x), __float2half(fc1234.y), __float2half(fc1234.z), __float2half(fc1234.w)};
    half c5678[4] = {__float2half(fc5678.x), __float2half(fc5678.y), __float2half(fc5678.z), __float2half(fc5678.w)};

    surf2DLayeredwrite(*reinterpret_cast<ushort4*>(c1234), dstImg, sizeof(c1234) * x, y, 0);
    surf2DLayeredwrite(*reinterpret_cast<ushort4*>(c5678), dstImg, sizeof(c5678) * x, y, 1);
}

__global__ static void conv8To8(
    hipTextureObject_t srcImg, hipSurfaceObject_t dstImg,
    unsigned int W, unsigned int H, int L)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H)
        return;

    const float4 tl1 = tex2DLayered<float4>(srcImg, x - 1, y - 1, 0);
    const float4 tc1 = tex2DLayered<float4>(srcImg, x, y - 1, 0);
    const float4 tr1 = tex2DLayered<float4>(srcImg, x + 1, y - 1, 0);
    const float4 ml1 = tex2DLayered<float4>(srcImg, x - 1, y, 0);
    const float4 mc1 = tex2DLayered<float4>(srcImg, x, y, 0);
    const float4 mr1 = tex2DLayered<float4>(srcImg, x + 1, y, 0);
    const float4 bl1 = tex2DLayered<float4>(srcImg, x - 1, y + 1, 0);
    const float4 bc1 = tex2DLayered<float4>(srcImg, x, y + 1, 0);
    const float4 br1 = tex2DLayered<float4>(srcImg, x + 1, y + 1, 0);

    const float4 tl2 = tex2DLayered<float4>(srcImg, x - 1, y - 1, 1);
    const float4 tc2 = tex2DLayered<float4>(srcImg, x, y - 1, 1);
    const float4 tr2 = tex2DLayered<float4>(srcImg, x + 1, y - 1, 1);
    const float4 ml2 = tex2DLayered<float4>(srcImg, x - 1, y, 1);
    const float4 mc2 = tex2DLayered<float4>(srcImg, x, y, 1);
    const float4 mr2 = tex2DLayered<float4>(srcImg, x + 1, y, 1);
    const float4 bl2 = tex2DLayered<float4>(srcImg, x - 1, y + 1, 1);
    const float4 bc2 = tex2DLayered<float4>(srcImg, x, y + 1, 1);
    const float4 br2 = tex2DLayered<float4>(srcImg, x + 1, y + 1, 1);

    const float4 fc1234 = make_float4(
        RELU(CHANNEL8TO8(0)),
        RELU(CHANNEL8TO8(1)),
        RELU(CHANNEL8TO8(2)),
        RELU(CHANNEL8TO8(3)));
    const float4 fc5678 = make_float4(
        RELU(CHANNEL8TO8(4)),
        RELU(CHANNEL8TO8(5)),
        RELU(CHANNEL8TO8(6)),
        RELU(CHANNEL8TO8(7)));

    half c1234[4] = {__float2half(fc1234.x), __float2half(fc1234.y), __float2half(fc1234.z), __float2half(fc1234.w)};
    half c5678[4] = {__float2half(fc5678.x), __float2half(fc5678.y), __float2half(fc5678.z), __float2half(fc5678.w)};

    surf2DLayeredwrite(*reinterpret_cast<ushort4*>(c1234), dstImg, sizeof(c1234) * x, y, 0);
    surf2DLayeredwrite(*reinterpret_cast<ushort4*>(c5678), dstImg, sizeof(c5678) * x, y, 1);
}

template <typename T>
__global__ static void convTranspose8To1(
    hipTextureObject_t srcImg, hipSurfaceObject_t dstImg,
    unsigned int W, unsigned int H)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H)
        return;

    const int index = (y & 1) * 2 + (x & 1);

    const unsigned int srcX = x / 2, srcY = y / 2;

    const float4 mc1 = tex2DLayered<float4>(srcImg, srcX, srcY, 0);
    const float4 mc2 = tex2DLayered<float4>(srcImg, srcX, srcY, 1);

    constexpr float scale = PixelValue<T>::max();
    constexpr float offset = std::is_floating_point<T>::value ? 0.0f : 0.5f;

    const T c = clamp(
                    mc1.x * kernelsL10[0 + index] +
                    mc1.y * kernelsL10[4 + index] +
                    mc1.z * kernelsL10[8 + index] +
                    mc1.w * kernelsL10[12 + index] +
                    mc2.x * kernelsL10[16 + index] +
                    mc2.y * kernelsL10[20 + index] +
                    mc2.z * kernelsL10[24 + index] +
                    mc2.w * kernelsL10[28 + index], 0.0f, 1.0f) * scale + offset;

    surf2Dwrite(c, dstImg, sizeof(c) * x, y);
}

template <typename T>
static void cuRunKernelACNetImpl(const T *inputData, T *outputData, Anime4KCPP::Hip::ACHipParamACNet *param)
{
    hipError_t err = hipSuccess;
    if (currHipDeviceID)
    {
        err = hipSetDevice(currHipDeviceID);
        CheckHipErr(err);
    }

    // hipStream_t stream;
    // hipStreamCreate(&stream);

    hipChannelFormatDesc inoutChannelDesc = hipCreateChannelDesc<T>();
    hipChannelFormatDesc tmpChannelDesc = hipCreateChannelDesc<T>();
    hipExtent extent = make_hipExtent(param->orgW, param->orgH, 2);

    const int W = 2 * param->orgW, H = 2 * param->orgH;

    hipArray_t cuInputArray;
    err = hipMallocArray(&cuInputArray, &inoutChannelDesc,
                          param->orgW, param->orgH);
    CheckHipErr(err);

    hipArray_t cuArray1;
    err = hipMalloc3DArray(&cuArray1, &tmpChannelDesc, extent,
                            /* hipArraySurfaceLoadStore | */ hipArrayLayered);
    CheckHipErr(err);

    hipArray_t cuArray2;
    err = hipMalloc3DArray(&cuArray2, &tmpChannelDesc, extent,
                            /* hipArraySurfaceLoadStore | */ hipArrayLayered);
    CheckHipErr(err);

    hipArray_t cuOutputArray;
    err = hipMallocArray(&cuOutputArray, &inoutChannelDesc,
                          W, H, hipArrayDefault /* hipArraySurfaceLoadStore */);
    CheckHipErr(err);

    struct hipResourceDesc resDesc;
    struct hipTextureDesc texDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    memset(&texDesc, 0, sizeof(texDesc));

    texDesc.addressMode[0] = hipAddressModeBorder;
    texDesc.addressMode[1] = hipAddressModeBorder;
    texDesc.addressMode[2] = hipAddressModeBorder;
    texDesc.readMode = std::is_floating_point<T>::value ? hipReadModeElementType : hipReadModeNormalizedFloat;
    texDesc.normalizedCoords = 0;

    resDesc.resType = hipResourceTypeArray;

    resDesc.res.array.array = cuInputArray;
    hipTextureObject_t inTex = 0;
    err = hipCreateTextureObject(&inTex, &resDesc, &texDesc, NULL);
    CheckHipErr(err);

    if(!std::is_floating_point<T>::value)
        texDesc.readMode = hipReadModeElementType;

    resDesc.res.array.array = cuArray1;
    hipSurfaceObject_t surf1 = 0;
    err = hipCreateSurfaceObject(&surf1, &resDesc);
    CheckHipErr(err);

    hipTextureObject_t tex1 = 0;
    err = hipCreateTextureObject(&tex1, &resDesc, &texDesc, NULL);
    CheckHipErr(err);

    resDesc.res.array.array = cuArray2;
    hipSurfaceObject_t surf2 = 0;
    err = hipCreateSurfaceObject(&surf2, &resDesc);
    CheckHipErr(err);

    hipTextureObject_t tex2 = 0;
    err = hipCreateTextureObject(&tex2, &resDesc, &texDesc, NULL);
    CheckHipErr(err);

    resDesc.res.array.array = cuOutputArray;
    hipSurfaceObject_t outSurf = 0;
    err = hipCreateSurfaceObject(&outSurf, &resDesc);
    CheckHipErr(err);

    err = hipMemcpy2DToArray(cuInputArray, 0, 0, inputData,
                                   param->stride, sizeof(T) * param->orgW, param->orgH,
                                   hipMemcpyHostToDevice);
    CheckHipErr(err);

    dim3 dimBlock(16, 16);
    dim3 dimGrid(
        (param->orgW + dimBlock.x - 1) / dimBlock.x,
        (param->orgH + dimBlock.y - 1) / dimBlock.y);
    dim3 dimGridout(
        (param->orgW * 2 + dimBlock.x - 1) / dimBlock.x,
        (param->orgH * 2 + dimBlock.y - 1) / dimBlock.y);

    conv1To8<<<dimGrid, dimBlock, 0>>>(inTex, surf1, param->orgW, param->orgH);
    conv8To8<<<dimGrid, dimBlock, 0>>>(tex1, surf2, param->orgW, param->orgH, L2);
    conv8To8<<<dimGrid, dimBlock, 0>>>(tex2, surf1, param->orgW, param->orgH, L3);
    conv8To8<<<dimGrid, dimBlock, 0>>>(tex1, surf2, param->orgW, param->orgH, L4);
    conv8To8<<<dimGrid, dimBlock, 0>>>(tex2, surf1, param->orgW, param->orgH, L5);
    conv8To8<<<dimGrid, dimBlock, 0>>>(tex1, surf2, param->orgW, param->orgH, L6);
    conv8To8<<<dimGrid, dimBlock, 0>>>(tex2, surf1, param->orgW, param->orgH, L7);
    conv8To8<<<dimGrid, dimBlock, 0>>>(tex1, surf2, param->orgW, param->orgH, L8);
    conv8To8<<<dimGrid, dimBlock, 0>>>(tex2, surf1, param->orgW, param->orgH, L9);
    convTranspose8To1<T><<<dimGridout, dimBlock, 0>>>(tex1, outSurf, W, H);

    err = hipHostRegister(outputData, sizeof(T) * W * H, hipHostRegisterDefault);
    CheckHipErr(err);

    err = hipMemcpy2DFromArray(outputData, sizeof(T) * W,
                                     cuOutputArray, 0, 0, sizeof(T) * W, H,
                                     hipMemcpyDeviceToHost);
    CheckHipErr(err);

    // err = hipStreamSynchronize(stream);
    // CheckHipErr(err);

    err = hipHostUnregister(outputData);
    CheckHipErr(err);

    err = hipGetLastError();
    CheckHipErr(err);

    hipDestroyTextureObject(inTex);
    hipDestroySurfaceObject(surf1);
    hipDestroyTextureObject(tex1);
    hipDestroySurfaceObject(surf2);
    hipDestroyTextureObject(tex2);
    hipDestroySurfaceObject(outSurf);

    hipFreeArray(cuInputArray);
    hipFreeArray(cuArray1);
    hipFreeArray(cuArray2);
    hipFreeArray(cuOutputArray);

    // hipStreamDestroy(stream);
}

#endif // !ANIME4KCPP_HIP_ACNET_COMMON_HIP
