#ifndef __ACNET_COMMON_CUH__
#define __ACNET_COMMON_CUH__

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

#define DECLARE_ACNET_HDN_INTERFACE_FUNCTION(level)                                                                                            \
    void Anime4KCPP::Cuda::cuRunKernelACNetHDN##level##B(const unsigned char *inputData, unsigned char *outputData, ACCudaParamACNet *param)   \
    {                                                                                                                                          \
        cuRunKernelACNetImpl<uchar>(inputData, outputData, param);                                                                             \
    }                                                                                                                                          \
    void Anime4KCPP::Cuda::cuRunKernelACNetHDN##level##W(const unsigned short *inputData, unsigned short *outputData, ACCudaParamACNet *param) \
    {                                                                                                                                          \
        cuRunKernelACNetImpl<ushort>(inputData, outputData, param);                                                                            \
    }                                                                                                                                          \
    void Anime4KCPP::Cuda::cuRunKernelACNetHDN##level##F(const float *inputData, float *outputData, ACCudaParamACNet *param)                   \
    {                                                                                                                                          \
        cuRunKernelACNetImpl<float>(inputData, outputData, param);                                                                             \
    }

constexpr static int L2 = 0, L3 = 1, L4 = 2, L5 = 3, L6 = 4, L7 = 5, L8 = 6, L9 = 7;

__global__ static void conv1To8(
    cudaTextureObject_t srcImg, cudaSurfaceObject_t dstImg,
    unsigned int W, unsigned int H)
{
    const unsigned int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const unsigned int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

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

    float4 c1234 = make_float4(
        RELU(CHANNEL1TO8(0)),
        RELU(CHANNEL1TO8(1)),
        RELU(CHANNEL1TO8(2)),
        RELU(CHANNEL1TO8(3)));
    float4 c5678 = make_float4(
        RELU(CHANNEL1TO8(4)),
        RELU(CHANNEL1TO8(5)),
        RELU(CHANNEL1TO8(6)),
        RELU(CHANNEL1TO8(7)));

    surf2DLayeredwrite(c1234, dstImg, __umul24(sizeof(c1234), x), y, 0, cudaBoundaryModeZero);
    surf2DLayeredwrite(c5678, dstImg, __umul24(sizeof(c5678), x), y, 1, cudaBoundaryModeZero);
}

__global__ static void conv8To8(
    cudaSurfaceObject_t srcImg, cudaSurfaceObject_t dstImg,
    unsigned int W, unsigned int H, int L)
{
    const unsigned int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const unsigned int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    if (x >= W || y >= H)
        return;

    float4 tl1, tc1, tr1, ml1, mc1, mr1, bl1, bc1, br1;
    float4 tl2, tc2, tr2, ml2, mc2, mr2, bl2, bc2, br2;

    surf2DLayeredread(&tl1, srcImg, __umul24(sizeof(mc1), x - 1), y - 1, 0, cudaBoundaryModeZero);
    surf2DLayeredread(&tc1, srcImg, __umul24(sizeof(mc1), x), y - 1, 0, cudaBoundaryModeZero);
    surf2DLayeredread(&tr1, srcImg, __umul24(sizeof(mc1), x + 1), y - 1, 0, cudaBoundaryModeZero);
    surf2DLayeredread(&ml1, srcImg, __umul24(sizeof(mc1), x - 1), y, 0, cudaBoundaryModeZero);
    surf2DLayeredread(&mc1, srcImg, __umul24(sizeof(mc1), x), y, 0, cudaBoundaryModeZero);
    surf2DLayeredread(&mr1, srcImg, __umul24(sizeof(mc1), x + 1), y, 0, cudaBoundaryModeZero);
    surf2DLayeredread(&bl1, srcImg, __umul24(sizeof(mc1), x - 1), y + 1, 0, cudaBoundaryModeZero);
    surf2DLayeredread(&bc1, srcImg, __umul24(sizeof(mc1), x), y + 1, 0, cudaBoundaryModeZero);
    surf2DLayeredread(&br1, srcImg, __umul24(sizeof(mc1), x + 1), y + 1, 0, cudaBoundaryModeZero);

    surf2DLayeredread(&tl2, srcImg, __umul24(sizeof(mc2), x - 1), y - 1, 1, cudaBoundaryModeZero);
    surf2DLayeredread(&tc2, srcImg, __umul24(sizeof(mc2), x), y - 1, 1, cudaBoundaryModeZero);
    surf2DLayeredread(&tr2, srcImg, __umul24(sizeof(mc2), x + 1), y - 1, 1, cudaBoundaryModeZero);
    surf2DLayeredread(&ml2, srcImg, __umul24(sizeof(mc2), x - 1), y, 1, cudaBoundaryModeZero);
    surf2DLayeredread(&mc2, srcImg, __umul24(sizeof(mc2), x), y, 1, cudaBoundaryModeZero);
    surf2DLayeredread(&mr2, srcImg, __umul24(sizeof(mc2), x + 1), y, 1, cudaBoundaryModeZero);
    surf2DLayeredread(&bl2, srcImg, __umul24(sizeof(mc2), x - 1), y + 1, 1, cudaBoundaryModeZero);
    surf2DLayeredread(&bc2, srcImg, __umul24(sizeof(mc2), x), y + 1, 1, cudaBoundaryModeZero);
    surf2DLayeredread(&br2, srcImg, __umul24(sizeof(mc2), x + 1), y + 1, 1, cudaBoundaryModeZero);

    float4 c1234 = make_float4(
        RELU(CHANNEL8TO8(0)),
        RELU(CHANNEL8TO8(1)),
        RELU(CHANNEL8TO8(2)),
        RELU(CHANNEL8TO8(3)));
    float4 c5678 = make_float4(
        RELU(CHANNEL8TO8(4)),
        RELU(CHANNEL8TO8(5)),
        RELU(CHANNEL8TO8(6)),
        RELU(CHANNEL8TO8(7)));

    surf2DLayeredwrite(c1234, dstImg, __umul24(sizeof(c1234), x), y, 0, cudaBoundaryModeZero);
    surf2DLayeredwrite(c5678, dstImg, __umul24(sizeof(c5678), x), y, 1, cudaBoundaryModeZero);
}

template <typename T>
__global__ static void convTranspose8To1(
    cudaSurfaceObject_t srcImg, cudaSurfaceObject_t dstImg,
    unsigned int W, unsigned int H)
{
    const unsigned int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const unsigned int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    if (x >= W || y >= H)
        return;

    int index = (y & 1) * 2 + (x & 1);

    float4 mc1, mc2;
    const unsigned int srcX = x / 2, srcY = y / 2;
    surf2DLayeredread(&mc1, srcImg, __umul24(sizeof(mc1), srcX), srcY, 0, cudaBoundaryModeZero);
    surf2DLayeredread(&mc2, srcImg, __umul24(sizeof(mc2), srcX), srcY, 1, cudaBoundaryModeZero);

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
                        mc2.w * kernelsL10[28 + index],
                    0.0f, 1.0f) *
                    scale +
                offset;

    surf2Dwrite(c, dstImg, __umul24(sizeof(c), x), y, cudaBoundaryModeZero);
}

template <typename T>
static void cuRunKernelACNetImpl(const T *inputData, T *outputData, Anime4KCPP::Cuda::ACCudaParamACNet *param)
{
    cudaError_t err = cudaSuccess;
    if (currCudaDeviceID)
    {
        err = cudaSetDevice(currCudaDeviceID);
        CheckCudaErr(err);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaChannelFormatDesc inoutChannelDesc = cudaCreateChannelDesc<T>();
    cudaChannelFormatDesc tmpChannelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaExtent extent = make_cudaExtent(param->orgW, param->orgH, 2);

    const int W = 2 * param->orgW, H = 2 * param->orgH;

    cudaArray_t cuInputArray;
    err = cudaMallocArray(&cuInputArray, &inoutChannelDesc,
                          param->orgW, param->orgH);
    CheckCudaErr(err);

    cudaArray_t cuArray1;
    err = cudaMalloc3DArray(&cuArray1, &tmpChannelDesc, extent,
                            cudaArraySurfaceLoadStore | cudaArrayLayered);
    CheckCudaErr(err);

    cudaArray_t cuArray2;
    err = cudaMalloc3DArray(&cuArray2, &tmpChannelDesc, extent,
                            cudaArraySurfaceLoadStore | cudaArrayLayered);
    CheckCudaErr(err);

    cudaArray_t cuOutputArray;
    err = cudaMallocArray(&cuOutputArray, &inoutChannelDesc,
                          W, H, cudaArraySurfaceLoadStore);
    CheckCudaErr(err);

    struct cudaResourceDesc resDesc;
    struct cudaTextureDesc texDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    memset(&texDesc, 0, sizeof(texDesc));

    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.readMode = std::is_floating_point<T>::value ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 0;

    resDesc.resType = cudaResourceTypeArray;

    resDesc.res.array.array = cuInputArray;
    cudaTextureObject_t inTex = 0;
    err = cudaCreateTextureObject(&inTex, &resDesc, &texDesc, NULL);
    CheckCudaErr(err);

    resDesc.res.array.array = cuArray1;
    cudaSurfaceObject_t surf1 = 0;
    err = cudaCreateSurfaceObject(&surf1, &resDesc);
    CheckCudaErr(err);

    resDesc.res.array.array = cuArray2;
    cudaSurfaceObject_t surf2 = 0;
    err = cudaCreateSurfaceObject(&surf2, &resDesc);
    CheckCudaErr(err);

    resDesc.res.array.array = cuOutputArray;
    cudaSurfaceObject_t outSurf = 0;
    err = cudaCreateSurfaceObject(&outSurf, &resDesc);
    CheckCudaErr(err);

    err = cudaMemcpy2DToArrayAsync(cuInputArray, 0, 0, inputData,
                                   param->stride, sizeof(T) * param->orgW, param->orgH,
                                   cudaMemcpyHostToDevice, stream);
    CheckCudaErr(err);

    dim3 dimBlock(16, 16);
    dim3 dimGrid(
        (param->orgW + dimBlock.x - 1) / dimBlock.x,
        (param->orgH + dimBlock.y - 1) / dimBlock.y);
    dim3 dimGridout(
        (param->orgW * 2 + dimBlock.x - 1) / dimBlock.x,
        (param->orgH * 2 + dimBlock.y - 1) / dimBlock.y);

    conv1To8<<<dimGrid, dimBlock, 0, stream>>>(inTex, surf1, param->orgW, param->orgH);
    conv8To8<<<dimGrid, dimBlock, 0, stream>>>(surf1, surf2, param->orgW, param->orgH, L2);
    conv8To8<<<dimGrid, dimBlock, 0, stream>>>(surf2, surf1, param->orgW, param->orgH, L3);
    conv8To8<<<dimGrid, dimBlock, 0, stream>>>(surf1, surf2, param->orgW, param->orgH, L4);
    conv8To8<<<dimGrid, dimBlock, 0, stream>>>(surf2, surf1, param->orgW, param->orgH, L5);
    conv8To8<<<dimGrid, dimBlock, 0, stream>>>(surf1, surf2, param->orgW, param->orgH, L6);
    conv8To8<<<dimGrid, dimBlock, 0, stream>>>(surf2, surf1, param->orgW, param->orgH, L7);
    conv8To8<<<dimGrid, dimBlock, 0, stream>>>(surf1, surf2, param->orgW, param->orgH, L8);
    conv8To8<<<dimGrid, dimBlock, 0, stream>>>(surf2, surf1, param->orgW, param->orgH, L9);
    convTranspose8To1<T><<<dimGridout, dimBlock, 0, stream>>>(surf1, outSurf, W, H);

    err = cudaHostRegister(outputData, sizeof(T) * W * H, cudaHostRegisterDefault);
    CheckCudaErr(err);

    err = cudaMemcpy2DFromArrayAsync(outputData, sizeof(T) * W,
                                     cuOutputArray, 0, 0, sizeof(T) * W, H,
                                     cudaMemcpyDeviceToHost, stream);
    CheckCudaErr(err);

    err = cudaStreamSynchronize(stream);
    CheckCudaErr(err);

    err = cudaHostUnregister(outputData);
    CheckCudaErr(err);

    err = cudaGetLastError();
    CheckCudaErr(err);

    cudaDestroyTextureObject(inTex);
    cudaDestroySurfaceObject(surf1);
    cudaDestroySurfaceObject(surf2);
    cudaDestroySurfaceObject(outSurf);

    cudaFreeArray(cuInputArray);
    cudaFreeArray(cuArray1);
    cudaFreeArray(cuArray2);
    cudaFreeArray(cuOutputArray);

    cudaStreamDestroy(stream);
}

#endif
