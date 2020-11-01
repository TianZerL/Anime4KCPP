#include"CudaHelper.cuh"
#include"CudaInterface.hpp"

typedef unsigned char uchar;

#define MAX3(a, b, c) fmaxf(fmaxf(a,b),c)
#define MIN3(a, b, c) fminf(fminf(a,b),c)
#define UNFLOAT(n) ((n) >= 255 ? 255 : ((n) <= 0 ? 0 : uchar((n) + 0.5)))

inline __device__ static void getLightest(uchar4& mc, uchar4& a, uchar4& b, uchar4& c, float strength)
{
    mc = make_uchar4(
        mc.x + strength * (__fdividef(a.x + b.x + c.x, 3.0f) - mc.x) + 0.5f,
        mc.y + strength * (__fdividef(a.y + b.y + c.y, 3.0f) - mc.y) + 0.5f,
        mc.z + strength * (__fdividef(a.z + b.z + c.z, 3.0f) - mc.z) + 0.5f,
        mc.w + strength * (__fdividef(a.w + b.w + c.w, 3.0f) - mc.w) + 0.5f
    );
}

inline __device__ static void getAVerage(uchar4& mc, uchar4& a, uchar4& b, uchar4& c, float strength)
{
    mc = make_uchar4(
        mc.x + strength * (__fdividef(a.x + b.x + c.x, 3.0f) - mc.x) + 0.5f,
        mc.y + strength * (__fdividef(a.y + b.y + c.y, 3.0f) - mc.y) + 0.5f,
        mc.z + strength * (__fdividef(a.z + b.z + c.z, 3.0f) - mc.z) + 0.5f,
        0.299f * mc.z + 0.587f * mc.y + 0.114f * mc.x + 0.5f
    );
}

__global__ static void getGray(
    cudaTextureObject_t srcImg, cudaSurfaceObject_t dstImg,
    int W, int H
)
{
    const unsigned int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const unsigned int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    if (x >= W || y >= H)
        return;

    float u = (x + 0.5f) / (float)(W);
    float v = (y + 0.5f) / (float)(H);

    float4 fmc = tex2D<float4>(srcImg, u, v);
    uchar4 mc = make_uchar4(
        fmc.x * 255.0f + 0.5f, fmc.y * 255.0f + 0.5f, fmc.z * 255.0f + 0.5f, fmc.w * 255.0f + 0.5f
    );
    mc.w = 0.299f * mc.z + 0.587f * mc.y + 0.114f * mc.x + 0.5f;
    surf2Dwrite(mc, dstImg, sizeof(mc) * x, y, cudaBoundaryModeZero);
}

__global__ static void pushColor(
    cudaSurfaceObject_t srcImg, cudaSurfaceObject_t dstImg,
    int W, int H, float strength
)
{
    const unsigned int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const unsigned int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    if (x >= W || y >= H)
        return;

    uchar4 tl, tc, tr, ml, mc, mr, bl, bc, br;
    surf2Dread(&tl, srcImg, __umul24(sizeof(mc), x - 1), y - 1, cudaBoundaryModeZero);
    surf2Dread(&tc, srcImg, __umul24(sizeof(mc), x), y - 1, cudaBoundaryModeZero);
    surf2Dread(&tr, srcImg, __umul24(sizeof(mc), x + 1), y - 1, cudaBoundaryModeZero);
    surf2Dread(&ml, srcImg, __umul24(sizeof(mc), x - 1), y, cudaBoundaryModeZero);
    surf2Dread(&mc, srcImg, __umul24(sizeof(mc), x), y, cudaBoundaryModeZero);
    surf2Dread(&mr, srcImg, __umul24(sizeof(mc), x + 1), y, cudaBoundaryModeZero);
    surf2Dread(&bl, srcImg, __umul24(sizeof(mc), x - 1), y + 1, cudaBoundaryModeZero);
    surf2Dread(&bc, srcImg, __umul24(sizeof(mc), x), y + 1, cudaBoundaryModeZero);
    surf2Dread(&br, srcImg, __umul24(sizeof(mc), x + 1), y + 1, cudaBoundaryModeZero);

    uchar maxD, minL;

    //top and bottom
    maxD = MAX3(bl.w, bc.w, br.w);
    minL = MIN3(tl.w, tc.w, tr.w);
    if (minL > mc.w && mc.w > maxD)
        getLightest(mc, tl, tc, tr, strength);
    else
    {
        maxD = MAX3(tl.w, tc.w, tr.w);
        minL = MIN3(bl.w, bc.w, br.w);
        if (minL > mc.w && mc.w > maxD)
            getLightest(mc, bl, bc, br, strength);
    }

    //sundiagonal
    maxD = MAX3(ml.w, mc.w, bc.w);
    minL = MIN3(tc.w, tr.w, mr.w);
    if (minL > maxD)
        getLightest(mc, tc, tr, mr, strength);
    else
    {
        maxD = MAX3(tc.w, mc.w, mr.w);
        minL = MIN3(ml.w, bl.w, bc.w);
        if (minL > maxD)
            getLightest(mc, ml, bl, bc, strength);
    }

    //left and right
    maxD = MAX3(tl.w, ml.w, bl.w);
    minL = MIN3(tr.w, mr.w, br.w);
    if (minL > mc.w && mc.w > maxD)
        getLightest(mc, tr, mr, br, strength);
    else
    {
        maxD = MAX3(tr.w, mr.w, br.w);
        minL = MIN3(tl.w, ml.w, bl.w);
        if (minL > mc.w && mc.w > maxD)
            getLightest(mc, tl, ml, bl, strength);
    }

    //diagonal
    maxD = MAX3(tc.w, mc.w, ml.w);
    minL = MIN3(mr.w, br.w, bc.w);
    if (minL > maxD)
        getLightest(mc, mr, br, bc, strength);
    else
    {
        maxD = MAX3(bc.w, mc.w, mr.w);
        minL = MIN3(ml.w, tl.w, tc.w);
        if (minL > maxD)
            getLightest(mc, ml, tl, tc, strength);
    }

    surf2Dwrite(mc, dstImg, sizeof(mc) * x, y, cudaBoundaryModeZero);
}

__global__ static void getGradient(
    cudaSurfaceObject_t srcImg, cudaSurfaceObject_t dstImg,
    int W, int H
)
{
    const unsigned int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const unsigned int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    if (x >= W || y >= H)
        return;

    uchar4 tl, tc, tr, ml, mc, mr, bl, bc, br;
    surf2Dread(&tl, srcImg, __umul24(sizeof(mc), x - 1), y - 1, cudaBoundaryModeZero);
    surf2Dread(&tc, srcImg, __umul24(sizeof(mc), x), y - 1, cudaBoundaryModeZero);
    surf2Dread(&tr, srcImg, __umul24(sizeof(mc), x + 1), y - 1, cudaBoundaryModeZero);
    surf2Dread(&ml, srcImg, __umul24(sizeof(mc), x - 1), y, cudaBoundaryModeZero);
    surf2Dread(&mc, srcImg, __umul24(sizeof(mc), x), y, cudaBoundaryModeZero);
    surf2Dread(&mr, srcImg, __umul24(sizeof(mc), x + 1), y, cudaBoundaryModeZero);
    surf2Dread(&bl, srcImg, __umul24(sizeof(mc), x - 1), y + 1, cudaBoundaryModeZero);
    surf2Dread(&bc, srcImg, __umul24(sizeof(mc), x), y + 1, cudaBoundaryModeZero);
    surf2Dread(&br, srcImg, __umul24(sizeof(mc), x + 1), y + 1, cudaBoundaryModeZero);

    const float gradX = tr.w + mr.w + mr.w + br.w - tl.w - ml.w - ml.w - bl.w;
    const float gradY = tl.w + tc.w + tc.w + tr.w - bl.w - bc.w - bc.w - br.w;

    const int grad = sqrtf(gradX * gradX + gradY * gradY);
    mc.w = (uchar)255 - UNFLOAT(grad);

    surf2Dwrite(mc, dstImg, sizeof(mc) * x, y, cudaBoundaryModeZero);
}

__global__ static void pushGradient(
    cudaSurfaceObject_t srcImg, cudaSurfaceObject_t dstImg,
    int W, int H, float strength
)
{
    const unsigned int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const unsigned int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    if (x >= W || y >= H)
        return;

    uchar4 tl, tc, tr, ml, mc, mr, bl, bc, br;
    surf2Dread(&tl, srcImg, __umul24(sizeof(mc), x - 1), y - 1, cudaBoundaryModeZero);
    surf2Dread(&tc, srcImg, __umul24(sizeof(mc), x), y - 1, cudaBoundaryModeZero);
    surf2Dread(&tr, srcImg, __umul24(sizeof(mc), x + 1), y - 1, cudaBoundaryModeZero);
    surf2Dread(&ml, srcImg, __umul24(sizeof(mc), x - 1), y, cudaBoundaryModeZero);
    surf2Dread(&mc, srcImg, __umul24(sizeof(mc), x), y, cudaBoundaryModeZero);
    surf2Dread(&mr, srcImg, __umul24(sizeof(mc), x + 1), y, cudaBoundaryModeZero);
    surf2Dread(&bl, srcImg, __umul24(sizeof(mc), x - 1), y + 1, cudaBoundaryModeZero);
    surf2Dread(&bc, srcImg, __umul24(sizeof(mc), x), y + 1, cudaBoundaryModeZero);
    surf2Dread(&br, srcImg, __umul24(sizeof(mc), x + 1), y + 1, cudaBoundaryModeZero);

    uchar maxD, minL;

    //top and bottom
    maxD = MAX3(bl.w, bc.w, br.w);
    minL = MIN3(tl.w, tc.w, tr.w);
    if (minL > mc.w && mc.w > maxD)
    {
        getAVerage(mc, tl, tc, tr, strength);
        surf2Dwrite(mc, dstImg, sizeof(mc) * x, y, cudaBoundaryModeZero);
        return;
    }

    maxD = MAX3(tl.w, tc.w, tr.w);
    minL = MIN3(bl.w, bc.w, br.w);
    if (minL > mc.w && mc.w > maxD)
    {
        getAVerage(mc, bl, bc, br, strength);
        surf2Dwrite(mc, dstImg, sizeof(mc) * x, y, cudaBoundaryModeZero);
        return;
    }

    //sundiagonal
    maxD = MAX3(ml.w, mc.w, bc.w);
    minL = MIN3(tc.w, tr.w, mr.w);
    if (minL > maxD)
    {
        getAVerage(mc, tc, tr, mr, strength);
        surf2Dwrite(mc, dstImg, sizeof(mc) * x, y, cudaBoundaryModeZero);
        return;
    }

    maxD = MAX3(tc.w, mc.w, mr.w);
    minL = MIN3(ml.w, bl.w, bc.w);
    if (minL > maxD)
    {
        getAVerage(mc, ml, bl, bc, strength);
        surf2Dwrite(mc, dstImg, sizeof(mc) * x, y, cudaBoundaryModeZero);
        return;
    }

    //left and right
    maxD = MAX3(tl.w, ml.w, bl.w);
    minL = MIN3(tr.w, mr.w, br.w);
    if (minL > mc.w && mc.w > maxD)
    {
        getAVerage(mc, tr, mr, br, strength);
        surf2Dwrite(mc, dstImg, sizeof(mc) * x, y, cudaBoundaryModeZero);
        return;
    }

    maxD = MAX3(tr.w, mr.w, br.w);
    minL = MIN3(tl.w, ml.w, bl.w);
    if (minL > mc.w && mc.w > maxD)
    {
        getAVerage(mc, tl, ml, bl, strength);
        surf2Dwrite(mc, dstImg, sizeof(mc) * x, y, cudaBoundaryModeZero);
        return;
    }

    //diagonal
    maxD = MAX3(tc.w, mc.w, ml.w);
    minL = MIN3(mr.w, br.w, bc.w);
    if (minL > maxD)
    {
        getAVerage(mc, mr, br, bc, strength);
        surf2Dwrite(mc, dstImg, sizeof(mc) * x, y, cudaBoundaryModeZero);
        return;
    }
    maxD = MAX3(bc.w, mc.w, mr.w);
    minL = MIN3(ml.w, tl.w, tc.w);
    if (minL > maxD)
    {
        getAVerage(mc, ml, tl, tc, strength);
        surf2Dwrite(mc, dstImg, sizeof(mc) * x, y, cudaBoundaryModeZero);
        return;
    }

    mc.w = 0.299f * mc.z + 0.587f * mc.y + 0.114f * mc.x + 0.5f;
    surf2Dwrite(mc, dstImg, sizeof(mc) * x, y, cudaBoundaryModeZero);
}

void cuRunKernelAnime4K09(const unsigned char* inputData, unsigned char* outputData, ACCudaParamAnime4K09 * param)
{
    cudaError_t err = cudaSuccess;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

    cudaArray_t cuArray0;
    err = cudaMallocArray(&cuArray0, &channelDesc, param->orgW, param->orgH);
    CheckCudaErr(err);

    cudaArray_t cuArray1;
    err = cudaMallocArray(&cuArray1, &channelDesc, param->W, param->H, cudaArraySurfaceLoadStore);
    CheckCudaErr(err);

    cudaArray_t cuArray2;
    err = cudaMallocArray(&cuArray2, &channelDesc, param->W, param->H, cudaArraySurfaceLoadStore);
    CheckCudaErr(err);

    cudaArray_t cuArray3;
    err = cudaMallocArray(&cuArray3, &channelDesc, param->W, param->H, cudaArraySurfaceLoadStore);
    CheckCudaErr(err);

    struct cudaResourceDesc resDesc;
    struct cudaTextureDesc texDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    memset(&texDesc, 0, sizeof(texDesc));

    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    resDesc.resType = cudaResourceTypeArray;

    resDesc.res.array.array = cuArray0;
    cudaTextureObject_t tex = 0;
    err = cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
    CheckCudaErr(err);

    resDesc.res.array.array = cuArray1;
    cudaSurfaceObject_t surf1 = 0;
    err = cudaCreateSurfaceObject(&surf1, &resDesc);
    CheckCudaErr(err);

    resDesc.res.array.array = cuArray2;
    cudaSurfaceObject_t surf2 = 0;
    err = cudaCreateSurfaceObject(&surf2, &resDesc);
    CheckCudaErr(err);

    resDesc.res.array.array = cuArray3;
    cudaSurfaceObject_t surf3 = 0;
    err = cudaCreateSurfaceObject(&surf3, &resDesc);
    CheckCudaErr(err);

    err = cudaMemcpy2DToArray(cuArray0, 0, 0, inputData,
        sizeof(uchar4) * param->orgW, sizeof(uchar4) * param->orgW, param->orgH,
        cudaMemcpyHostToDevice);
    CheckCudaErr(err);

    dim3 dimBlock(16, 16);
    dim3 dimGrid(
        (param->W + dimBlock.x - 1) / dimBlock.x,
        (param->H + dimBlock.y - 1) / dimBlock.y
    );

    {
        int i;
        getGray <<<dimGrid, dimBlock>>> (tex, surf1, param->W, param->H);
        for (i = 0; i < param->passes && i < param->pushColorCount; i++)
        {
            pushColor <<<dimGrid, dimBlock>>> (surf1, surf2, param->W, param->H, param->strengthColor);
            getGradient <<<dimGrid, dimBlock>>> (surf2, surf3, param->W, param->H);
            pushGradient <<<dimGrid, dimBlock>>> (surf3, surf1, param->W, param->H, param->strengthGradient);
        }
        while (i++ < param->passes)
        {
            getGradient <<<dimGrid, dimBlock>>> (surf1, surf2, param->W, param->H);
            pushGradient <<<dimGrid, dimBlock>>> (surf2, surf1, param->W, param->H, param->strengthGradient);
        }
    }

    err = cudaMemcpy2DFromArray(outputData, sizeof(uchar4) * param->W, cuArray1, 0, 0,
        sizeof(uchar4) * param->W, param->H,
        cudaMemcpyDeviceToHost);
    CheckCudaErr(err);

    cudaDestroyTextureObject(tex);
    cudaDestroySurfaceObject(surf1);
    cudaDestroySurfaceObject(surf2);
    cudaDestroySurfaceObject(surf3);

    cudaFreeArray(cuArray0);
    cudaFreeArray(cuArray1);
    cudaFreeArray(cuArray2);
    cudaFreeArray(cuArray3);
}
