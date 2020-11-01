#pragma once

#ifdef ENABLE_CUDA

typedef struct
{
    size_t orgW, orgH;
    size_t W, H;
    int passes, pushColorCount;
    float strengthColor, strengthGradient;
}ACCudaParamAnime4K09;

typedef struct
{
    size_t orgW, orgH;
    int HDNLevel;
}ACCudaParamACNet;

void runKernelAnime4K09(const unsigned char* inputData, unsigned char* outputData, ACCudaParamAnime4K09* param);
void runKernelACNet(const unsigned char* inputData, unsigned char* outputData, ACCudaParamACNet* param);
void initCuda(const unsigned int id);
void releaseCuda();

#endif
