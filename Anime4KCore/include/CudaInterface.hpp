#pragma once

#ifdef ENABLE_CUDA

#include <string>

namespace Anime4KCPP
{
    namespace Cuda
    {
        typedef struct
        {
            int orgW, orgH;
            int W, H;
            size_t stride;
            int passes, pushColorCount;
            float strengthColor, strengthGradient;
        }ACCudaParamAnime4K09;

        typedef struct
        {
            int orgW, orgH;
            size_t stride;
            int HDNLevel;
        }ACCudaParamACNet;

        void cuRunKernelAnime4K09B(const unsigned char* inputData, unsigned char* outputData, ACCudaParamAnime4K09* param);
        void cuRunKernelAnime4K09W(const unsigned short int* inputData, unsigned short int* outputData, ACCudaParamAnime4K09* param);
        void cuRunKernelAnime4K09F(const float* inputData, float* outputData, ACCudaParamAnime4K09* param);

        void cuRunKernelACNetB(const unsigned char* inputData, unsigned char* outputData, ACCudaParamACNet* param);
        void cuRunKernelACNetW(const unsigned short int* inputData, unsigned short int* outputData, ACCudaParamACNet* param);
        void cuRunKernelACNetF(const float* inputData, float* outputData, ACCudaParamACNet* param);

        void cuSetDeviceID(const int id);
        int cuGetDeviceID() noexcept;
        void cuReleaseCuda() noexcept;

        int cuGetDeviceCount() noexcept;
        std::string cuGetDeviceInfo(const unsigned int id);
        std::string cuGetCudaInfo();
        bool cuCheckDeviceSupport(const unsigned int id) noexcept;
    }
}

#endif
