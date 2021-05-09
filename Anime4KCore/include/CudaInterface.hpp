#pragma once

#ifdef ENABLE_CUDA

#include<string>

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
        }ACCudaParamACNet;

        void cuRunKernelAnime4K09B(const unsigned char* inputData, unsigned char* outputData, ACCudaParamAnime4K09* param);
        void cuRunKernelAnime4K09W(const unsigned short* inputData, unsigned short* outputData, ACCudaParamAnime4K09* param);
        void cuRunKernelAnime4K09F(const float* inputData, float* outputData, ACCudaParamAnime4K09* param);

        void cuRunKernelACNetHDN0B(const unsigned char* inputData, unsigned char* outputData, ACCudaParamACNet* param);
        void cuRunKernelACNetHDN1B(const unsigned char* inputData, unsigned char* outputData, ACCudaParamACNet* param);
        void cuRunKernelACNetHDN2B(const unsigned char* inputData, unsigned char* outputData, ACCudaParamACNet* param);
        void cuRunKernelACNetHDN3B(const unsigned char* inputData, unsigned char* outputData, ACCudaParamACNet* param);
        
        void cuRunKernelACNetHDN0W(const unsigned short* inputData, unsigned short* outputData, ACCudaParamACNet* param);
        void cuRunKernelACNetHDN1W(const unsigned short* inputData, unsigned short* outputData, ACCudaParamACNet* param);
        void cuRunKernelACNetHDN2W(const unsigned short* inputData, unsigned short* outputData, ACCudaParamACNet* param);
        void cuRunKernelACNetHDN3W(const unsigned short* inputData, unsigned short* outputData, ACCudaParamACNet* param);

        void cuRunKernelACNetHDN0F(const float* inputData, float* outputData, ACCudaParamACNet* param);
        void cuRunKernelACNetHDN1F(const float* inputData, float* outputData, ACCudaParamACNet* param);
        void cuRunKernelACNetHDN2F(const float* inputData, float* outputData, ACCudaParamACNet* param);
        void cuRunKernelACNetHDN3F(const float* inputData, float* outputData, ACCudaParamACNet* param);

        void cuSetDeviceID(const int id);
        int cuGetDeviceID() noexcept;
        void cuReleaseCuda() noexcept;

        int cuGetDeviceCount() noexcept;
        std::string cuGetDeviceInfo(const int id);
        std::string cuGetCudaInfo();
        bool cuCheckDeviceSupport(const int id) noexcept;
    }
}

#endif
