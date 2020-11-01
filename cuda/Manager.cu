#include"CudaHelper.cuh"
#include"CudaInterface.hpp"

void initCuda(const unsigned int id)
{
    cudaSetDevice(id);
}

void releaseCuda()
{
    cudaError_t err = cudaDeviceReset();
    CheckCudaErr(err);
}
