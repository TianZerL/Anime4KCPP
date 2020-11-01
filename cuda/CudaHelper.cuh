#ifndef __CUDA_HELPER__
#define __CUDA_HELPER__

#include "ACException.hpp"
#include "device_launch_parameters.h"

#ifndef  __CUDACC__
#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include <surface_functions.h>
#include <texture_fetch_functions.h>

#define CheckCudaErr(err) \
if (cudaSuccess != err) \
    throw Anime4KCPP::ACException<Anime4KCPP::ExceptionType::GPU, true>(cudaGetErrorString(err), std::string(__FILE__), __LINE__)

#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaCheckError(const char* file, const int line)
{
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
            file, line, cudaGetErrorString(err));
        exit(-1);
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
            file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

#endif
