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
if (err != cudaSuccess) throw Anime4KCPP::ACException<Anime4KCPP::ExceptionType::GPU, true>(cudaGetErrorString(err), std::string(__FILE__), __LINE__)

#endif
