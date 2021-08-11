#ifndef __CUDA_HELPER__
#define __CUDA_HELPER__

#include <type_traits>
#include <limits>

#include "ACException.hpp"
#include "device_launch_parameters.h"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <surface_functions.h>
#include <texture_fetch_functions.h>

#define CheckCudaErr(err)   \
    if (err != cudaSuccess) \
    throw Anime4KCPP::ACException<Anime4KCPP::ExceptionType::GPU, true>(cudaGetErrorString(err), std::string(__FILE__), __LINE__)

typedef unsigned char uchar;
typedef unsigned short ushort;

extern int currCudaDeviceID;

template <typename T>
struct PixelValue
{
    template <typename P = T, std::enable_if_t<std::is_integral<P>::value> * = nullptr>
    __device__ static constexpr P max()
    {
        return std::numeric_limits<P>::max();
    }

    template <typename P = T, std::enable_if_t<std::is_integral<P>::value> * = nullptr>
    __device__ static constexpr P min()
    {
        return std::numeric_limits<P>::min();
    }

    template <typename P = T, std::enable_if_t<std::is_floating_point<P>::value> * = nullptr>
    __device__ static constexpr P max()
    {
        return 1.0f;
    }

    template <typename P = T, std::enable_if_t<std::std::is_floating_point<P>::value> * = nullptr>
    __device__ static constexpr P min()
    {
        return 0.0f;
    }
}

template <typename T, int dim>
struct Vec;

template <>
struct Vec<uchar, 2>
{
    using type = uchar2;
};

template <>
struct Vec<uchar, 4>
{
    using type = uchar4;
};

template <>
struct Vec<ushort, 2>
{
    using type = ushort2;
};

template <>
struct Vec<ushort, 4>
{
    using type = ushort4;
};

template <>
struct Vec<float, 2>
{
    using type = float2;
};

template <>
struct Vec<float, 4>
{
    using type = float4;
};

template <typename T>
using Vec2 = Vec<T, 2>;

template <typename T>
using Vec4 = Vec<T, 4>;

template <typename T>
__inline__ __device__ typename Vec4<T>::type makeVec4(T x, T y, T z, T w);

template <>
__inline__ __device__ typename Vec4<uchar>::type makeVec4(uchar x, uchar y, uchar z, uchar w)
{
    return make_uchar4(x, y, z, w);
}

template <>
__inline__ __device__ typename Vec4<ushort>::type makeVec4(ushort x, ushort y, ushort z, ushort w)
{
    return make_ushort4(x, y, z, w);
}

template <>
__inline__ __device__ typename Vec4<float>::type makeVec4(float x, float y, float z, float w)
{
    return make_float4(x, y, z, w);
}

__inline__ __device__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}

#endif
