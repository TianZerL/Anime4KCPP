#ifndef ANIME4KCPP_CUDA_CUDA_HELPER_CUH
#define ANIME4KCPP_CUDA_CUDA_HELPER_CUH

#include <type_traits>
#include <cstdint>

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <surface_functions.h>
#include <texture_fetch_functions.h>

#include "ACException.hpp"

#define CheckCudaErr(err)   \
    if (err != cudaSuccess) \
    throw Anime4KCPP::ACException<Anime4KCPP::ExceptionType::GPU, true>(cudaGetErrorString(err), std::string(__FILE__), __LINE__)

using uchar = std::uint8_t;
using ushort = std::uint16_t;

extern int currCudaDeviceID;

template <typename T>
struct PixelValue;

template <>
struct PixelValue<uchar>
{
    __device__ static constexpr uchar max()
    {
        return 255;
    }

    __device__ static constexpr uchar min()
    {
        return 0;
    }
};

template <>
struct PixelValue<ushort>
{
    __device__ static constexpr ushort max()
    {
        return 65535;
    }

    __device__ static constexpr ushort min()
    {
        return 0;
    }
};

template <>
struct PixelValue<float>
{
    __device__ static constexpr float max()
    {
        return 1.0f;
    }

    __device__ static constexpr float min()
    {
        return 0.0f;
    }
};

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
inline __device__ typename Vec4<T>::type makeVec4(T x, T y, T z, T w);

template <>
inline __device__ typename Vec4<uchar>::type makeVec4(uchar x, uchar y, uchar z, uchar w)
{
    return make_uchar4(x, y, z, w);
}

template <>
inline __device__ typename Vec4<ushort>::type makeVec4(ushort x, ushort y, ushort z, ushort w)
{
    return make_ushort4(x, y, z, w);
}

template <>
inline __device__ typename Vec4<float>::type makeVec4(float x, float y, float z, float w)
{
    return make_float4(x, y, z, w);
}

inline __device__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}

#endif // !ANIME4KCPP_CUDA_CUDA_HELPER_CUH
