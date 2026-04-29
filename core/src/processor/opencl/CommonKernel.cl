#ifndef AC_CORE_OPENCL_COMMON_KERNEL_CL
#define AC_CORE_OPENCL_COMMON_KERNEL_CL

#ifdef PASS_WEIGHTS_BY_CONSTANT
#   define WEIGHTS_SPACE constant
#else
#   define WEIGHTS_SPACE global
#endif

#define Identity(v) (v)
#define ReLU(v) (fmax(v, 0.0f))
#define LReLU(v, n) (fmax(v, v * n))
#define PReLU(v, n) (fmax(v, 0.0f) + n * fmin(v, 0.0f))

constant sampler_t n_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

inline void conv1x1(
    read_only image2d_array_t src, float* const out,
    const int cin, const int cout,
    WEIGHTS_SPACE const float* const restrict kernels,
    WEIGHTS_SPACE const float* const restrict biases,
    const int x, const int y)
{
    const int count = cin / 8;

    for(int n = 0; n < cout; n++)
    {
        WEIGHTS_SPACE const float* const restrict kptr = kernels + n * cin;
        float8 s = (float8)(0.0f);
        for(int idx = 0; idx < count; idx++)
        {
            float8 r = (float8)(read_imagef(src, n_sampler, (int4)(x, y, idx * 2 + 0, 0)), read_imagef(src, n_sampler, (int4)(x, y, idx * 2 + 1, 0)));
            float8 k = vload8(idx, kptr);
            s += r * k;
        }
        out[n] = dot(s.lo + s.hi, (float4)(1.0f)) + biases[n];
    }
}

inline void conv1x1_from_array(
    const float* const in, float* const out,
    const int cin, const int cout,
    WEIGHTS_SPACE const float* const restrict kernels,
    WEIGHTS_SPACE const float* const restrict biases,
    const int x, const int y)
{
    const int count = cin / 8;

    for(int n = 0; n < cout; n++)
    {
        WEIGHTS_SPACE const float* const restrict kptr = kernels + n * cin;
        float8 s = (float8)(0.0f);
        for(int idx = 0; idx < count; idx++)
        {
            float8 r = vload8(idx, in);
            float8 k = vload8(idx, kptr);
            s += r * k;
        }
        out[n] = dot(s.lo + s.hi, (float4)(1.0f)) + biases[n];
    }
}

inline void conv1x1_cin8(
    read_only image2d_array_t src,
    float* const out, const int cout,
    WEIGHTS_SPACE const float* const restrict kernels,
    WEIGHTS_SPACE const float* const restrict biases,
    const int x, const int y)
{
    float8 r = (float8)(read_imagef(src, n_sampler, (int4)(x, y, 0, 0)), read_imagef(src, n_sampler, (int4)(x, y, 1, 0)));

    for(int n = 0; n < cout; n++)
    {
        float8 k = vload8(n, kernels);
        float8 s = r * k;
        out[n] = dot(s.lo + s.hi, (float4)(1.0f)) + biases[n];
    }
}

inline void conv1x1_cin8_from_vector(
    const float8 r, float* const out, const int cout,
    WEIGHTS_SPACE const float* const restrict kernels,
    WEIGHTS_SPACE const float* const restrict biases,
    const int x, const int y)
{
    for(int n = 0; n < cout; n++)
    {
        float8 k = vload8(n, kernels);
        float8 s = r * k;
        out[n] = dot(s.lo + s.hi, (float4)(1.0f)) + biases[n];
    }
}

inline void conv3x3(
    read_only image2d_array_t src, float* const out,
    const int cin, const int cout,
    WEIGHTS_SPACE const float* const restrict kernels,
    WEIGHTS_SPACE const float* const restrict biases,
    const int x, const int y)
{
    const int count = cin / 8;

    for(int n = 0; n < cout; n++)
    {
        WEIGHTS_SPACE const float* const restrict kptr = kernels + n * cin * 9;
        float8 s = (float8)(0.0f);
        for(int ypos = -1; ypos <= 1; ypos++)
        {
            for(int xpos = -1; xpos <= 1; xpos++)
            {
                for(int idx = 0; idx < count; idx++)
                {
                    float8 r = (float8)(read_imagef(src, n_sampler, (int4)(x + xpos, y + ypos, idx * 2 + 0, 0)), read_imagef(src, n_sampler, (int4)(x-1, y-1, idx * 2 + 1, 0)));
                    float8 k = vload8(count * ((ypos + 1) * 3 + xpos + 1) + idx, kptr);
                    s += r * k;
                }
            }
        }
        out[n] = dot(s.lo + s.hi, (float4)(1.0f)) + biases[n];
    }
}

inline void conv3x3_cin1(
    read_only image2d_t src,
    float* const out, const int cout,
    WEIGHTS_SPACE const float* const restrict kernels,
    WEIGHTS_SPACE const float* const restrict biases,
    const int x, const int y)
{
    float8 r0 = (float8)(
        read_imagef(src, n_sampler, (int2)(x-1, y-1)).x,
        read_imagef(src, n_sampler, (int2)(x  , y-1)).x,
        read_imagef(src, n_sampler, (int2)(x+1, y-1)).x,
        read_imagef(src, n_sampler, (int2)(x-1, y  )).x,
        read_imagef(src, n_sampler, (int2)(x  , y  )).x,
        read_imagef(src, n_sampler, (int2)(x+1, y  )).x,
        read_imagef(src, n_sampler, (int2)(x-1, y+1)).x,
        read_imagef(src, n_sampler, (int2)(x  , y+1)).x
    );
    float r8 = read_imagef(src, n_sampler, (int2)(x+1, y+1)).x;

    for(int n = 0; n < cout; n++)
    {
        float8 k0 = vload8(0, kernels + n * 9 + 0);
        float k8 = kernels[n * 9 + 8];
        out[n] = dot(r0.lo, k0.lo) + dot(r0.hi, k0.hi) + r8 * k8 + biases[n];
    }
}

inline void conv5x5_cin1(
    read_only image2d_t src,
    float* const out, const int cout,
    WEIGHTS_SPACE const float* const restrict kernels,
    WEIGHTS_SPACE const float* const restrict biases,
    const int x, const int y)
{
    float8 r0 = (float8)(
        read_imagef(src, n_sampler, (int2)(x-2, y-2)).x,
        read_imagef(src, n_sampler, (int2)(x-1, y-2)).x,
        read_imagef(src, n_sampler, (int2)(x  , y-2)).x,
        read_imagef(src, n_sampler, (int2)(x+1, y-2)).x,
        read_imagef(src, n_sampler, (int2)(x+2, y-2)).x,
        read_imagef(src, n_sampler, (int2)(x-2, y-1)).x,
        read_imagef(src, n_sampler, (int2)(x-1, y-1)).x,
        read_imagef(src, n_sampler, (int2)(x  , y-1)).x
    );
    float8 r8 = (float8)(
        read_imagef(src, n_sampler, (int2)(x+1, y-1)).x,
        read_imagef(src, n_sampler, (int2)(x+2, y-1)).x,
        read_imagef(src, n_sampler, (int2)(x-2, y  )).x,
        read_imagef(src, n_sampler, (int2)(x-1, y  )).x,
        read_imagef(src, n_sampler, (int2)(x  , y  )).x,
        read_imagef(src, n_sampler, (int2)(x+1, y  )).x,
        read_imagef(src, n_sampler, (int2)(x+2, y  )).x,
        read_imagef(src, n_sampler, (int2)(x-2, y+1)).x
    );
    float8 r16 = (float8)(
        read_imagef(src, n_sampler, (int2)(x-1, y+1)).x,
        read_imagef(src, n_sampler, (int2)(x  , y+1)).x,
        read_imagef(src, n_sampler, (int2)(x+1, y+1)).x,
        read_imagef(src, n_sampler, (int2)(x+2, y+1)).x,
        read_imagef(src, n_sampler, (int2)(x-2, y+2)).x,
        read_imagef(src, n_sampler, (int2)(x-1, y+2)).x,
        read_imagef(src, n_sampler, (int2)(x  , y+2)).x,
        read_imagef(src, n_sampler, (int2)(x+1, y+2)).x
    );
    float r24 = read_imagef(src, n_sampler, (int2)(x+2, y+2)).x;

    for(int n = 0; n < cout; n++)
    {
        float8 k0 = vload8(0, kernels + n * 25 + 0);
        float8 k8 = vload8(0, kernels + n * 25 + 8);
        float8 k16 = vload8(0, kernels + n * 25 + 16);
        float k24 = kernels[n * 25 + 24];

        float8 s = r0 * k0 + r8 * k8 + r16 * k16;
        out[n] = dot(s.lo + s.hi, (float4)(1.0f)) + r24 * k24 + biases[n];
    }
}

#endif
