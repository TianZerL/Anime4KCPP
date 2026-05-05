#ifndef AC_CORE_OPENCL_COMMON_KERNEL_CL
#define AC_CORE_OPENCL_COMMON_KERNEL_CL

#ifndef WEIGHTS_PASS_SPACE
#   define WEIGHTS_PASS_SPACE global
#endif

#ifndef WEIGHTS_STORAGE_SPACE
#   define WEIGHTS_STORAGE_SPACE WEIGHTS_PASS_SPACE
#endif

#define Identity(v) (v)
#define ReLU(v) (fmax(v, 0.0f))
#define LReLU(v, n) (fmax(v, v * n))
#define PReLU(v, n) (fmax(v, 0.0f) + n * fmin(v, 0.0f))

constant sampler_t n_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

inline void copy_to_local(
    local float* const restrict lptr,
    WEIGHTS_PASS_SPACE const float* const restrict pptr,
    const int size)
{
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    const int local_size_x = get_local_size(0);
    const int local_size_y = get_local_size(1);

    const int threads = local_size_x * local_size_y;
    const int tid = local_y * local_size_x + local_x;

    if (threads < size)
    {
        const int line = size / threads;
        const int remain = size % threads;
        for (int i = 0; i < line; ++i)
        {
            const int idx = tid + i * threads;
            lptr[idx] = pptr[idx];
        }
        if (remain > 0 && tid < remain)
        {
            const int idx = tid + line * threads;
            lptr[idx] = pptr[idx];
        }
    }
    else if (tid < size) lptr[tid] = pptr[tid];
}

inline void conv1x1_from_array(
    const float* const in, float* const out,
    const int cin, const int cout,
    WEIGHTS_STORAGE_SPACE const float* const restrict kernels,
    WEIGHTS_STORAGE_SPACE const float* const restrict biases,
    const int x, const int y)
{
    const int count = cin / 8;

    for(int n = 0; n < cout; n++)
    {
        WEIGHTS_STORAGE_SPACE const float* const restrict kptr = kernels + n * cin;
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

inline void conv1x1_cin8_from_vector(
    const float8 r,  float* const out, const int cout,
    WEIGHTS_STORAGE_SPACE const float* const restrict kernels,
    WEIGHTS_STORAGE_SPACE const float* const restrict biases,
    const int x, const int y)
{
    for(int n = 0; n < cout; n++)
    {
        float8 k = vload8(n, kernels);
        float8 s = r * k;
        out[n] = dot(s.lo + s.hi, (float4)(1.0f)) + biases[n];
    }
}

inline void conv3x3_cin8_chunk(
    read_only image2d_array_t src, float* const out,
    const int chunk, const int cin, const int cout,
    WEIGHTS_STORAGE_SPACE const float* const restrict kernels,
    const int x, const int y)
{
    const int count = cin / 8;
    const int layer = chunk * 2;

#if defined (ARCH_AMD_GCN) || defined (ARCH_INTEL) || defined (ARCH_ADRENO)
    float8 r0 = (float8)(read_imagef(src, n_sampler, (int4)(x-1, y-1, layer + 0, 0)), read_imagef(src, n_sampler, (int4)(x-1, y-1, layer + 1, 0)));
    float8 r1 = (float8)(read_imagef(src, n_sampler, (int4)(x  , y-1, layer + 0, 0)), read_imagef(src, n_sampler, (int4)(x  , y-1, layer + 1, 0)));
    float8 r2 = (float8)(read_imagef(src, n_sampler, (int4)(x+1, y-1, layer + 0, 0)), read_imagef(src, n_sampler, (int4)(x+1, y-1, layer + 1, 0)));
    float8 r3 = (float8)(read_imagef(src, n_sampler, (int4)(x-1, y  , layer + 0, 0)), read_imagef(src, n_sampler, (int4)(x-1, y  , layer + 1, 0)));
    float8 r4 = (float8)(read_imagef(src, n_sampler, (int4)(x  , y  , layer + 0, 0)), read_imagef(src, n_sampler, (int4)(x  , y  , layer + 1, 0)));
    float8 r5 = (float8)(read_imagef(src, n_sampler, (int4)(x+1, y  , layer + 0, 0)), read_imagef(src, n_sampler, (int4)(x+1, y  , layer + 1, 0)));
    float8 r6 = (float8)(read_imagef(src, n_sampler, (int4)(x-1, y+1, layer + 0, 0)), read_imagef(src, n_sampler, (int4)(x-1, y+1, layer + 1, 0)));
    float8 r7 = (float8)(read_imagef(src, n_sampler, (int4)(x  , y+1, layer + 0, 0)), read_imagef(src, n_sampler, (int4)(x  , y+1, layer + 1, 0)));
    float8 r8 = (float8)(read_imagef(src, n_sampler, (int4)(x+1, y+1, layer + 0, 0)), read_imagef(src, n_sampler, (int4)(x+1, y+1, layer + 1, 0)));

    for(int n = 0; n < cout; n++)
    {
        WEIGHTS_STORAGE_SPACE const float* const restrict kptr = kernels + n * cin * 9;

        float8 k0 = vload8(count * 0 + chunk, kptr);
        float8 k1 = vload8(count * 1 + chunk, kptr);
        float8 k2 = vload8(count * 2 + chunk, kptr);
        float8 k3 = vload8(count * 3 + chunk, kptr);
        float8 k4 = vload8(count * 4 + chunk, kptr);
        float8 k5 = vload8(count * 5 + chunk, kptr);
        float8 k6 = vload8(count * 6 + chunk, kptr);
        float8 k7 = vload8(count * 7 + chunk, kptr);
        float8 k8 = vload8(count * 8 + chunk, kptr);

#   if defined (ARCH_AMD_GCN)
        float8 s0 = (float8)(0.0f);
        float8 s1 = (float8)(0.0f);
        s0 = mad(r0, k0, s0);
        s1 = mad(r1, k1, s1);
        s0 = mad(r2, k2, s0);
        s1 = mad(r3, k3, s1);
        s0 = mad(r4, k4, s0);
        s1 = mad(r5, k5, s1);
        s0 = mad(r6, k6, s0);
        s1 = mad(r7, k7, s1) + r8 * k8;

        out[n] += dot(s0.lo + s0.hi + s1.lo + s1.hi, (float4)(1.0f));
#   else
        float8 s0 = r0 * k0 +
                    r1 * k1 +
                    r2 * k2 +
                    r3 * k3 +
                    r4 * k4 +
                    r5 * k5 +
                    r6 * k6 +
                    r7 * k7 +
                    r8 * k8 ;

        out[n] += dot(s0.lo + s0.hi, (float4)(1.0f));
#   endif
    }
#else
    for(int n = 0; n < cout; n++)
    {
        float8 s = (float8)(0.0f);
        {
            for(int ypos = -1; ypos <= 1; ypos++)
            {
                for(int xpos = -1; xpos <= 1; xpos++)
                {
                    int pos = (ypos + 1) * 3 + (xpos + 1);
                    float8 r = (float8)(read_imagef(src, n_sampler, (int4)(x + xpos, y + ypos, layer + 0, 0)), read_imagef(src, n_sampler, (int4)(x + xpos, y + ypos, layer + 1, 0)));
                    float8 k = vload8(count * (pos + n * 9) + chunk, kernels);
                    s += r * k;
                }
            }
        }
        out[n] += dot(s.lo + s.hi, (float4)(1.0f));
    }
#endif
}

inline void conv3x3(
    read_only image2d_array_t src, float* const out,
    const int cin, const int cout,
    WEIGHTS_STORAGE_SPACE const float* const restrict kernels,
    WEIGHTS_STORAGE_SPACE const float* const restrict biases,
    const int x, const int y)
{
    const int count = cin / 8;

#if defined(ARCH_NVIDIA)
    for(int n = 0; n < cout; n++)
    {
        float8 s = (float8)(0.0f);
        {
            for(int ypos = -1; ypos <= 1; ypos++)
            {
                for(int xpos = -1; xpos <= 1; xpos++)
                {
                    int pos = (ypos + 1) * 3 + (xpos + 1);
                    for(int idx = 0; idx < count; idx++)
                    {
                        float8 r = (float8)(read_imagef(src, n_sampler, (int4)(x + xpos, y + ypos, idx * 2 + 0, 0)), read_imagef(src, n_sampler, (int4)(x + xpos, y + ypos, idx * 2 + 1, 0)));
                        float8 k = vload8(count * (pos + n * 9) + idx, kernels);
                        s += r * k;
                    }
                }
            }
        }
        out[n] = dot(s.lo + s.hi, (float4)(1.0f)) + biases[n];
    }
#else
    for(int n = 0; n < cout; n++) out[n] = biases[n];
    for(int idx = 0; idx < count; idx++) conv3x3_cin8_chunk(src, out, idx, cin, cout, kernels, x, y);
#endif
}

inline void conv3x3_cin1(
    read_only image2d_t src,
    float* const out, const int cout,
    WEIGHTS_STORAGE_SPACE const float* const restrict kernels,
    WEIGHTS_STORAGE_SPACE const float* const restrict biases,
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
    WEIGHTS_STORAGE_SPACE const float* const restrict kernels,
    WEIGHTS_STORAGE_SPACE const float* const restrict biases,
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
