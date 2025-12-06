#ifndef AC_CORE_OPENCL_COMMON_KERNEL_CL
#define AC_CORE_OPENCL_COMMON_KERNEL_CL

#define Identity(v) (v)
#define ReLU(v) (fmax(v, 0.0f))
#define LReLU(v, n) (fmax(v, v * n))

constant sampler_t n_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

inline void conv3x3_cin1(
    read_only image2d_t src,
    float* const out, const int cout,
    global const float* const restrict kernels, const int koffset,
    global const float* const restrict biases, const int boffset,
    const int x, const int y)
{
    global const float* const restrict kptr = kernels + koffset;
    global const float* const restrict bptr = biases + boffset;

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
        float8 k0 = vload8(0, kptr + n * 9 + 0);
        float k8 = kptr[n * 9 + 8];
        out[n] = dot(r0.lo, k0.lo) + dot(r0.hi, k0.hi) + r8 * k8 + bptr[n];
    }
}

inline void conv3x3_cin8(
    read_only image2d_array_t src,
    float* const out, const int cout,
    global const float* const restrict kernels, const int koffset,
    global const float* const restrict biases, const int boffset,
    const int x, const int y)
{
    global const float* const restrict kptr = kernels + koffset;
    global const float* const restrict bptr = biases + boffset;

    float8 r0 = (float8)(read_imagef(src, n_sampler, (int4)(x-1, y-1, 0, 0)), read_imagef(src, n_sampler, (int4)(x-1, y-1, 1, 0)));
    float8 r1 = (float8)(read_imagef(src, n_sampler, (int4)(x  , y-1, 0, 0)), read_imagef(src, n_sampler, (int4)(x  , y-1, 1, 0)));
    float8 r2 = (float8)(read_imagef(src, n_sampler, (int4)(x+1, y-1, 0, 0)), read_imagef(src, n_sampler, (int4)(x+1, y-1, 1, 0)));
    float8 r3 = (float8)(read_imagef(src, n_sampler, (int4)(x-1, y  , 0, 0)), read_imagef(src, n_sampler, (int4)(x-1, y  , 1, 0)));
    float8 r4 = (float8)(read_imagef(src, n_sampler, (int4)(x  , y  , 0, 0)), read_imagef(src, n_sampler, (int4)(x  , y  , 1, 0)));
    float8 r5 = (float8)(read_imagef(src, n_sampler, (int4)(x+1, y  , 0, 0)), read_imagef(src, n_sampler, (int4)(x+1, y  , 1, 0)));
    float8 r6 = (float8)(read_imagef(src, n_sampler, (int4)(x-1, y+1, 0, 0)), read_imagef(src, n_sampler, (int4)(x-1, y+1, 1, 0)));
    float8 r7 = (float8)(read_imagef(src, n_sampler, (int4)(x  , y+1, 0, 0)), read_imagef(src, n_sampler, (int4)(x  , y+1, 1, 0)));
    float8 r8 = (float8)(read_imagef(src, n_sampler, (int4)(x+1, y+1, 0, 0)), read_imagef(src, n_sampler, (int4)(x+1, y+1, 1, 0)));

    for(int n = 0; n < cout; n++)
    {
        global const float* const restrict k = kptr + n * 8 * 9;

        float8 k0 = vload8(0, k);
        float8 k1 = vload8(1, k);
        float8 k2 = vload8(2, k);
        float8 k3 = vload8(3, k);
        float8 k4 = vload8(4, k);
        float8 k5 = vload8(5, k);
        float8 k6 = vload8(6, k);
        float8 k7 = vload8(7, k);
        float8 k8 = vload8(8, k);

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

        out[n] = dot(s0.lo + s0.hi + s1.lo + s1.hi, (float4)(1.0f)) + bptr[n];
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

        out[n] = dot(s0.lo + s0.hi, (float4)(1.0f)) + bptr[n];
#   endif
    }
}

#endif
