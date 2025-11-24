#ifndef AC_CORE_OPENCL_COMMON_KERNEL_CL
#define AC_CORE_OPENCL_COMMON_KERNEL_CL

#define Identity(v) (v)
#define ReLU(v) (fmax(v, 0.0f))
#define LReLU(v, n) (fmax(v, v * n))

constant sampler_t n_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

inline void conv3x3_cin1(
    read_only image2d_t src,
    float* const out, const int cout,
    constant float* const kernels,
    constant float* const biases,
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

inline void conv3x3_cin8(
    read_only image2d_array_t src,
    float* const out, const int cout,
    constant float* const kernels,
    constant float* const biases,
    const int x, const int y)
{
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
        constant float* k = kernels + n * 8 * 9;

        float8 k0 = vload8(0, k);
        float8 k1 = vload8(1, k);
        float8 k2 = vload8(2, k);
        float8 k3 = vload8(3, k);
        float8 k4 = vload8(4, k);
        float8 k5 = vload8(5, k);
        float8 k6 = vload8(6, k);
        float8 k7 = vload8(7, k);
        float8 k8 = vload8(8, k);

#   if defined (ARCH_MESA) || defined (ARCH_AMD_RDNA) || defined (ARCH_INTEL)
        float8 s0 = r0 * k0 +
                    r1 * k1 +
                    r2 * k2 +
                    r3 * k3 +
                    r4 * k4 +
                    r5 * k5 +
                    r6 * k6 +
                    r7 * k7 +
                    r8 * k8 ;

        out[n] = dot(s0.lo + s0.hi, (float4)(1.0f)) + biases[n];
#   elif defined(ARCH_AMD_GCN)
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

        out[n] = dot(s0.lo + s0.hi + s1.lo + s1.hi, (float4)(1.0f)) + biases[n];
#   else
        float s0 = dot(r0.lo, k0.lo) + dot(r0.hi, k0.hi);
        float s1 = dot(r1.lo, k1.lo) + dot(r1.hi, k1.hi);
        float s2 = dot(r2.lo, k2.lo) + dot(r2.hi, k2.hi);
        float s3 = dot(r3.lo, k3.lo) + dot(r3.hi, k3.hi);
        float s4 = dot(r4.lo, k4.lo) + dot(r4.hi, k4.hi);
        float s5 = dot(r5.lo, k5.lo) + dot(r5.hi, k5.hi);
        float s6 = dot(r6.lo, k6.lo) + dot(r6.hi, k6.hi);
        float s7 = dot(r7.lo, k7.lo) + dot(r7.hi, k7.hi);
        float s8 = dot(r8.lo, k8.lo) + dot(r8.hi, k8.hi);

        out[n] = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + biases[n];
#   endif
    }
}

#endif
