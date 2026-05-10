#include "CommonKernel.cl"

kernel void conv3x3_1to8_prelu(
    read_only image2d_t src,
    write_only image2d_array_t dst,
    WEIGHTS_PASS_SPACE const float* restrict kernels,
    const int koffset,
    WEIGHTS_PASS_SPACE const float* restrict biases,
    const int boffset,
    WEIGHTS_PASS_SPACE const float* restrict alphas,
    const int aoffset)
{
#ifdef USE_WEIGHTS_OFFSET
    kernels += koffset;
    biases += boffset;
    alphas += aoffset;
#endif

#ifdef LOCAL_WEIGHTS_STORAGE_SPACE
    local float kptr[8 * 3 * 3 * 1];
    local float bptr[8];
    copy_to_local(kptr, kernels, 8 * 3 * 3 * 1);
    copy_to_local(bptr, biases, 8);
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    WEIGHTS_STORAGE_SPACE const float* const restrict kptr = kernels;
    WEIGHTS_STORAGE_SPACE const float* const restrict bptr = biases;
#endif

    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(src) || y >= get_image_height(src)) return;

    float s[8];
    conv3x3_cin1(src, s, 8, kptr, bptr, x, y);

    float8 v = vload8(0, s);
    v = PReLU(v, vload8(0, alphas));

    write_imagef(dst, (int4)(x, y, 0, 0), v.lo);
    write_imagef(dst, (int4)(x, y, 1, 0), v.hi);
}

kernel void conv3x3_8to8_prelu(
    read_only image2d_array_t src,
    write_only image2d_array_t dst,
    WEIGHTS_PASS_SPACE const float* restrict kernels,
    const int koffset,
    WEIGHTS_PASS_SPACE const float* restrict biases,
    const int boffset,
    WEIGHTS_PASS_SPACE const float* restrict alphas,
    const int aoffset)
{
#ifdef USE_WEIGHTS_OFFSET
    kernels += koffset;
    biases += boffset;
    alphas += aoffset;
#endif

#ifdef LOCAL_WEIGHTS_STORAGE_SPACE
    local float kptr[8 * 3 * 3 * 8];
    local float bptr[8];
    copy_to_local(kptr, kernels, 8 * 3 * 3 * 8);
    copy_to_local(bptr, biases, 8);
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    WEIGHTS_STORAGE_SPACE const float* const restrict kptr = kernels;
    WEIGHTS_STORAGE_SPACE const float* const restrict bptr = biases;
#endif

    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(src) || y >= get_image_height(src)) return;

    float s[8];
    conv3x3(src, s, 8, 8, kptr, bptr, x, y);

    float8 v = vload8(0, s);
    v = PReLU(v, vload8(0, alphas));

    write_imagef(dst, (int4)(x, y, 0, 0), v.lo);
    write_imagef(dst, (int4)(x, y, 1, 0), v.hi);
}

kernel void conv3x3_8to4_identity_pixelshuffle_4to1_add(
    read_only image2d_array_t src,
    write_only image2d_t dst,
    WEIGHTS_PASS_SPACE const float* restrict kernels, const int koffset,
    WEIGHTS_PASS_SPACE const float* restrict biases, const int boffset,
    read_only image2d_t id)
{
#ifdef USE_WEIGHTS_OFFSET
    kernels += koffset;
    biases += boffset;
#endif

#ifdef LOCAL_WEIGHTS_STORAGE_SPACE
    local float kptr[4 * 3 * 3 * 8];
    local float bptr[4];
    copy_to_local(kptr, kernels, 4 * 3 * 3 * 8);
    copy_to_local(bptr, biases, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    WEIGHTS_STORAGE_SPACE const float* const restrict kptr = kernels;
    WEIGHTS_STORAGE_SPACE const float* const restrict bptr = biases;
#endif

    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(src) || y >= get_image_height(src)) return;

    float s[4];
    conv3x3(src, s, 8, 4, kptr, bptr, x, y);

    float4 pixel = clamp(Identity(vload4(0, s)) + read_imagef(id, n_sampler, (int2)(x, y)).x, 0.0f, 1.0f);
    int2 dst_coord = (int2)(x, y) * 2;
    write_imagef(dst, dst_coord + (int2)(0, 0), (float4)(pixel.s0, 0.0f, 0.0f, 1.0f));
    write_imagef(dst, dst_coord + (int2)(1, 0), (float4)(pixel.s1, 0.0f, 0.0f, 1.0f));
    write_imagef(dst, dst_coord + (int2)(0, 1), (float4)(pixel.s2, 0.0f, 0.0f, 1.0f));
    write_imagef(dst, dst_coord + (int2)(1, 1), (float4)(pixel.s3, 0.0f, 0.0f, 1.0f));
}
