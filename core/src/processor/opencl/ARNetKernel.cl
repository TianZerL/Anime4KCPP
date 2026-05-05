#include "CommonKernel.cl"

kernel void conv3x3_1to8_identity(
    read_only image2d_t src,
    write_only image2d_array_t dst,
    WEIGHTS_PASS_SPACE const float* restrict kernels,
    const int koffset,
    WEIGHTS_PASS_SPACE const float* restrict biases,
    const int boffset)
{
#ifdef USE_WEIGHTS_OFFSET
    kernels += koffset;
    biases += boffset;
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

    write_imagef(dst, (int4)(x, y, 0, 0), Identity(vload4(0, s)));
    write_imagef(dst, (int4)(x, y, 1, 0), Identity(vload4(1, s)));
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
    local float aptr[8];
    copy_to_local(kptr, kernels, 8 * 3 * 3 * 8);
    copy_to_local(bptr, biases, 8);
    copy_to_local(aptr, alphas, 8);
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    WEIGHTS_STORAGE_SPACE const float* const restrict kptr = kernels;
    WEIGHTS_STORAGE_SPACE const float* const restrict bptr = biases;
    WEIGHTS_STORAGE_SPACE const float* const restrict aptr = alphas;
#endif

    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(src) || y >= get_image_height(src)) return;

    float s[8];
    conv3x3(src, s, 8, 8, kptr, bptr, x, y);

    float8 v = vload8(0, s);
    v = PReLU(v, vload8(0, aptr));

    write_imagef(dst, (int4)(x, y, 0, 0), v.lo);
    write_imagef(dst, (int4)(x, y, 1, 0), v.hi);
}

kernel void conv3x3_8to8_identity_residual(
    read_only image2d_array_t src,
    write_only image2d_array_t dst,
    WEIGHTS_PASS_SPACE const float* restrict kernels,
    const int koffset,
    WEIGHTS_PASS_SPACE const float* restrict biases,
    const int boffset,
    read_only image2d_array_t id,
    const float scale)
{
#ifdef USE_WEIGHTS_OFFSET
    kernels += koffset;
    biases += boffset;
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

    write_imagef(dst, (int4)(x, y, 0, 0), Identity(vload4(0, s)) * scale + read_imagef(id, n_sampler, (int4)(x, y, 0, 0)));
    write_imagef(dst, (int4)(x, y, 1, 0), Identity(vload4(1, s)) * scale + read_imagef(id, n_sampler, (int4)(x, y, 1, 0)));
}

kernel void conv3x3_8to8_identity_residual_conv1x1_8to8_prelu_add(
    read_only image2d_array_t src,
    write_only image2d_array_t dst,
    WEIGHTS_PASS_SPACE const float* restrict kernels1, const int koffset1,
    WEIGHTS_PASS_SPACE const float* restrict biases1, const int boffset1,
    read_only image2d_array_t id, const float scale,
    WEIGHTS_PASS_SPACE const float* restrict kernels2, const int koffset2,
    WEIGHTS_PASS_SPACE const float* restrict biases2, const int boffset2,
    WEIGHTS_PASS_SPACE const float* restrict alphas2, const int aoffset2,
    read_only image2d_array_t feat)
{
#ifdef USE_WEIGHTS_OFFSET
    kernels1 += koffset1;
    biases1 += boffset1;

    kernels2 += koffset2;
    biases2 += boffset2;
    alphas2 += aoffset2;
#endif

#ifdef LOCAL_WEIGHTS_STORAGE_SPACE
    local float kptr1[8 * 3 * 3 * 8];
    local float bptr1[8];
    local float kptr2[8 * 1 * 1 * 8];
    local float bptr2[8];
    local float aptr2[8];
    copy_to_local(kptr1, kernels1, 8 * 3 * 3 * 8);
    copy_to_local(bptr1, biases1, 8);
    copy_to_local(kptr2, kernels2, 8 * 1 * 1 * 8);
    copy_to_local(bptr2, biases2, 8);
    copy_to_local(aptr2, alphas2, 8);
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    WEIGHTS_STORAGE_SPACE const float* const restrict kptr1 = kernels1;
    WEIGHTS_STORAGE_SPACE const float* const restrict bptr1 = biases1;
    WEIGHTS_STORAGE_SPACE const float* const restrict kptr2 = kernels2;
    WEIGHTS_STORAGE_SPACE const float* const restrict bptr2 = biases2;
    WEIGHTS_STORAGE_SPACE const float* const restrict aptr2 = alphas2;
#endif

    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(src) || y >= get_image_height(src)) return;

    float s[8];
    conv3x3(src, s, 8, 8, kptr1, bptr1, x, y);

    float8 v = Identity(vload8(0, s)) * scale + (float8)(read_imagef(id, n_sampler, (int4)(x, y, 0, 0)), read_imagef(id, n_sampler, (int4)(x, y, 1, 0)));

    conv1x1_cin8_from_vector(v, s, 8, kptr2, bptr2, x, y);

    v = vload8(0, s);
    v = PReLU(v, vload8(0, aptr2)) + (float8)(read_imagef(feat, n_sampler, (int4)(x, y, 0, 0)), read_imagef(feat, n_sampler, (int4)(x, y, 1, 0)));

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
