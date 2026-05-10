#include "CommonKernel.cl"

kernel void conv3x3_1to32_identity(
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
    local float kptr[32 * 3 * 3 * 1];
    local float bptr[32];
    copy_to_local(kptr, kernels, 32 * 3 * 3 * 1);
    copy_to_local(bptr, biases, 32);
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    WEIGHTS_STORAGE_SPACE const float* const restrict kptr = kernels;
    WEIGHTS_STORAGE_SPACE const float* const restrict bptr = biases;
#endif

    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(src) || y >= get_image_height(src)) return;

    float s[16];

    for(int n = 0; n < 2; n++)
    {
        conv3x3_cin1(src, s, 16, kptr + n * 16 * 9 * 1, bptr + n * 16, x, y);
        write_imagef(dst, (int4)(x, y, n * 4 + 0, 0), Identity(vload4(0, s)));
        write_imagef(dst, (int4)(x, y, n * 4 + 1, 0), Identity(vload4(1, s)));
        write_imagef(dst, (int4)(x, y, n * 4 + 2, 0), Identity(vload4(2, s)));
        write_imagef(dst, (int4)(x, y, n * 4 + 3, 0), Identity(vload4(3, s)));
    }
}

kernel void conv3x3_32to32_relu(
    read_only image2d_array_t src,
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
    local float kptr[32 * 3 * 3 * 32];
    local float bptr[32];
    copy_to_local(kptr, kernels, 32 * 3 * 3 * 32);
    copy_to_local(bptr, biases, 32);
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    WEIGHTS_STORAGE_SPACE const float* const restrict kptr = kernels;
    WEIGHTS_STORAGE_SPACE const float* const restrict bptr = biases;
#endif

    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(src) || y >= get_image_height(src)) return;

    float s[16];

    for(int n = 0; n < 2; n++)
    {
        conv3x3(src, s, 32, 16, kptr + n * 16 * 9 * 32, bptr + n * 16, x, y);
        write_imagef(dst, (int4)(x, y, n * 4 + 0, 0), ReLU(vload4(0, s)));
        write_imagef(dst, (int4)(x, y, n * 4 + 1, 0), ReLU(vload4(1, s)));
        write_imagef(dst, (int4)(x, y, n * 4 + 2, 0), ReLU(vload4(2, s)));
        write_imagef(dst, (int4)(x, y, n * 4 + 3, 0), ReLU(vload4(3, s)));
    }
}

kernel void conv3x3_32to32_identity_add(
    read_only image2d_array_t src,
    write_only image2d_array_t dst,
    WEIGHTS_PASS_SPACE const float* restrict kernels,
    const int koffset,
    WEIGHTS_PASS_SPACE const float* restrict biases,
    const int boffset,
    read_only image2d_array_t feat)
{
#ifdef USE_WEIGHTS_OFFSET
    kernels += koffset;
    biases += boffset;
#endif

#ifdef LOCAL_WEIGHTS_STORAGE_SPACE
    local float kptr[32 * 3 * 3 * 32];
    local float bptr[32];
    copy_to_local(kptr, kernels, 32 * 3 * 3 * 32);
    copy_to_local(bptr, biases, 32);
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    WEIGHTS_STORAGE_SPACE const float* const restrict kptr = kernels;
    WEIGHTS_STORAGE_SPACE const float* const restrict bptr = biases;
#endif

    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(src) || y >= get_image_height(src)) return;

    float s[16];

    for(int n = 0; n < 2; n++)
    {
        conv3x3(src, s, 32, 16, kptr + n * 16 * 9 * 32, bptr + n * 16, x, y);
        write_imagef(dst, (int4)(x, y, n * 4 + 0, 0), Identity(vload4(0, s)) + read_imagef(feat, n_sampler, (int4)(x, y, n * 4 + 0, 0)));
        write_imagef(dst, (int4)(x, y, n * 4 + 1, 0), Identity(vload4(1, s)) + read_imagef(feat, n_sampler, (int4)(x, y, n * 4 + 1, 0)));
        write_imagef(dst, (int4)(x, y, n * 4 + 2, 0), Identity(vload4(2, s)) + read_imagef(feat, n_sampler, (int4)(x, y, n * 4 + 2, 0)));
        write_imagef(dst, (int4)(x, y, n * 4 + 3, 0), Identity(vload4(3, s)) + read_imagef(feat, n_sampler, (int4)(x, y, n * 4 + 3, 0)));
    }
}

kernel void conv3x3_32to4_identity_pixelshuffle_4to1(
    read_only image2d_array_t src,
    write_only image2d_t dst,
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
    local float kptr[4 * 3 * 3 * 32];
    local float bptr[4];
    copy_to_local(kptr, kernels, 4 * 3 * 3 * 32);
    copy_to_local(bptr, biases, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    WEIGHTS_STORAGE_SPACE const float* const restrict kptr = kernels;
    WEIGHTS_STORAGE_SPACE const float* const restrict bptr = biases;
#endif

    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(src) || y >= get_image_height(src)) return;

    float s[4];
    conv3x3(src, s, 32, 4, kptr, bptr, x, y);

    float4 pixel = clamp(Identity(vload4(0, s)), 0.0f, 1.0f);
    int2 dst_coord = (int2)(x, y) * 2;
    write_imagef(dst, dst_coord + (int2)(0, 0), (float4)(pixel.s0, 0.0f, 0.0f, 1.0f));
    write_imagef(dst, dst_coord + (int2)(1, 0), (float4)(pixel.s1, 0.0f, 0.0f, 1.0f));
    write_imagef(dst, dst_coord + (int2)(0, 1), (float4)(pixel.s2, 0.0f, 0.0f, 1.0f));
    write_imagef(dst, dst_coord + (int2)(1, 1), (float4)(pixel.s3, 0.0f, 0.0f, 1.0f));
}
