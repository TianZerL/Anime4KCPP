/* #include "CommonKernel.cl" */

kernel void conv3x3_1to8_relu(
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

    write_imagef(dst, (int4)(x, y, 0, 0), ReLU(vload4(0, s)));
    write_imagef(dst, (int4)(x, y, 1, 0), ReLU(vload4(1, s)));
}

kernel void conv3x3_8to8_relu(
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

    write_imagef(dst, (int4)(x, y, 0, 0), ReLU(vload4(0, s)));
    write_imagef(dst, (int4)(x, y, 1, 0), ReLU(vload4(1, s)));
}

kernel void conv3x3_8to8_relu_deconv2x2_8to1(
    read_only image2d_array_t src,
    write_only image2d_t dst,
    WEIGHTS_PASS_SPACE const float* restrict kernels1, const int koffset1,
    WEIGHTS_PASS_SPACE const float* restrict biases1, const int boffset1,
    WEIGHTS_PASS_SPACE const float* restrict kernels2, const int koffset2)
{
#ifdef USE_WEIGHTS_OFFSET
    kernels1 += koffset1;
    biases1 += boffset1;
    kernels2 += koffset2;
#endif

#ifdef LOCAL_WEIGHTS_STORAGE_SPACE
    local float kptr1[8 * 3 * 3 * 8];
    local float bptr1[8];
    local float kptr2[1 * 2 * 2 * 8];
    copy_to_local(kptr1, kernels1, 8 * 3 * 3 * 8);
    copy_to_local(bptr1, biases1, 8);
    copy_to_local(kptr2, kernels2, 1 * 2 * 2 * 8);
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    WEIGHTS_STORAGE_SPACE const float* const restrict kptr1 = kernels1;
    WEIGHTS_STORAGE_SPACE const float* const restrict bptr1 = biases1;
    WEIGHTS_STORAGE_SPACE const float* const restrict kptr2 = kernels2;
#endif

    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(src) || y >= get_image_height(src)) return;

    float s[8];
    conv3x3(src, s, 8, 8, kptr1, bptr1, x, y);

    float8 v = ReLU(vload8(0, s));

    int2 dst_coord = (int2)(x, y) * 2;

    write_imagef(dst, dst_coord + (int2)(0, 0), (float4)(clamp(dot(v.lo, vload4(0, kptr2)) + dot(v.hi, vload4(1, kptr2)), 0.0f, 1.0f), 0.0f, 0.0f, 1.0f));
    write_imagef(dst, dst_coord + (int2)(1, 0), (float4)(clamp(dot(v.lo, vload4(2, kptr2)) + dot(v.hi, vload4(3, kptr2)), 0.0f, 1.0f), 0.0f, 0.0f, 1.0f));
    write_imagef(dst, dst_coord + (int2)(0, 1), (float4)(clamp(dot(v.lo, vload4(4, kptr2)) + dot(v.hi, vload4(5, kptr2)), 0.0f, 1.0f), 0.0f, 0.0f, 1.0f));
    write_imagef(dst, dst_coord + (int2)(1, 1), (float4)(clamp(dot(v.lo, vload4(6, kptr2)) + dot(v.hi, vload4(7, kptr2)), 0.0f, 1.0f), 0.0f, 0.0f, 1.0f));
}
