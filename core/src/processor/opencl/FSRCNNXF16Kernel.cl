#include "CommonKernel.cl"

kernel void conv5x5_1to16_identity(
    read_only image2d_t src,
    write_only image2d_array_t dst,
    WEIGHTS_SPACE const float* restrict kernels,
    const int koffset,
    WEIGHTS_SPACE const float* restrict biases,
    const int boffset)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(src) || y >= get_image_height(src)) return;

#ifdef USE_WEIGHTS_OFFSET
    kernels += koffset;
    biases += boffset;
#endif

    float s[16];
    conv5x5_cin1(src, s, 16, kernels, biases, x, y);

    write_imagef(dst, (int4)(x, y, 0, 0), Identity(vload4(0, s)));
    write_imagef(dst, (int4)(x, y, 1, 0), Identity(vload4(1, s)));
    write_imagef(dst, (int4)(x, y, 2, 0), Identity(vload4(2, s)));
    write_imagef(dst, (int4)(x, y, 3, 0), Identity(vload4(3, s)));
}

kernel void conv3x3_16to16_prelu(
    read_only image2d_array_t src,
    write_only image2d_array_t dst,
    WEIGHTS_SPACE const float* restrict kernels,
    const int koffset,
    WEIGHTS_SPACE const float* restrict biases,
    const int boffset,
    WEIGHTS_SPACE const float* restrict alphas,
    const int aoffset)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(src) || y >= get_image_height(src)) return;

#ifdef USE_WEIGHTS_OFFSET
    kernels += koffset;
    biases += boffset;
    alphas += aoffset;
#endif

    float s[16];
    conv3x3(src, s, 16, 16, kernels, biases, x, y);

    float16 v = vload16(0, s);
    v = PReLU(v, vload16(0, alphas));

    write_imagef(dst, (int4)(x, y, 0, 0), v.lo.lo);
    write_imagef(dst, (int4)(x, y, 1, 0), v.lo.hi);
    write_imagef(dst, (int4)(x, y, 2, 0), v.hi.lo);
    write_imagef(dst, (int4)(x, y, 3, 0), v.hi.hi);
}

kernel void conv3x3_16to16_prelu_conv1x1_16to16_add_prelu(
    read_only image2d_array_t src,
    write_only image2d_array_t dst,
    WEIGHTS_SPACE const float* restrict kernels1, const int koffset1,
    WEIGHTS_SPACE const float* restrict biases1, const int boffset1,
    WEIGHTS_SPACE const float* restrict alphas1, const int aoffset1,
    WEIGHTS_SPACE const float* restrict kernels2, const int koffset2,
    WEIGHTS_SPACE const float* restrict biases2, const int boffset2,
    WEIGHTS_SPACE const float* restrict alphas2, const int aoffset2,
    read_only image2d_array_t feat)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(src) || y >= get_image_height(src)) return;

#ifdef USE_WEIGHTS_OFFSET
    kernels1 += koffset1;
    biases1 += boffset1;
    alphas1 += aoffset1;

    kernels2 += koffset2;
    biases2 += boffset2;
    alphas2 += aoffset2;
#endif

    float buffer[16];
    conv3x3(src, buffer, 16, 16, kernels1, biases1, x, y);

    float16 v = vload16(0, buffer);
    v = PReLU(v, vload16(0, alphas1));
    vstore16(v, 0, buffer);

    float s[16];
    conv1x1_from_array(buffer, s, 16, 16, kernels2, biases2, x, y);

    v = vload16(0, s);
    v = v + (float16)(
        read_imagef(feat, n_sampler, (int4)(x, y, 0, 0)), read_imagef(feat, n_sampler, (int4)(x, y, 1, 0)),
        read_imagef(feat, n_sampler, (int4)(x, y, 2, 0)), read_imagef(feat, n_sampler, (int4)(x, y, 3, 0))
    );
    v = PReLU(v, vload16(0, alphas2));

    write_imagef(dst, (int4)(x, y, 0, 0), v.lo.lo);
    write_imagef(dst, (int4)(x, y, 1, 0), v.lo.hi);
    write_imagef(dst, (int4)(x, y, 2, 0), v.hi.lo);
    write_imagef(dst, (int4)(x, y, 3, 0), v.hi.hi);
}

kernel void conv3x3_16to4_identity_pixelshuffle_4to1(
    read_only image2d_array_t src,
    write_only image2d_t dst,
    WEIGHTS_SPACE const float* restrict kernels,
    const int koffset,
    WEIGHTS_SPACE const float* restrict biases,
    const int boffset)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(src) || y >= get_image_height(src)) return;

#ifdef USE_WEIGHTS_OFFSET
    kernels += koffset;
    biases += boffset;
#endif

    float s[4];
    conv3x3(src, s, 16, 4, kernels, biases, x, y);

    float4 pixel = clamp(Identity(vload4(0, s)), 0.0f, 1.0f);
    int2 dst_coord = (int2)(x, y) * 2;
    write_imagef(dst, dst_coord + (int2)(0, 0), (float4)(pixel.s0, 0.0f, 0.0f, 1.0f));
    write_imagef(dst, dst_coord + (int2)(1, 0), (float4)(pixel.s1, 0.0f, 0.0f, 1.0f));
    write_imagef(dst, dst_coord + (int2)(0, 1), (float4)(pixel.s2, 0.0f, 0.0f, 1.0f));
    write_imagef(dst, dst_coord + (int2)(1, 1), (float4)(pixel.s3, 0.0f, 0.0f, 1.0f));
}
