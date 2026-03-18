#include "CommonKernel.cl"

kernel void conv5x5_1to8_identity(
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

    float s[8];
    conv5x5_cin1(src, s, 8, kernels, biases, x, y);

    write_imagef(dst, (int4)(x, y, 0, 0), Identity(vload4(0, s)));
    write_imagef(dst, (int4)(x, y, 1, 0), Identity(vload4(1, s)));
}

kernel void conv3x3_8to8_prelu(
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

    float s[8];
    conv3x3_cin8(src, s, 8, kernels, biases, x, y);

    write_imagef(dst, (int4)(x, y, 0, 0), LReLU(vload4(0, s), vload4(0, alphas)));
    write_imagef(dst, (int4)(x, y, 1, 0), LReLU(vload4(1, s), vload4(1, alphas)));
}

kernel void conv3x3_8to8_prelu_conv1x1_8to8_prelu_add(
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

    float s[8];
    conv3x3_cin8(src, s, 8, kernels1, biases1, x, y);

    conv1x1_cin8_from_vector(LReLU(vload8(0, s), vload8(0, alphas1)), s, 8, kernels2, biases2, x, y);

    write_imagef(dst, (int4)(x, y, 0, 0), LReLU(vload4(0, s), vload4(0, alphas2)) + read_imagef(feat, n_sampler, (int4)(x, y, 0, 0)));
    write_imagef(dst, (int4)(x, y, 1, 0), LReLU(vload4(1, s), vload4(1, alphas2)) + read_imagef(feat, n_sampler, (int4)(x, y, 1, 0)));
}

kernel void conv3x3_8to4_identity_pixelshuffle_4to1(
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
    conv3x3_cin8(src, s, 4, kernels, biases, x, y);

    float4 pixel = clamp(Identity(vload4(0, s)), 0.0f, 1.0f);
    int2 dst_coord = (int2)(x, y) * 2;
    write_imagef(dst, dst_coord + (int2)(0, 0), (float4)(pixel.s0, 0.0f, 0.0f, 1.0f));
    write_imagef(dst, dst_coord + (int2)(1, 0), (float4)(pixel.s1, 0.0f, 0.0f, 1.0f));
    write_imagef(dst, dst_coord + (int2)(0, 1), (float4)(pixel.s2, 0.0f, 0.0f, 1.0f));
    write_imagef(dst, dst_coord + (int2)(1, 1), (float4)(pixel.s3, 0.0f, 0.0f, 1.0f));
}
