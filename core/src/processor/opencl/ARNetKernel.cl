#include "CommonKernel.cl"

kernel void conv3x3_1to8_identity(
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
    conv3x3_cin1(src, s, 8, kernels, biases, x, y);

    write_imagef(dst, (int4)(x, y, 0, 0), Identity(vload4(0, s)));
    write_imagef(dst, (int4)(x, y, 1, 0), Identity(vload4(1, s)));
}

kernel void conv3x3_8to8_lrelu(
    read_only image2d_array_t src,
    write_only image2d_array_t dst,
    WEIGHTS_SPACE const float* restrict kernels,
    const int koffset,
    WEIGHTS_SPACE const float* restrict biases,
    const int boffset,
    const float negative_slope)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(src) || y >= get_image_height(src)) return;

#ifdef USE_WEIGHTS_OFFSET
    kernels += koffset;
    biases += boffset;
#endif

    float s[8];
    conv3x3_cin8(src, s, 8, kernels, biases, x, y);

    write_imagef(dst, (int4)(x, y, 0, 0), LReLU(vload4(0, s), negative_slope));
    write_imagef(dst, (int4)(x, y, 1, 0), LReLU(vload4(1, s), negative_slope));
}

kernel void conv3x3_8to8_residual_identity(
    read_only image2d_array_t src,
    write_only image2d_array_t dst,
    WEIGHTS_SPACE const float* restrict kernels,
    const int koffset,
    WEIGHTS_SPACE const float* restrict biases,
    const int boffset,
    read_only image2d_array_t id,
    const float scale)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(src) || y >= get_image_height(src)) return;

#ifdef USE_WEIGHTS_OFFSET
    kernels += koffset;
    biases += boffset;
#endif

    float s[8];
    conv3x3_cin8(src, s, 8, kernels, biases, x, y);

    write_imagef(dst, (int4)(x, y, 0, 0), Identity(vload4(0, s) * scale + read_imagef(id, n_sampler, (int4)(x, y, 0, 0))));
    write_imagef(dst, (int4)(x, y, 1, 0), Identity(vload4(1, s) * scale + read_imagef(id, n_sampler, (int4)(x, y, 1, 0))));
}

kernel void conv3x3_8to8_residual_add_identity(
    read_only image2d_array_t src,
    write_only image2d_array_t dst,
    WEIGHTS_SPACE const float* restrict kernels,
    const int koffset,
    WEIGHTS_SPACE const float* restrict biases,
    const int boffset,
    read_only image2d_array_t id,
    const float scale,
    read_only image2d_array_t feat)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(src) || y >= get_image_height(src)) return;

#ifdef USE_WEIGHTS_OFFSET
    kernels += koffset;
    biases += boffset;
#endif

    float s[8];
    conv3x3_cin8(src, s, 8, kernels, biases, x, y);

    write_imagef(dst, (int4)(x, y, 0, 0), Identity(vload4(0, s) * scale + read_imagef(id, n_sampler, (int4)(x, y, 0, 0)) + read_imagef(feat, n_sampler, (int4)(x, y, 0, 0))));
    write_imagef(dst, (int4)(x, y, 1, 0), Identity(vload4(1, s) * scale + read_imagef(id, n_sampler, (int4)(x, y, 1, 0)) + read_imagef(feat, n_sampler, (int4)(x, y, 1, 0))));
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
