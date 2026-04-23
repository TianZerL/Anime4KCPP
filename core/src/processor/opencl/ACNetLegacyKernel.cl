#include "CommonKernel.cl"

kernel void conv3x3_1to8_relu(
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

    write_imagef(dst, (int4)(x, y, 0, 0), ReLU(vload4(0, s)));
    write_imagef(dst, (int4)(x, y, 1, 0), ReLU(vload4(1, s)));
}

kernel void conv3x3_8to8_relu(
    read_only image2d_array_t src,
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
    conv3x3_cin8(src, s, 8, kernels, biases, x, y);

    write_imagef(dst, (int4)(x, y, 0, 0), ReLU(vload4(0, s)));
    write_imagef(dst, (int4)(x, y, 1, 0), ReLU(vload4(1, s)));
}

kernel void deconv2x2_8to1(
    read_only image2d_array_t src,
    write_only image2d_t dst,
    WEIGHTS_SPACE const float* restrict kernels,
    const int koffset)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(dst) || y >= get_image_height(dst)) return;

#ifdef USE_WEIGHTS_OFFSET
    kernels += koffset;
#endif

    int2 dst_coord = (int2)(x, y);
    int2 src_coord = dst_coord / 2;
    int2 pos = dst_coord & 1;
    int index = pos.y * 2 + pos.x;

    float8 r = (float8)(read_imagef(src, n_sampler, (int4)(src_coord, 0, 0)), read_imagef(src, n_sampler, (int4)(src_coord, 1, 0)));
    float8 k = vload8(0, kernels + 8 * index);
    float4 s = (float4)(clamp(dot(r.lo, k.lo) + dot(r.hi, k.hi), 0.0f, 1.0f), 0.0f, 0.0f, 1.0f);
    write_imagef(dst, dst_coord, s);
}
