#include "CommonKernel.cl"

kernel void conv3x3_1to32_identity(
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

    float s[4];

    for (int i = 0; i < 8; i++)
    {
        conv3x3_cin1(src, s, 4, kernels + 1 * 9 * 4 * i, biases + 4 * i, x, y);
        write_imagef(dst, (int4)(x, y, i, 0), Identity(vload4(0, s)));
    }
}

kernel void conv3x3_32to32_relu(
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

    float buffer[16];

    for(int n = 0; n < 2; n++)
    {
        float16 s = vload16(n, biases);
        for(int c = 0; c < 4; c++)
        {
            conv3x3_cin8_chunk(src, buffer, c, 32, 16, kernels + 32 * 9 * 16 * n, x, y);
            s += vload16(0, buffer);
        }
        write_imagef(dst, (int4)(x, y, n * 4 + 0, 0), ReLU(s.s0123));
        write_imagef(dst, (int4)(x, y, n * 4 + 1, 0), ReLU(s.s4567));
        write_imagef(dst, (int4)(x, y, n * 4 + 2, 0), ReLU(s.s89ab));
        write_imagef(dst, (int4)(x, y, n * 4 + 3, 0), ReLU(s.scdef));
    }
}

kernel void conv3x3_32to32_identity_add(
    read_only image2d_array_t src,
    write_only image2d_array_t dst,
    WEIGHTS_SPACE const float* restrict kernels,
    const int koffset,
    WEIGHTS_SPACE const float* restrict biases,
    const int boffset,
    read_only image2d_array_t feat)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(src) || y >= get_image_height(src)) return;

#ifdef USE_WEIGHTS_OFFSET
    kernels += koffset;
    biases += boffset;
#endif

    float buffer[16];

    for(int n = 0; n < 2; n++)
    {
        float16 s = vload16(n, biases);
        for(int c = 0; c < 4; c++)
        {
            conv3x3_cin8_chunk(src, buffer, c, 32, 16, kernels + 32 * 9 * 16 * n, x, y);
            s += vload16(0, buffer);
        }
        write_imagef(dst, (int4)(x, y, n * 4 + 0, 0), Identity(s.s0123) + read_imagef(feat, n_sampler, (int4)(x, y, n * 4 + 0, 0)));
        write_imagef(dst, (int4)(x, y, n * 4 + 1, 0), Identity(s.s4567) + read_imagef(feat, n_sampler, (int4)(x, y, n * 4 + 1, 0)));
        write_imagef(dst, (int4)(x, y, n * 4 + 2, 0), Identity(s.s89ab) + read_imagef(feat, n_sampler, (int4)(x, y, n * 4 + 2, 0)));
        write_imagef(dst, (int4)(x, y, n * 4 + 3, 0), Identity(s.scdef) + read_imagef(feat, n_sampler, (int4)(x, y, n * 4 + 3, 0)));
    }
}

kernel void conv3x3_32to4_identity_pixelshuffle_4to1(
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
    conv3x3(src, s, 32, 4, kernels, biases, x, y);

    float4 pixel = clamp(Identity(vload4(0, s)), 0.0f, 1.0f);
    int2 dst_coord = (int2)(x, y) * 2;
    write_imagef(dst, dst_coord + (int2)(0, 0), (float4)(pixel.s0, 0.0f, 0.0f, 1.0f));
    write_imagef(dst, dst_coord + (int2)(1, 0), (float4)(pixel.s1, 0.0f, 0.0f, 1.0f));
    write_imagef(dst, dst_coord + (int2)(0, 1), (float4)(pixel.s2, 0.0f, 0.0f, 1.0f));
    write_imagef(dst, dst_coord + (int2)(1, 1), (float4)(pixel.s3, 0.0f, 0.0f, 1.0f));
}
