#pragma OPENCL EXTENSION cl_khr_fp16 : enable
constant sampler_t n_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
kernel void conv3x3_1to8(
    read_only image2d_t src,
    write_only image2d_array_t dst,
    constant half* kernels,
    const int koffset,
    constant half* baises,
    const int boffset
)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(src) || y >= get_image_height(src)) return;

    constant half* kptr = kernels + koffset;
    constant half* bptr = baises + boffset;

    half4 tl = read_imageh(src, n_sampler, (int2)(x-1, y-1));
    half4 tc = read_imageh(src, n_sampler, (int2)(x  , y-1));
    half4 tr = read_imageh(src, n_sampler, (int2)(x+1, y-1));
    half4 ml = read_imageh(src, n_sampler, (int2)(x-1, y  ));
    half4 mc = read_imageh(src, n_sampler, (int2)(x  , y  ));
    half4 mr = read_imageh(src, n_sampler, (int2)(x+1, y  ));
    half4 bl = read_imageh(src, n_sampler, (int2)(x-1, y+1));
    half4 bc = read_imageh(src, n_sampler, (int2)(x  , y+1));
    half4 br = read_imageh(src, n_sampler, (int2)(x+1, y+1));

    half8 r0 = (half8)(tl.s0, tc.s0, tr.s0, ml.s0, mc.s0, mr.s0, bl.s0, bc.s0);
    half r8 = br.s0;

    half s[8] = {};
    for(int n = 0; n < 8; n++)
    {
        half8 k0 = vload8(0, kptr + n * 9 + 0);
        half k8 = *(kptr + n * 9 + 8);
        half8 h8 = r0 * k0;
        s[n] = fmax(h8.s0 + h8.s1 + h8.s2 + h8.s3 + h8.s4 + h8.s5 + h8.s6 + h8.s7 + r8 * k8 + bptr[n], 0.0h);
    }
    write_imageh(dst, (int4)(x, y, 0, 0), (half4)(s[0], s[1], s[2], s[3]));
    write_imageh(dst, (int4)(x, y, 1, 0), (half4)(s[4], s[5], s[6], s[7]));
}
kernel void conv3x3_8to8(
    read_only image2d_array_t src,
    write_only image2d_array_t dst,
    constant half* kernels,
    const int koffset,
    constant half* baises,
    const int boffset
)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(src) || y >= get_image_height(src)) return;

    constant half* kptr = kernels + koffset;
    constant half* bptr = baises + boffset;

    half8 r0 = (half8)(read_imageh(src, n_sampler, (int4)(x-1, y-1, 0, 0)), read_imageh(src, n_sampler, (int4)(x-1, y-1, 1, 0)));
    half8 r1 = (half8)(read_imageh(src, n_sampler, (int4)(x  , y-1, 0, 0)), read_imageh(src, n_sampler, (int4)(x  , y-1, 1, 0)));
    half8 r2 = (half8)(read_imageh(src, n_sampler, (int4)(x+1, y-1, 0, 0)), read_imageh(src, n_sampler, (int4)(x+1, y-1, 1, 0)));
    half8 r3 = (half8)(read_imageh(src, n_sampler, (int4)(x-1, y  , 0, 0)), read_imageh(src, n_sampler, (int4)(x-1, y  , 1, 0)));
    half8 r4 = (half8)(read_imageh(src, n_sampler, (int4)(x  , y  , 0, 0)), read_imageh(src, n_sampler, (int4)(x  , y  , 1, 0)));
    half8 r5 = (half8)(read_imageh(src, n_sampler, (int4)(x+1, y  , 0, 0)), read_imageh(src, n_sampler, (int4)(x+1, y  , 1, 0)));
    half8 r6 = (half8)(read_imageh(src, n_sampler, (int4)(x-1, y+1, 0, 0)), read_imageh(src, n_sampler, (int4)(x-1, y+1, 1, 0)));
    half8 r7 = (half8)(read_imageh(src, n_sampler, (int4)(x  , y+1, 0, 0)), read_imageh(src, n_sampler, (int4)(x  , y+1, 1, 0)));
    half8 r8 = (half8)(read_imageh(src, n_sampler, (int4)(x+1, y+1, 0, 0)), read_imageh(src, n_sampler, (int4)(x+1, y+1, 1, 0)));

    half s[8] = {};
    for(int n = 0; n < 8; n++)
    {
        constant half* k = kptr + n * 8 * 9;

        half8 k0 = vload8(0, k);
        half8 k1 = vload8(1, k);
        half8 k2 = vload8(2, k);
        half8 k3 = vload8(3, k);
        half8 k4 = vload8(4, k);
        half8 k5 = vload8(5, k);
        half8 k6 = vload8(6, k);
        half8 k7 = vload8(7, k);
        half8 k8 = vload8(8, k);

        half8 s0 = 0, s1 = 0, s2 = 0;

        s0 = mad(r0, k0, s0);
        s1 = mad(r1, k1, s1);
        s2 = mad(r2, k2, s2);
        s0 = mad(r3, k3, s0);
        s1 = mad(r4, k4, s1);
        s2 = mad(r5, k5, s2);
        s0 = mad(r6, k6, s0);
        s1 = mad(r7, k7, s1);
        s2 = mad(r8, k8, s2);

        half8 h8 = s0 + s1 + s2;
        s[n] = fmax(h8.s0 + h8.s1 + h8.s2 + h8.s3 + h8.s4 + h8.s5 + h8.s6 + h8.s7 + bptr[n], 0.0h);
    }
    write_imageh(dst, (int4)(x, y, 0, 0), (half4)(s[0], s[1], s[2], s[3]));
    write_imageh(dst, (int4)(x, y, 1, 0), (half4)(s[4], s[5], s[6], s[7]));
}
kernel void deconv2x2_8to1(
    read_only image2d_array_t src,
    write_only image2d_t dst,
    constant half* kernels,
    const int koffset
)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(dst) || y >= get_image_height(dst)) return;

    constant half* kptr = kernels + koffset;

    int2 dst_coord = (int2)(x, y);
    int2 src_coord = dst_coord / 2;
    int2 pos = dst_coord & 1;
    int index = pos.y * 2 + pos.x;

    half8 r = (half8)(read_imageh(src, n_sampler, (int4)(src_coord, 0, 0)), read_imageh(src, n_sampler, (int4)(src_coord, 1, 0)));
    half8 k = (half8)(
        kptr[ 0 + index],
        kptr[ 4 + index],
        kptr[ 8 + index],
        kptr[12 + index],
        kptr[16 + index],
        kptr[20 + index],
        kptr[24 + index],
        kptr[28 + index]
    );
    half8 h8 = r * k;
    half4 s = (half4)(clamp(h8.s0 + h8.s1 + h8.s2 + h8.s3 + h8.s4 + h8.s5 + h8.s6 + h8.s7, 0.0h, 1.0h), 0.0h, 0.0h, 1.0h);
    write_imageh(dst, dst_coord, s);
}
