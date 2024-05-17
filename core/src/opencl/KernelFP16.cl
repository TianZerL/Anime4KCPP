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
        half k8 = kptr[n * 9 + 8];
        s[n] = fmax(dot(r0.lo, k0.lo) + dot(r0.hi, k0.hi) + r8 * k8 + bptr[n], 0.0h);
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
        half8 k0 = vload8(0, kptr);
        half8 k1 = vload8(1, kptr);
        half8 k2 = vload8(2, kptr);
        half8 k3 = vload8(3, kptr);
        half8 k4 = vload8(4, kptr);
        half8 k5 = vload8(5, kptr);
        half8 k6 = vload8(6, kptr);
        half8 k7 = vload8(7, kptr);
        half8 k8 = vload8(8, kptr);
        kptr += 8 * 9;
        half s0 = dot(r0.lo, k0.lo) + dot(r0.hi, k0.hi);
        half s1 = dot(r1.lo, k1.lo) + dot(r1.hi, k1.hi);
        half s2 = dot(r2.lo, k2.lo) + dot(r2.hi, k2.hi);
        half s3 = dot(r3.lo, k3.lo) + dot(r3.hi, k3.hi);
        half s4 = dot(r4.lo, k4.lo) + dot(r4.hi, k4.hi);
        half s5 = dot(r5.lo, k5.lo) + dot(r5.hi, k5.hi);
        half s6 = dot(r6.lo, k6.lo) + dot(r6.hi, k6.hi);
        half s7 = dot(r7.lo, k7.lo) + dot(r7.hi, k7.hi);
        half s8 = dot(r8.lo, k8.lo) + dot(r8.hi, k8.hi);

        s[n] = fmax(s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + bptr[n], 0.0h);
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
    half4 s = (half4)(clamp(dot(r.lo, k.lo) + dot(r.hi, k.hi), 0.0h, 1.0h), 0.0h, 0.0h, 1.0h);
    write_imageh(dst, dst_coord, s);
}
