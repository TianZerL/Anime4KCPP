#define MAX3(a, b, c) max(max(a,b),c)
#define MIN3(a, b, c) min(min(a,b),c)

__constant sampler_t samplers = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

void getLightest(float4 *mc, float4 *a, float4 *b, float4 *c, float strength)
{
    (*mc).x = (*mc).x * (1 - strength) + (((*a).x + (*b).x + (*c).x) / 3) * strength;
    (*mc).y = (*mc).y * (1 - strength) + (((*a).y + (*b).y + (*c).y) / 3) * strength;
    (*mc).z = (*mc).z * (1 - strength) + (((*a).z + (*b).z + (*c).z) / 3) * strength;
    (*mc).w = (*mc).w * (1 - strength) + (((*a).w + (*b).w + (*c).w) / 3) * strength;
}

void getAVerage(float4 *mc, float4 *a, float4 *b, float4 *c, float strength)
{
    (*mc).x = (*mc).x * (1 - strength) + (((*a).x + (*b).x + (*c).x) / 3) * strength;
    (*mc).y = (*mc).y * (1 - strength) + (((*a).y + (*b).y + (*c).y) / 3) * strength;
    (*mc).z = (*mc).z * (1 - strength) + (((*a).z + (*b).z + (*c).z) / 3) * strength;
    (*mc).w = 1.0f;
}

__kernel void getGray(__read_only image2d_t srcImg, __write_only image2d_t dstImg) 
{
    const int x = get_global_id(0), y = get_global_id(1);
    int2 coord = (int2)(x, y);
    float4 BGRA = read_imagef(srcImg, samplers, coord);
    BGRA.w = 0.299 * BGRA.z  + 0.587 * BGRA.y  + 0.114 * BGRA.x ;
    write_imagef(dstImg, coord, BGRA);
}

__kernel void pushColor(__read_only image2d_t srcImg, __write_only image2d_t dstImg, float strength)
{
    const int x = get_global_id(0), y = get_global_id(1);
    int2 coord = (int2)(x, y);

    float4 tl = read_imagef(srcImg, samplers, (int2)(x-1,y-1));
    float4 tc = read_imagef(srcImg, samplers, (int2)(x,y-1));
    float4 tr = read_imagef(srcImg, samplers, (int2)(x+1,y-1));
    float4 ml = read_imagef(srcImg, samplers, (int2)(x-1,y));
    float4 mc = read_imagef(srcImg, samplers, coord);
    float4 mr = read_imagef(srcImg, samplers, (int2)(x+1,y));
    float4 bl = read_imagef(srcImg, samplers, (int2)(x-1,y+1));
    float4 bc = read_imagef(srcImg, samplers, (int2)(x,y+1));
    float4 br = read_imagef(srcImg, samplers, (int2)(x+1,y+1));

    float maxD,minL;

    //top and bottom
    maxD = MAX3(bl.w, bc.w, br.w);
    minL = MIN3(tl.w, tc.w, tr.w);
    if (minL > mc.w && mc.w > maxD)
        getLightest(&mc, &tl, &tc, &tr, strength);
    else
    {
        maxD = MAX3(tl.w, tc.w, tr.w);
        minL = MIN3(bl.w, bc.w, br.w);
        if (minL > mc.w && mc.w > maxD)
            getLightest(&mc, &bl, &bc, &br, strength);
    }

    //sundiagonal
    maxD = MAX3(ml.w, mc.w, bc.w);
    minL = MIN3(tc.w, tr.w, mr.w);
    if (minL > maxD)
        getLightest(&mc, &tc, &tr, &mr, strength);
    else
    {
        maxD = MAX3(tc.w, mc.w, mr.w);
        minL = MIN3(ml.w, bl.w, bc.w);
        if (minL > maxD)
            getLightest(&mc, &ml, &bl, &bc, strength);
    }

    //left and right
    maxD = MAX3(tl.w, ml.w, bl.w);
    minL = MIN3(tr.w, mr.w, br.w);
    if (minL > mc.w && mc.w > maxD)
        getLightest(&mc, &tr, &mr, &br, strength);
    else
    {
        maxD = MAX3(tr.w, mr.w, br.w);
        minL = MIN3(tl.w, ml.w, bl.w);
        if (minL > mc.w && mc.w > maxD)
            getLightest(&mc, &tl, &ml, &bl, strength);
    }

    //diagonal
    maxD = MAX3(tc.w, mc.w, ml.w);
    minL = MIN3(mr.w, br.w, bc.w);
    if (minL > maxD)
        getLightest(&mc, &mr, &br, &bc, strength);
    else
    {
        maxD = MAX3(bc.w, mc.w, mr.w);
        minL = MIN3(ml.w, tl.w, tc.w);
        if (minL > maxD)
            getLightest(&mc, &ml, &tl, &tc, strength);
    }
    
    write_imagef(dstImg, coord, mc);
}

__kernel void getGradient(__read_only image2d_t srcImg, __write_only image2d_t dstImg)
{
    const int x = get_global_id(0), y = get_global_id(1);
    int2 coord = (int2)(x, y);

    float4 tl = read_imagef(srcImg, samplers, (int2)(x-1,y-1));
    float4 tc = read_imagef(srcImg, samplers, (int2)(x,y-1));
    float4 tr = read_imagef(srcImg, samplers, (int2)(x+1,y-1));
    float4 ml = read_imagef(srcImg, samplers, (int2)(x-1,y));
    float4 mc = read_imagef(srcImg, samplers, coord);
    float4 mr = read_imagef(srcImg, samplers, (int2)(x+1,y));
    float4 bl = read_imagef(srcImg, samplers, (int2)(x-1,y+1));
    float4 bc = read_imagef(srcImg, samplers, (int2)(x,y+1));
    float4 br = read_imagef(srcImg, samplers, (int2)(x+1,y+1));

    const float gradX = tr.w + mr.w + mr.w + br.w - tl.w - ml.w - ml.w - bl.w;
    const float gradY = tl.w + tc.w + tc.w + tr.w - bl.w - bc.w - bc.w - br.w;

    const float grad = clamp(sqrt(gradX * gradX + gradY * gradY), 0.0f, 1.0f);
    mc.w = 1.0f - grad;

    write_imagef(dstImg, coord, mc);
}

__kernel void pushGradient(__read_only image2d_t srcImg, __write_only image2d_t dstImg, float strength)
{
    const int x = get_global_id(0), y = get_global_id(1);
    int2 coord = (int2)(x, y);

    float4 tl = read_imagef(srcImg, samplers, (int2)(x-1,y-1));
    float4 tc = read_imagef(srcImg, samplers, (int2)(x,y-1));
    float4 tr = read_imagef(srcImg, samplers, (int2)(x+1,y-1));
    float4 ml = read_imagef(srcImg, samplers, (int2)(x-1,y));
    float4 mc = read_imagef(srcImg, samplers, coord);
    float4 mr = read_imagef(srcImg, samplers, (int2)(x+1,y));
    float4 bl = read_imagef(srcImg, samplers, (int2)(x-1,y+1));
    float4 bc = read_imagef(srcImg, samplers, (int2)(x,y+1));
    float4 br = read_imagef(srcImg, samplers, (int2)(x+1,y+1));

    float maxD,minL;

    //top and bottom
    maxD = MAX3(bl.w, bc.w, br.w);
    minL = MIN3(tl.w, tc.w, tr.w);
    if (minL > mc.w && mc.w > maxD)
    {
        getAVerage(&mc, &tl, &tc, &tr, strength);
        write_imagef(dstImg, coord, mc);
        return;
    }    

    maxD = MAX3(tl.w, tc.w, tr.w);
    minL = MIN3(bl.w, bc.w, br.w);
    if (minL > mc.w && mc.w > maxD)
    {
        getAVerage(&mc, &bl, &bc, &br, strength);
        write_imagef(dstImg, coord, mc);
        return;
    }

    //sundiagonal
    maxD = MAX3(ml.w, mc.w, bc.w);
    minL = MIN3(tc.w, tr.w, mr.w);
    if (minL > maxD)
    {
        getAVerage(&mc, &tc, &tr, &mr, strength);
        write_imagef(dstImg, coord, mc);
        return;
    }

    maxD = MAX3(tc.w, mc.w, mr.w);
    minL = MIN3(ml.w, bl.w, bc.w);
    if (minL > maxD)
    {
        getAVerage(&mc, &ml, &bl, &bc, strength);
        write_imagef(dstImg, coord, mc);
        return;
    }

    //left and right
    maxD = MAX3(tl.w, ml.w, bl.w);
    minL = MIN3(tr.w, mr.w, br.w);
    if (minL > mc.w && mc.w > maxD)
    {
        getAVerage(&mc, &tr, &mr, &br, strength);
        write_imagef(dstImg, coord, mc);
        return;
    }

    maxD = MAX3(tr.w, mr.w, br.w);
    minL = MIN3(tl.w, ml.w, bl.w);
    if (minL > mc.w && mc.w > maxD)
    {
        getAVerage(&mc, &tl, &ml, &bl, strength);
        write_imagef(dstImg, coord, mc);
        return;
    }

    //diagonal
    maxD = MAX3(tc.w, mc.w, ml.w);
    minL = MIN3(mr.w, br.w, bc.w);
    if (minL > maxD)
    {
        getAVerage(&mc, &mr, &br, &bc, strength);
        write_imagef(dstImg, coord, mc);
        return;
    }    
    maxD = MAX3(bc.w, mc.w, mr.w);
    minL = MIN3(ml.w, tl.w, tc.w);
    if (minL > maxD)
    {
        getAVerage(&mc, &ml, &tl, &tc, strength);
        write_imagef(dstImg, coord, mc);
        return;
    }
}