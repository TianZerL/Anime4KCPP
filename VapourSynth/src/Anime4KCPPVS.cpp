#include "Anime4K.h"
#include "Anime4KGPU.h"
#include <VapourSynth.h>
#include <VSHelper.h>

typedef struct {
    VSNodeRef* node;
    VSVideoInfo vi;
    int passes;
    double strengthColor;
    double strengthGradient;
    double zoomFactor;
    bool GPU;
}Anime4KCPPData;

static void VS_CC Anime4KCPPInit(VSMap* in, VSMap* out, void** instanceData, VSNode* node, VSCore* core, const VSAPI* vsapi)
{
    Anime4KCPPData* data = (Anime4KCPPData*)(*instanceData);
    vsapi->setVideoInfo(&data->vi, 1, node);
}

static const VSFrameRef *VS_CC Anime4KCPPGetFrame(int n, int activationReason, void** instanceData, void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi)
{
    Anime4KCPPData* data = (Anime4KCPPData*)(*instanceData);

    if (activationReason == arInitial)
        vsapi->requestFrameFilter(n, data->node, frameCtx);
    else if (activationReason == arAllFramesReady)
    {
        const VSFrameRef* src = vsapi->getFrameFilter(n, data->node, frameCtx);
        const VSFormat* fi = data->vi.format;
        int h = vsapi->getFrameHeight(src, 0);
        int w = vsapi->getFrameWidth(src, 0);

        VSFrameRef* dst = vsapi->newVideoFrame(fi, data->vi.width, data->vi.height, src, core);

        int srcSrtide = vsapi->getStride(src, 0);

        unsigned char* srcR = const_cast<unsigned char*>(vsapi->getReadPtr(src, 0));
        unsigned char* srcG = const_cast<unsigned char*>(vsapi->getReadPtr(src, 1));
        unsigned char* srcB = const_cast<unsigned char*>(vsapi->getReadPtr(src, 2));

        unsigned char* dstR = vsapi->getWritePtr(dst, 0);
        unsigned char* dstG = vsapi->getWritePtr(dst, 1);
        unsigned char* dstB = vsapi->getWritePtr(dst, 2);

        Anime4K* anime4kcpp;

        if (data->GPU)
            anime4kcpp = new Anime4KGPU(data->passes, 2, data->strengthColor, data->strengthGradient, data->zoomFactor);
        else
            anime4kcpp = new Anime4K(data->passes, 2, data->strengthColor, data->strengthGradient, data->zoomFactor);
        
        anime4kcpp->loadImage(h, srcSrtide, srcR, srcG, srcB);
        anime4kcpp->process();
        anime4kcpp->saveImage(dstR, dstG, dstB);

        delete anime4kcpp;
        vsapi->freeFrame(src);

        return dst;
    }
    return 0;
}

static void VS_CC Anime4KCPPFree(void* instanceData, VSCore* core, const VSAPI* vsapi)
{
    Anime4KCPPData* data = (Anime4KCPPData*)instanceData;
    vsapi->freeNode(data->node);
    delete data;
}

static void VS_CC Anime4KCPPCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi)
{
    Anime4KCPPData tmpData;
    int err;

    tmpData.node = vsapi->propGetNode(in, "src", 0, 0);
    tmpData.vi = *vsapi->getVideoInfo(tmpData.node);

    if (!isConstantFormat(&tmpData.vi) || tmpData.vi.format->colorFamily != cmRGB)
    {
        vsapi->setError(out, "Anime4KCPP: only RGB24 supported");
        vsapi->freeNode(tmpData.node);
        return;
    }

    tmpData.passes = vsapi->propGetInt(in, "passes", 0, &err);
    if (err)
        tmpData.passes = 2;

    tmpData.strengthColor = vsapi->propGetFloat(in, "strengthColor", 0, &err);
    if (err)
        tmpData.strengthColor = 0.3;

    tmpData.strengthGradient = vsapi->propGetFloat(in, "strengthGradient", 0, &err);
    if (err)
        tmpData.strengthGradient = 1.0;

    tmpData.zoomFactor = vsapi->propGetFloat(in, "upscale", 0, &err);
    if (err)
        tmpData.zoomFactor = 2;

    tmpData.GPU = vsapi->propGetInt(in, "GPUMode", 0, &err);
    if (err)
        tmpData.passes = false;

    if (tmpData.zoomFactor != 1)
    {
        if (tmpData.vi.width % 32 != 0)//32-byte alignment
            tmpData.vi.width = 32 * (tmpData.vi.width / 32 + 1);
        tmpData.vi.width *= tmpData.zoomFactor;
        tmpData.vi.height *= tmpData.zoomFactor;
        if (tmpData.vi.width % 32 != 0)//32-byte alignment
            tmpData.vi.width = 32 * (tmpData.vi.width / 32 + 1);
    }

    Anime4KCPPData* data = new Anime4KCPPData;
    *data = tmpData;

    vsapi->createFilter(in, out, "Anime4KCPP", Anime4KCPPInit, Anime4KCPPGetFrame, Anime4KCPPFree, fmParallel, 0, data, core);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin* plugin)
{
    configFunc("github.tianzerl.anime4kcpp", "anime4kcpp", "Anime4KCPP for VapourSynth", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("Anime4KCPP", 
        "src:clip;"
        "passes:int:opt;"
        "strengthColor:float:opt;"
        "strengthGradient:float:opt;"
        "upscale:float:opt;"
        "GPUMode:int:opt;"
        , Anime4KCPPCreate, 0, plugin);
}

