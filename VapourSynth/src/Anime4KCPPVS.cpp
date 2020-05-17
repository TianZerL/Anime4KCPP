#include "Anime4K.h"
#include "Anime4KGPU.h"
#include <VapourSynth.h>
#include <VSHelper.h>

typedef struct {
    VSNodeRef* node;
    VSVideoInfo vi;
    int passes;
    int pushColorCount;
    double strengthColor;
    double strengthGradient;
    double zoomFactor;
    bool GPU;
    unsigned int pID, dID;
}Anime4KCPPData;

static void VS_CC Anime4KCPPInit(VSMap* in, VSMap* out, void** instanceData, VSNode* node, VSCore* core, const VSAPI* vsapi)
{
    Anime4KCPPData* data = (Anime4KCPPData*)(*instanceData);
    if (data->GPU)
        Anime4KGPU::initGPU(data->pID, data->dID);
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

        int h = vsapi->getFrameHeight(src, 0);
        int w = vsapi->getFrameWidth(src, 0);

        VSFrameRef* dst = vsapi->newVideoFrame(data->vi.format, data->vi.width, data->vi.height, src, core);

        int srcSrtide = vsapi->getStride(src, 0);

        unsigned char* srcR = const_cast<unsigned char*>(vsapi->getReadPtr(src, 0));
        unsigned char* srcG = const_cast<unsigned char*>(vsapi->getReadPtr(src, 1));
        unsigned char* srcB = const_cast<unsigned char*>(vsapi->getReadPtr(src, 2));

        unsigned char* dstR = vsapi->getWritePtr(dst, 0);
        unsigned char* dstG = vsapi->getWritePtr(dst, 1);
        unsigned char* dstB = vsapi->getWritePtr(dst, 2);

        Anime4K* anime4kcpp;

        if (data->GPU)
            anime4kcpp = new Anime4KGPU(
                data->passes, 
                data->pushColorCount, 
                data->strengthColor, 
                data->strengthGradient, 
                data->zoomFactor);
        else
            anime4kcpp = new Anime4K(
                data->passes, 
                data->pushColorCount, 
                data->strengthColor, 
                data->strengthGradient, 
                data->zoomFactor);
        
        anime4kcpp->loadImage(h, srcSrtide, srcR, srcG, srcB);
        anime4kcpp->process();
        anime4kcpp->saveImage(dstR, dstG, dstB);

        delete anime4kcpp;
        vsapi->freeFrame(src);

        return dst;
    }
    return nullptr;
}

static void VS_CC Anime4KCPPFree(void* instanceData, VSCore* core, const VSAPI* vsapi)
{
    Anime4KCPPData* data = (Anime4KCPPData*)instanceData;
    if(data->GPU)
        Anime4KGPU::releaseGPU();
    vsapi->freeNode(data->node);
    delete data;
}

static void VS_CC Anime4KCPPCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi)
{
    Anime4KCPPData tmpData;
    int err;

    tmpData.node = vsapi->propGetNode(in, "src", 0, 0);
    tmpData.vi = *vsapi->getVideoInfo(tmpData.node);

    if (!isConstantFormat(&tmpData.vi) || tmpData.vi.format->id != pfRGB24)
    {
        vsapi->setError(out, "Anime4KCPP: only RGB24 supported");
        vsapi->freeNode(tmpData.node);
        return;
    }

    tmpData.passes = vsapi->propGetInt(in, "passes", 0, &err);
    if (err)
        tmpData.passes = 2;

    tmpData.pushColorCount = vsapi->propGetInt(in, "pushColorCount", 0, &err);
    if (err)
        tmpData.pushColorCount = 2;

    tmpData.strengthColor = vsapi->propGetFloat(in, "strengthColor", 0, &err);
    if (err)
        tmpData.strengthColor = 0.3;
    else if(tmpData.strengthColor < 0.0 || tmpData.strengthColor > 1.0)
    {
        vsapi->setError(out, "Anime4KCPP: strengthColor must range from 0 to 1");
        vsapi->freeNode(tmpData.node);
        return;
    }

    tmpData.strengthGradient = vsapi->propGetFloat(in, "strengthGradient", 0, &err);
    if (err)
        tmpData.strengthGradient = 1.0;
    else if(tmpData.strengthGradient < 0.0 || tmpData.strengthGradient > 1.0)
    {
        vsapi->setError(out, "Anime4KCPP: strengthGradient must range from 0 to 1");
        vsapi->freeNode(tmpData.node);
        return;
    }

    tmpData.zoomFactor = vsapi->propGetInt(in, "zoomFactor", 0, &err);
    if (err)
        tmpData.zoomFactor = 1.0;
    else if (tmpData.zoomFactor < 1.0)
    {
        vsapi->setError(out, "Anime4KCPP: zoomFactor must be an integer which >= 1");
        vsapi->freeNode(tmpData.node);
        return;
    }

    tmpData.GPU = vsapi->propGetInt(in, "GPUMode", 0, &err);
    if (err)
        tmpData.GPU = false;

    tmpData.pID = vsapi->propGetInt(in, "platformID", 0, &err);
    if (err || !tmpData.GPU)
        tmpData.pID = 0;

    tmpData.dID = vsapi->propGetInt(in, "deviceID", 0, &err);
    if (err || !tmpData.GPU)
        tmpData.dID = 0;

    if (tmpData.GPU)
    {
        std::pair<std::pair<int, std::vector<int>>, std::string> GPUinfo = Anime4KGPU::listGPUs();
        if (tmpData.pID >= static_cast<unsigned int>(GPUinfo.first.first) || 
            tmpData.dID >= static_cast<unsigned int>(GPUinfo.first.second[tmpData.pID]))
        {
            std::ostringstream err;
            err << "Platform ID or device ID index out of range" << std::endl
                << "Run core.anim4kcpp.listGPUs for available platforms and devices" << std::endl
                << "Your input is: " << std::endl
                << "    platform ID: " << tmpData.pID << std::endl
                << "    device ID: " << tmpData.dID << std::endl;
            vsapi->setError(out, err.str().c_str());
            vsapi->freeNode(tmpData.node);
            return;
        }
        std::pair<bool,std::string> ret = 
            Anime4KGPU::checkGPUSupport(tmpData.pID, tmpData.dID);
        if (!ret.first)
        {
            std::ostringstream err;
            err << "The current device is unavailable" << std::endl
                << "Your input is: " << std::endl
                << "    platform ID: " << tmpData.pID << std::endl
                << "    device ID: " << tmpData.dID << std::endl
                << "Error: " << std::endl
                << "    " + ret.second << std::endl;
            vsapi->setError(out, err.str().c_str());
            vsapi->freeNode(tmpData.node);
            return;
        }
        vsapi->logMessage(mtDebug, ("Current GPU infomation: \n" + ret.second).c_str());
    }

    if (tmpData.zoomFactor != 1.0)
    {
        if (tmpData.vi.width % 32 != 0)//32-byte alignment
        {
            tmpData.vi.width = ((tmpData.vi.width >> 5) + 1) << 5;
            vsapi->logMessage(mtWarning, 
                "The width of the input video is not a multiple of 32 (required by VapourSynth), "
                "there will be black border of output video, please cut it off manually.");
        }
        tmpData.vi.width *= tmpData.zoomFactor;
        tmpData.vi.height *= tmpData.zoomFactor;
    }

    Anime4KCPPData* data = new Anime4KCPPData;
    *data = tmpData;

    vsapi->createFilter(in, out, "Anime4KCPP", Anime4KCPPInit, Anime4KCPPGetFrame, Anime4KCPPFree, fmParallel, 0, data, core);
}

static void VS_CC Anime4KCPPListGPUs(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi)
{
    vsapi->logMessage(mtDebug, Anime4KGPU::listGPUs().second.c_str());
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin* plugin)
{
    configFunc("github.tianzerl.anime4kcpp", "anime4kcpp", "Anime4KCPP for VapourSynth", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("Anime4KCPP",
        "src:clip;"
        "passes:int:opt;"
        "pushColorCount:int:opt;"
        "strengthColor:float:opt;"
        "strengthGradient:float:opt;"
        "zoomFactor:int:opt;"
        "GPUMode:int:opt;"
        "platformID:int:opt;"
        "deviceID:int:opt",
        Anime4KCPPCreate, nullptr, plugin);

    registerFunc("listGPUs", "", Anime4KCPPListGPUs, nullptr, plugin);
}

