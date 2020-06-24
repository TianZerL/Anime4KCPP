#include "Anime4KCPP.h"
#include <VapourSynth.h>
#include <VSHelper.h>

typedef struct {
    VSNodeRef* node = nullptr;
    VSVideoInfo vi = VSVideoInfo();
    int passes = 2;
    int pushColorCount = 2;
    double strengthColor = 0.3;
    double strengthGradient = 1.0;
    double zoomFactor = 2.0;
    bool GPU = false;
    bool CNN = false;
    bool HDN = false;
    unsigned int pID = 0, dID = 0;
    Anime4KCPP::Anime4KCreator* anime4KCreator = nullptr;
    Anime4KCPP::Parameters parameters;
}Anime4KCPPData;

static void VS_CC Anime4KCPPInit(VSMap* in, VSMap* out, void** instanceData, VSNode* node, VSCore* core, const VSAPI* vsapi)
{
    Anime4KCPPData* data = (Anime4KCPPData*)(*instanceData);
    if (data->GPU)
        data->anime4KCreator = new Anime4KCPP::Anime4KCreator(true, data->CNN, data->pID, data->dID);
    else
        data->anime4KCreator = new Anime4KCPP::Anime4KCreator(false);

    data->parameters.passes = data->passes;
    data->parameters.pushColorCount = data->pushColorCount;
    data->parameters.strengthColor = data->strengthColor;
    data->parameters.strengthGradient = data->strengthGradient;
    data->parameters.zoomFactor = data->zoomFactor;
    data->parameters.HDN = data->HDN;

    vsapi->setVideoInfo(&data->vi, 1, node);
}

static const VSFrameRef* VS_CC Anime4KCPPGetFrame(int n, int activationReason, void** instanceData, void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi)
{
    Anime4KCPPData* data = (Anime4KCPPData*)(*instanceData);

    if (activationReason == arInitial)
        vsapi->requestFrameFilter(n, data->node, frameCtx);
    else if (activationReason == arAllFramesReady)
    {
        const VSFrameRef* src = vsapi->getFrameFilter(n, data->node, frameCtx);

        int h = vsapi->getFrameHeight(src, 0);

        VSFrameRef* dst = vsapi->newVideoFrame(data->vi.format, data->vi.width, data->vi.height, src, core);

        int srcSrtide = vsapi->getStride(src, 0);

        unsigned char* srcR = const_cast<unsigned char*>(vsapi->getReadPtr(src, 0));
        unsigned char* srcG = const_cast<unsigned char*>(vsapi->getReadPtr(src, 1));
        unsigned char* srcB = const_cast<unsigned char*>(vsapi->getReadPtr(src, 2));

        unsigned char* dstR = vsapi->getWritePtr(dst, 0);
        unsigned char* dstG = vsapi->getWritePtr(dst, 1);
        unsigned char* dstB = vsapi->getWritePtr(dst, 2);

        Anime4KCPP::Anime4K* anime4K;

        if (data->CNN)
            if (data->GPU)
                anime4K = data->anime4KCreator->create(data->parameters, Anime4KCPP::ProcessorType::GPUCNN);
            else
                anime4K = data->anime4KCreator->create(data->parameters, Anime4KCPP::ProcessorType::CPUCNN);
        else
            if (data->GPU)
                anime4K = data->anime4KCreator->create(data->parameters, Anime4KCPP::ProcessorType::GPU);
            else
                anime4K = data->anime4KCreator->create(data->parameters, Anime4KCPP::ProcessorType::CPU);

        anime4K->loadImage(h, srcSrtide, srcR, srcG, srcB);
        anime4K->process();
        anime4K->saveImage(dstR, dstG, dstB);

        data->anime4KCreator->release(anime4K);
        vsapi->freeFrame(src);

        return dst;
    }
    return nullptr;
}

static const VSFrameRef* VS_CC Anime4KCPPGetFrameYUV(int n, int activationReason, void** instanceData, void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi)
{
    Anime4KCPPData* data = (Anime4KCPPData*)(*instanceData);

    if (activationReason == arInitial)
        vsapi->requestFrameFilter(n, data->node, frameCtx);
    else if (activationReason == arAllFramesReady)
    {
        const VSFrameRef* src = vsapi->getFrameFilter(n, data->node, frameCtx);

        int h = vsapi->getFrameHeight(src, 0);

        VSFrameRef* dst = vsapi->newVideoFrame(data->vi.format, data->vi.width, data->vi.height, src, core);

        int srcSrtide = vsapi->getStride(src, 0);

        unsigned char* srcY = const_cast<unsigned char*>(vsapi->getReadPtr(src, 0));
        unsigned char* srcU = const_cast<unsigned char*>(vsapi->getReadPtr(src, 1));
        unsigned char* srcV = const_cast<unsigned char*>(vsapi->getReadPtr(src, 2));

        unsigned char* dstY = vsapi->getWritePtr(dst, 0);
        unsigned char* dstU = vsapi->getWritePtr(dst, 1);
        unsigned char* dstV = vsapi->getWritePtr(dst, 2);

        Anime4KCPP::Anime4K* anime4K;

        if (data->GPU)
            anime4K = data->anime4KCreator->create(data->parameters, Anime4KCPP::ProcessorType::GPUCNN);
        else
            anime4K = data->anime4KCreator->create(data->parameters, Anime4KCPP::ProcessorType::CPUCNN);

        anime4K->loadImage(h, srcSrtide, srcY, srcU, srcV, true);
        anime4K->process();
        anime4K->saveImage(dstY, dstU, dstV);

        data->anime4KCreator->release(anime4K);
        vsapi->freeFrame(src);

        return dst;
    }
    return nullptr;
}

static const VSFrameRef* VS_CC Anime4KCPPGetFrameSafe(int n, int activationReason, void** instanceData, void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi)
{
    Anime4KCPPData* data = (Anime4KCPPData*)(*instanceData);

    if (activationReason == arInitial)
        vsapi->requestFrameFilter(n, data->node, frameCtx);
    else if (activationReason == arAllFramesReady)
    {
        const VSFrameRef* src = vsapi->getFrameFilter(n, data->node, frameCtx);

        int srcH = vsapi->getFrameHeight(src, 0);
        int srcW = vsapi->getFrameWidth(src, 0);

        int srcSrtide = vsapi->getStride(src, 0);

        VSFrameRef* dst = vsapi->newVideoFrame(data->vi.format, data->vi.width, data->vi.height, src, core);

        int dstH = vsapi->getFrameHeight(dst, 0);
        int dstW = vsapi->getFrameWidth(dst, 0);

        int dstSrtide = vsapi->getStride(dst, 0);

        const unsigned char* srcR = vsapi->getReadPtr(src, 0);
        const unsigned char* srcG = vsapi->getReadPtr(src, 1);
        const unsigned char* srcB = vsapi->getReadPtr(src, 2);

        unsigned char* srcRSafe = new unsigned char[static_cast<size_t>(srcH) * static_cast<size_t>(srcW)];
        unsigned char* srcGSafe = new unsigned char[static_cast<size_t>(srcH) * static_cast<size_t>(srcW)];
        unsigned char* srcBSafe = new unsigned char[static_cast<size_t>(srcH) * static_cast<size_t>(srcW)];

        unsigned char* dstR = vsapi->getWritePtr(dst, 0);
        unsigned char* dstG = vsapi->getWritePtr(dst, 1);
        unsigned char* dstB = vsapi->getWritePtr(dst, 2);

        cv::Mat dstRSafe;
        cv::Mat dstGSafe;
        cv::Mat dstBSafe;

        for (int y = 0; y < srcH; y++)
        {
            memcpy(srcRSafe + y * static_cast<size_t>(srcW), srcR, srcW);
            memcpy(srcGSafe + y * static_cast<size_t>(srcW), srcG, srcW);
            memcpy(srcBSafe + y * static_cast<size_t>(srcW), srcB, srcW);
            srcR += srcSrtide;
            srcG += srcSrtide;
            srcB += srcSrtide;
        }

        Anime4KCPP::Anime4K* anime4K;

        if (data->CNN)
            if (data->GPU)
                anime4K = data->anime4KCreator->create(data->parameters, Anime4KCPP::ProcessorType::GPUCNN);
            else
                anime4K = data->anime4KCreator->create(data->parameters, Anime4KCPP::ProcessorType::CPUCNN);
        else
            if (data->GPU)
                anime4K = data->anime4KCreator->create(data->parameters, Anime4KCPP::ProcessorType::GPU);
            else
                anime4K = data->anime4KCreator->create(data->parameters, Anime4KCPP::ProcessorType::CPU);

        anime4K->loadImage(srcH, srcW, srcRSafe, srcGSafe, srcBSafe);
        anime4K->process();
        anime4K->saveImage(dstRSafe, dstGSafe, dstBSafe);

        for (int y = 0; y < dstH; y++)
        {
            memcpy(dstR, dstRSafe.data + y * static_cast<size_t>(dstW), dstW);
            memcpy(dstG, dstGSafe.data + y * static_cast<size_t>(dstW), dstW);
            memcpy(dstB, dstBSafe.data + y * static_cast<size_t>(dstW), dstW);
            dstR += dstSrtide;
            dstG += dstSrtide;
            dstB += dstSrtide;
        }

        data->anime4KCreator->release(anime4K);

        delete[] srcRSafe;
        delete[] srcGSafe;
        delete[] srcBSafe;

        vsapi->freeFrame(src);

        return dst;
    }
    return nullptr;
}

static const VSFrameRef* VS_CC Anime4KCPPGetFrameYUVSafe(int n, int activationReason, void** instanceData, void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi)
{
    Anime4KCPPData* data = (Anime4KCPPData*)(*instanceData);

    if (activationReason == arInitial)
        vsapi->requestFrameFilter(n, data->node, frameCtx);
    else if (activationReason == arAllFramesReady)
    {
        const VSFrameRef* src = vsapi->getFrameFilter(n, data->node, frameCtx);

        int srcHY = vsapi->getFrameHeight(src, 0);
        int srcWY = vsapi->getFrameWidth(src, 0);
        int srcHU = vsapi->getFrameHeight(src, 1);
        int srcWU = vsapi->getFrameWidth(src, 1);
        int srcHV = vsapi->getFrameHeight(src, 2);
        int srcWV = vsapi->getFrameWidth(src, 2);

        int srcSrtideY = vsapi->getStride(src, 0);
        int srcSrtideU = vsapi->getStride(src, 1);
        int srcSrtideV = vsapi->getStride(src, 2);

        VSFrameRef* dst = vsapi->newVideoFrame(data->vi.format, data->vi.width, data->vi.height, src, core);

        int dstHY = vsapi->getFrameHeight(dst, 0);
        int dstWY = vsapi->getFrameWidth(dst, 0);
        int dstHU = vsapi->getFrameHeight(dst, 1);
        int dstWU = vsapi->getFrameWidth(dst, 1);
        int dstHV = vsapi->getFrameHeight(dst, 2);
        int dstWV = vsapi->getFrameWidth(dst, 2);

        int dstSrtideY = vsapi->getStride(dst, 0);
        int dstSrtideU = vsapi->getStride(dst, 1);
        int dstSrtideV = vsapi->getStride(dst, 2);

        const unsigned char* srcY = vsapi->getReadPtr(src, 0);
        const unsigned char* srcU = vsapi->getReadPtr(src, 1);
        const unsigned char* srcV = vsapi->getReadPtr(src, 2);

        unsigned char* srcYSafe = new unsigned char[static_cast<size_t>(srcHY) * static_cast<size_t>(srcWY)];
        unsigned char* srcUSafe = new unsigned char[static_cast<size_t>(srcHU) * static_cast<size_t>(srcWU)];
        unsigned char* srcVSafe = new unsigned char[static_cast<size_t>(srcHV) * static_cast<size_t>(srcWV)];

        unsigned char* dstY = vsapi->getWritePtr(dst, 0);
        unsigned char* dstU = vsapi->getWritePtr(dst, 1);
        unsigned char* dstV = vsapi->getWritePtr(dst, 2);

        cv::Mat dstYSafe;
        cv::Mat dstUSafe;
        cv::Mat dstVSafe;

        for (int y = 0; y < srcHY; y++)
        {
            memcpy(srcYSafe + y * static_cast<size_t>(srcWY), srcY, srcWY);
            srcY += srcSrtideY;
            if (y < srcHU)
            {
                memcpy(srcUSafe + y * static_cast<size_t>(srcWU), srcU, srcWU);
                srcU += srcSrtideU;
            }
            if (y < srcHV)
            {
                memcpy(srcVSafe + y * static_cast<size_t>(srcWV), srcV, srcWV);
                srcV += srcSrtideV;
            }
        }

        Anime4KCPP::Anime4K* anime4K;

        if (data->GPU)
            anime4K = data->anime4KCreator->create(data->parameters, Anime4KCPP::ProcessorType::GPUCNN);
        else
            anime4K = data->anime4KCreator->create(data->parameters, Anime4KCPP::ProcessorType::CPUCNN);

        anime4K->loadImage(srcHY, srcWY, srcYSafe, srcHU, srcWU, srcUSafe, srcHV, srcWV, srcVSafe);
        anime4K->process();
        anime4K->saveImage(dstYSafe, dstUSafe, dstVSafe);

        for (int y = 0; y < dstHY; y++)
        {
            memcpy(dstY, dstYSafe.data + y * static_cast<size_t>(dstWY), dstWY);
            dstY += dstSrtideY;
            if (y < dstHU)
            {
                memcpy(dstU, dstUSafe.data + y * static_cast<size_t>(dstWU), dstWU);
                dstU += dstSrtideU;
            }
            if (y < dstHV)
            {
                memcpy(dstV, dstVSafe.data + y * static_cast<size_t>(dstWV), dstWV);
                dstV += dstSrtideV;
            }
        }

        data->anime4KCreator->release(anime4K);

        delete[] srcYSafe;
        delete[] srcUSafe;
        delete[] srcVSafe;

        vsapi->freeFrame(src);

        return dst;
    }
    return nullptr;
}

static void VS_CC Anime4KCPPFree(void* instanceData, VSCore* core, const VSAPI* vsapi)
{
    Anime4KCPPData* data = (Anime4KCPPData*)instanceData;
    delete data->anime4KCreator;
    vsapi->freeNode(data->node);
    delete data;
}

static void VS_CC Anime4KCPPCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi)
{
    Anime4KCPPData tmpData;
    int err;

    tmpData.node = vsapi->propGetNode(in, "src", 0, 0);
    tmpData.vi = *vsapi->getVideoInfo(tmpData.node);

    if (!isConstantFormat(&tmpData.vi) || (tmpData.vi.format->id != pfRGB24 && (tmpData.vi.format->bitsPerSample != 8 || tmpData.vi.format->colorFamily != cmYUV)))
    {
        vsapi->setError(out, "Anime4KCPP: only RGB24 or YUV 8bit supported");
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
    else if (tmpData.strengthColor < 0.0 || tmpData.strengthColor > 1.0)
    {
        vsapi->setError(out, "Anime4KCPP: strengthColor must range from 0 to 1");
        vsapi->freeNode(tmpData.node);
        return;
    }

    tmpData.strengthGradient = vsapi->propGetFloat(in, "strengthGradient", 0, &err);
    if (err)
        tmpData.strengthGradient = 1.0;
    else if (tmpData.strengthGradient < 0.0 || tmpData.strengthGradient > 1.0)
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

    tmpData.CNN = vsapi->propGetInt(in, "ACNet", 0, &err);
    if (err)
        tmpData.CNN = false;

    if (tmpData.vi.format->colorFamily == cmYUV && tmpData.CNN == false)
    {
        vsapi->setError(out, "Anime4KCPP: YUV444P8 only for ACNet");
        vsapi->freeNode(tmpData.node);
        return;
    }

    tmpData.HDN = vsapi->propGetInt(in, "HDN", 0, &err);
    if (err)
        tmpData.HDN = false;

    tmpData.GPU = vsapi->propGetInt(in, "GPUMode", 0, &err);
    if (err)
        tmpData.GPU = false;

    tmpData.pID = vsapi->propGetInt(in, "platformID", 0, &err);
    if (err || !tmpData.GPU)
        tmpData.pID = 0;

    tmpData.dID = vsapi->propGetInt(in, "deviceID", 0, &err);
    if (err || !tmpData.GPU)
        tmpData.dID = 0;

    bool safeMode = vsapi->propGetInt(in, "safeMode", 0, &err);
    if (err)
        safeMode = true;

    if (!safeMode && tmpData.vi.format->id != pfRGB24 && tmpData.vi.format->id != pfYUV444P8)
    {
        vsapi->setError(out, "Anime4KCPP: RGB24 or YUV444P8 only if safeMode was disabled");
        vsapi->freeNode(tmpData.node);
        return;
    }

    if (tmpData.GPU)
    {
        std::pair<std::pair<int, std::vector<int>>, std::string> GPUinfo = Anime4KCPP::Anime4KGPU::listGPUs();
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
        std::pair<bool, std::string> ret =
            Anime4KCPP::Anime4KGPU::checkGPUSupport(tmpData.pID, tmpData.dID);
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
        if (!safeMode && tmpData.vi.width % 32 != 0)//32-byte alignment
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

    if (safeMode)
        if (tmpData.vi.format->colorFamily == cmYUV)
            vsapi->createFilter(in, out, "Anime4KCPP", Anime4KCPPInit, Anime4KCPPGetFrameYUVSafe, Anime4KCPPFree, fmParallel, 0, data, core);
        else
            vsapi->createFilter(in, out, "Anime4KCPP", Anime4KCPPInit, Anime4KCPPGetFrameSafe, Anime4KCPPFree, fmParallel, 0, data, core);
    else
        if (tmpData.vi.format->colorFamily == cmYUV)
            vsapi->createFilter(in, out, "Anime4KCPP", Anime4KCPPInit, Anime4KCPPGetFrameYUV, Anime4KCPPFree, fmParallel, 0, data, core);
        else
            vsapi->createFilter(in, out, "Anime4KCPP", Anime4KCPPInit, Anime4KCPPGetFrame, Anime4KCPPFree, fmParallel, 0, data, core);
}

static void VS_CC Anime4KCPPListGPUs(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi)
{
    vsapi->logMessage(mtDebug, Anime4KCPP::Anime4KGPU::listGPUs().second.c_str());
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
        "ACNet:int:opt;"
        "GPUMode:int:opt;"
        "HDN:int:opt;"
        "platformID:int:opt;"
        "deviceID:int:opt;"
        "safeMode:int:opt",
        Anime4KCPPCreate, nullptr, plugin);

    registerFunc("listGPUs", "", Anime4KCPPListGPUs, nullptr, plugin);
}
