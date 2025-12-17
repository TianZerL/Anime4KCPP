#include <cstdint>
#include <memory>

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include "AC/Core.hpp"

#define EXIT_WITH_ERROR(msg) do { vsapi->mapSetError(out, (msg)); if (node) vsapi->freeNode(node); return; } while(0)

struct Context
{
    int type;
    double factor;
    std::shared_ptr<ac::core::Processor> processor;
    VSNode* node;
    VSVideoInfo vi;
};

static const VSFrame* VS_CC filter(int n, int activationReason, void* instanceCtx, void** /*frameData*/, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi)
{
    auto ctx = static_cast<Context*>(instanceCtx);

    if (activationReason == arInitial) vsapi->requestFrameFilter(n, ctx->node, frameCtx);
    else if (activationReason == arAllFramesReady)
    {
        auto src = vsapi->getFrameFilter(n, ctx->node, frameCtx);
        auto fi = vsapi->getVideoFrameFormat(src);
        auto dst = vsapi->newVideoFrame(fi, ctx->vi.width, ctx->vi.height, src, core);
        //y
        ac::core::Image srcy{ vsapi->getFrameWidth(src, 0), vsapi->getFrameHeight(src, 0), 1, ctx->type, const_cast<std::uint8_t*>(vsapi->getReadPtr(src, 0)), static_cast<int>(vsapi->getStride(src, 0)) };
        ac::core::Image dsty{ vsapi->getFrameWidth(dst, 0), vsapi->getFrameHeight(dst, 0), 1, ctx->type, vsapi->getWritePtr(dst, 0), static_cast<int>(vsapi->getStride(dst, 0)) };
        ctx->processor->process(srcy, dsty, ctx->factor);
        if (!ctx->processor->ok()) vsapi->setFilterError(ctx->processor->error(), frameCtx);
        //uv
        for (int p = 1; p < fi->numPlanes; p++)
        {
            ac::core::Image srcp{ vsapi->getFrameWidth(src, p), vsapi->getFrameHeight(src, p), 1, ctx->type, const_cast<std::uint8_t*>(vsapi->getReadPtr(src, p)), static_cast<int>(vsapi->getStride(src, p)) };
            ac::core::Image dstp{ vsapi->getFrameWidth(dst, p), vsapi->getFrameHeight(dst, p), 1, ctx->type, vsapi->getWritePtr(dst, p), static_cast<int>(vsapi->getStride(dst, p)) };
            ac::core::resize(srcp, dstp, 0.0, 0.0);
        }

        vsapi->freeFrame(src);
        return dst;
    }
    return nullptr;
}

static void VS_CC destroy(void* instanceCtx, VSCore* /*core*/, const VSAPI* vsapi)
{
    auto ctx = static_cast<Context*>(instanceCtx);
    vsapi->freeNode(ctx->node);
    delete ctx;
}

static void VS_CC create(const VSMap* in, VSMap* out, void* /*userData*/, VSCore* core, const VSAPI* vsapi)
{
    int err = peSuccess;

    auto node = vsapi->mapGetNode(in, "clip", 0, &err);
    if (err != peSuccess) EXIT_WITH_ERROR("Anime4KCPP: no clip");
    auto vi = vsapi->getVideoInfo(node);

    auto type = [&]() ->int {
        if (vsh::isConstantVideoFormat(vi) && (vi->format.colorFamily == cfYUV || vi->format.colorFamily == cfGray))
        {
            if (vi->format.sampleType == stInteger && vi->format.bitsPerSample == 8) return ac::core::Image::UInt8;
            if (vi->format.sampleType == stInteger && vi->format.bitsPerSample == 16) return ac::core::Image::UInt16;
            if (vi->format.sampleType == stFloat && vi->format.bitsPerSample == 32) return ac::core::Image::Float32;
        }
        return 0;
    }();
    if (!type) EXIT_WITH_ERROR("Anime4KCPP: only planar YUV uint8, uint16 and float32 input supported");

    auto factor = static_cast<double>(vsapi->mapGetFloat(in, "factor", 0, &err));
    if (err != peSuccess) factor = 2.0;
    if (factor <= 1.0) EXIT_WITH_ERROR("Anime4KCPP: this is a upscaler, so make sure factor > 1.0");

    auto processorType = vsapi->mapGetData(in, "processor", 0, &err);
    if (err != peSuccess) processorType = "cpu";

    auto device = static_cast<int>(vsapi->mapGetInt(in, "device", 0, &err));
    if (err != peSuccess) device = 0;
    if (device < 0) EXIT_WITH_ERROR("Anime4KCPP: the device index cannot be negative");

    auto model = vsapi->mapGetData(in, "model", 0, &err);
    if (err != peSuccess) model = "acnet-hdn0";

    auto processor = ac::core::Processor::create(processorType, device, model);
    if (!processor->ok()) EXIT_WITH_ERROR(processor->error());

    auto ctx = new Context{};
    ctx->node = node;
    ctx->vi = *vi;
    ctx->vi.width = static_cast<decltype(ctx->vi.width)>(vi->width * factor);
    ctx->vi.height = static_cast<decltype(ctx->vi.height)>(vi->height * factor);
    ctx->type = type;
    ctx->factor = factor;
    ctx->processor = processor;

    VSFilterDependency deps[] = { {node, rpGeneral} };
    vsapi->createVideoFilter(out, "Upscale", &ctx->vi, filter, destroy, fmParallel, deps, 1, ctx, core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi)
{
    vspapi->configPlugin("github.tianzerl.anime4kcpp", "anime4kcpp", "Anime4KCPP for VapourSynth", VS_MAKE_VERSION(3, 1), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("ACUpscale",
        "clip:vnode;"
        "factor:float:opt;"
        "processor:data:opt;"
        "device:int:opt;"
        "model:data:opt;",
        "clip:vnode;", create, nullptr, plugin);

    vspapi->registerFunction("ACInfoList",
        "",
        "info:data[];",
        [](const VSMap* /*in*/, VSMap* out, void* /*userData*/, VSCore* /*core*/, const VSAPI* vsapi) -> void {
            vsapi->mapSetData(out, "info", ac::core::Processor::info<ac::core::Processor::CPU>(), -1, dtUtf8, maAppend);
#       ifdef AC_CORE_WITH_OPENCL
            vsapi->mapSetData(out, "info", ac::core::Processor::info<ac::core::Processor::OpenCL>(), -1, dtUtf8, maAppend);
#       endif
#       ifdef AC_CORE_WITH_CUDA
            vsapi->mapSetData(out, "info", ac::core::Processor::info<ac::core::Processor::CUDA>(), -1, dtUtf8, maAppend);
#       endif
        }, nullptr, plugin);
}
