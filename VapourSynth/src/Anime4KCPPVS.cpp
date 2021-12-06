#include <thread>

#include <VapourSynth.h>
#include <VSHelper.h>

#include "Anime4KCPP.hpp"
#include "Benchmark.hpp"

enum class GPGPU
{
    OpenCL, CUDA
};

struct Anime4KCPPData 
{
    VSNodeRef* node = nullptr;
    VSVideoInfo vi{};
    bool CNN = true;
    bool GPU = false;
    int pID = 0;
    int dID = 0;
    int OpenCLQueueNum = 4;
    bool OpenCLParallelIO = false;
    Anime4KCPP::ACInitializer initializer;
    Anime4KCPP::Parameters parameters;
    GPGPU GPGPUModel = GPGPU::OpenCL;
};

static void VS_CC Anime4KCPPInit(VSMap* in, VSMap* out, void** instanceData, VSNode* node, VSCore* core, const VSAPI* vsapi)
{
    auto data = reinterpret_cast<Anime4KCPPData*>(*instanceData);

    if (data->GPU)
    {
        switch (data->GPGPUModel)
        {
        case GPGPU::OpenCL:
#ifdef ENABLE_OPENCL
            if (data->CNN)
                data->initializer.pushManager<Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::ACNet>>(
                    data->pID, data->dID,
                    Anime4KCPP::CNNType::Default,
                    data->OpenCLQueueNum,
                    data->OpenCLParallelIO);
            else
                data->initializer.pushManager<Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::Anime4K09>>(
                    data->pID, data->dID,
                    data->OpenCLQueueNum,
                    data->OpenCLParallelIO);
#endif // ENABLE_OPENCL
            break;
        case GPGPU::CUDA:
#ifdef ENABLE_CUDA
            data->initializer.pushManager<Anime4KCPP::Cuda::Manager>(data->dID);
#endif // ENABLE_CUDA
            break;
        }
        if (data->initializer.init() != data->initializer.size())
        {
            std::ostringstream oss("Unable to initialize:\n", std::ios_base::ate);
            for (auto& error : data->initializer.failure())
                oss << "  " << error;
            oss << '\n';

            vsapi->setError(out, oss.str().c_str());
            vsapi->freeNode(data->node);
            std::default_delete<Anime4KCPPData>{}(data);
            return;
        }
    }

    vsapi->setVideoInfo(&data->vi, 1, node);
}

template<typename T>
static const VSFrameRef* VS_CC Anime4KCPPGetFrame(int n, int activationReason, void** instanceData, void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi)
{
    auto data = reinterpret_cast<Anime4KCPPData*>(*instanceData);

    if (activationReason == arInitial)
        vsapi->requestFrameFilter(n, data->node, frameCtx);
    else if (activationReason == arAllFramesReady)
    {
        const VSFrameRef* src = vsapi->getFrameFilter(n, data->node, frameCtx);

        int srcH = vsapi->getFrameHeight(src, 0);
        int srcW = vsapi->getFrameWidth(src, 0);

        int srcSrtide = vsapi->getStride(src, 0);

        VSFrameRef* dst = vsapi->newVideoFrame(data->vi.format, data->vi.width, data->vi.height, src, core);

        int dstSrtide = vsapi->getStride(dst, 0);

        T* srcR = const_cast<T*>(reinterpret_cast<const T*>(vsapi->getReadPtr(src, 0)));
        T* srcG = const_cast<T*>(reinterpret_cast<const T*>(vsapi->getReadPtr(src, 1)));
        T* srcB = const_cast<T*>(reinterpret_cast<const T*>(vsapi->getReadPtr(src, 2)));

        std::uint8_t* dstR = vsapi->getWritePtr(dst, 0);
        std::uint8_t* dstG = vsapi->getWritePtr(dst, 1);
        std::uint8_t* dstB = vsapi->getWritePtr(dst, 2);

        std::unique_ptr<Anime4KCPP::AC> ac;

        if (data->GPU)
        {
            switch (data->GPGPUModel)
            {
            case GPGPU::OpenCL:
#ifdef ENABLE_OPENCL
                if (data->CNN)
                    ac = Anime4KCPP::ACCreator::createUP(data->parameters, Anime4KCPP::Processor::Type::OpenCL_ACNet);
                else
                    ac = Anime4KCPP::ACCreator::createUP(data->parameters, Anime4KCPP::Processor::Type::OpenCL_Anime4K09);
#endif
                break;
            case GPGPU::CUDA:
#ifdef ENABLE_CUDA
                if (data->CNN)
                    ac = Anime4KCPP::ACCreator::createUP(data->parameters, Anime4KCPP::Processor::Type::Cuda_ACNet);
                else
                    ac = Anime4KCPP::ACCreator::createUP(data->parameters, Anime4KCPP::Processor::Type::Cuda_Anime4K09);
#endif
                break;
            }
        }
        else
        {
            if (data->CNN)
                ac = Anime4KCPP::ACCreator::createUP(data->parameters, Anime4KCPP::Processor::Type::CPU_ACNet);
            else
                ac = Anime4KCPP::ACCreator::createUP(data->parameters, Anime4KCPP::Processor::Type::CPU_Anime4K09);
        }

        try
        {
            ac->loadImage(srcH, srcW, srcSrtide, srcR, srcG, srcB);
            ac->process();
            ac->saveImage(dstR, dstSrtide, dstG, dstSrtide, dstB, dstSrtide);
        }
        catch (const std::exception& e)
        {
            vsapi->freeFrame(src);
            vsapi->freeFrame(dst);
            vsapi->setFilterError(e.what(), frameCtx);
            return nullptr;
        }

        vsapi->freeFrame(src);

        return dst;
    }
    return nullptr;
}

template<typename T>
static const VSFrameRef* VS_CC Anime4KCPPGetFrameYUV(int n, int activationReason, void** instanceData, void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi)
{
    auto data = reinterpret_cast<Anime4KCPPData*>(*instanceData);

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

        int dstSrtideY = vsapi->getStride(dst, 0);
        int dstSrtideU = vsapi->getStride(dst, 1);
        int dstSrtideV = vsapi->getStride(dst, 2);

        T* srcY = const_cast<T*>(reinterpret_cast<const T*>(vsapi->getReadPtr(src, 0)));
        T* srcU = const_cast<T*>(reinterpret_cast<const T*>(vsapi->getReadPtr(src, 1)));
        T* srcV = const_cast<T*>(reinterpret_cast<const T*>(vsapi->getReadPtr(src, 2)));

        std::uint8_t* dstY = vsapi->getWritePtr(dst, 0);
        std::uint8_t* dstU = vsapi->getWritePtr(dst, 1);
        std::uint8_t* dstV = vsapi->getWritePtr(dst, 2);

        std::unique_ptr<Anime4KCPP::AC> ac;

        if (data->GPU)
        {
            switch (data->GPGPUModel)
            {
            case GPGPU::OpenCL:
#ifdef ENABLE_OPENCL
                if (data->CNN)
                    ac = Anime4KCPP::ACCreator::createUP(data->parameters, Anime4KCPP::Processor::Type::OpenCL_ACNet);
                else
                    ac = Anime4KCPP::ACCreator::createUP(data->parameters, Anime4KCPP::Processor::Type::OpenCL_Anime4K09);
#endif
                break;
            case GPGPU::CUDA:
#ifdef ENABLE_CUDA
                if (data->CNN)
                    ac = Anime4KCPP::ACCreator::createUP(data->parameters, Anime4KCPP::Processor::Type::Cuda_ACNet);
                else
                    ac = Anime4KCPP::ACCreator::createUP(data->parameters, Anime4KCPP::Processor::Type::Cuda_Anime4K09);
#endif
                break;
            }
        }
        else
        {
            if (data->CNN)
                ac = Anime4KCPP::ACCreator::createUP(data->parameters, Anime4KCPP::Processor::Type::CPU_ACNet);
            else
                ac = Anime4KCPP::ACCreator::createUP(data->parameters, Anime4KCPP::Processor::Type::CPU_Anime4K09);
        }

        try
        {
            ac->loadImage(
                srcHY, srcWY, srcSrtideY, srcY,
                srcHU, srcWU, srcSrtideU, srcU,
                srcHV, srcWV, srcSrtideV, srcV);
            ac->process();
            ac->saveImage(dstY, dstSrtideY, dstU, dstSrtideU, dstV, dstSrtideV);
        }
        catch (const std::exception& e)
        {
            vsapi->freeFrame(src);
            vsapi->freeFrame(dst);
            vsapi->setFilterError(e.what(), frameCtx);
            return nullptr;
        }

        vsapi->freeFrame(src);

        return dst;
    }
    return nullptr;
}

template<typename T>
static const VSFrameRef* VS_CC Anime4KCPPGetFrameGrayscale(int n, int activationReason, void** instanceData, void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi)
{
    auto data = reinterpret_cast<Anime4KCPPData*>(*instanceData);

    if (activationReason == arInitial)
        vsapi->requestFrameFilter(n, data->node, frameCtx);
    else if (activationReason == arAllFramesReady)
    {
        const VSFrameRef* src = vsapi->getFrameFilter(n, data->node, frameCtx);

        int srcH = vsapi->getFrameHeight(src, 0);
        int srcW = vsapi->getFrameWidth(src, 0);

        int srcSrtide = vsapi->getStride(src, 0);

        VSFrameRef* dst = vsapi->newVideoFrame(data->vi.format, data->vi.width, data->vi.height, src, core);

        int dstSrtide = vsapi->getStride(dst, 0);

        T* srcY = const_cast<T*>(reinterpret_cast<const T*>(vsapi->getReadPtr(src, 0)));

        std::uint8_t* dstY = vsapi->getWritePtr(dst, 0);

        std::unique_ptr<Anime4KCPP::AC> ac;

        if (data->GPU)
        {
            switch (data->GPGPUModel)
            {
            case GPGPU::OpenCL:
#ifdef ENABLE_OPENCL
                if (data->CNN)
                    ac = Anime4KCPP::ACCreator::createUP(data->parameters, Anime4KCPP::Processor::Type::OpenCL_ACNet);
                else
                    ac = Anime4KCPP::ACCreator::createUP(data->parameters, Anime4KCPP::Processor::Type::OpenCL_Anime4K09);
#endif
                break;
            case GPGPU::CUDA:
#ifdef ENABLE_CUDA
                if (data->CNN)
                    ac = Anime4KCPP::ACCreator::createUP(data->parameters, Anime4KCPP::Processor::Type::Cuda_ACNet);
                else
                    ac = Anime4KCPP::ACCreator::createUP(data->parameters, Anime4KCPP::Processor::Type::Cuda_Anime4K09);
#endif
                break;
            }
        }
        else
        {
            if (data->CNN)
                ac = Anime4KCPP::ACCreator::createUP(data->parameters, Anime4KCPP::Processor::Type::CPU_ACNet);
            else
                ac = Anime4KCPP::ACCreator::createUP(data->parameters, Anime4KCPP::Processor::Type::CPU_Anime4K09);
        }

        try
        {
            ac->loadImage(srcH, srcW, srcSrtide, srcY, false, false, true);
            ac->process();
            ac->saveImage(dstY, dstSrtide);
        }
        catch (const std::exception& e)
        {
            vsapi->freeFrame(src);
            vsapi->freeFrame(dst);
            vsapi->setFilterError(e.what(), frameCtx);
            return nullptr;
        }

        vsapi->freeFrame(src);

        return dst;
    }
    return nullptr;
}

static void VS_CC Anime4KCPPFree(void* instanceData, VSCore* core, const VSAPI* vsapi)
{
    std::unique_ptr<Anime4KCPPData> data{ reinterpret_cast<Anime4KCPPData*>(instanceData) };
    vsapi->freeNode(data->node);
}

static void VS_CC Anime4KCPPCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi)
{
    auto data = std::make_unique<Anime4KCPPData>();
    int err = 0;

    data->node = vsapi->propGetNode(in, "src", 0, 0);
    data->vi = *vsapi->getVideoInfo(data->node);

    if (!isConstantFormat(&data->vi) ||
        (data->vi.format->colorFamily != cmYUV && data->vi.format->colorFamily != cmRGB && data->vi.format->colorFamily != cmGray) ||
        (data->vi.format->bitsPerSample != 8 && data->vi.format->bitsPerSample != 16 && data->vi.format->bitsPerSample != 32))
    {
        vsapi->setError(out,
            "Anime4KCPP: supported data type: RGB, YUV and Grayscale, depth with 8 or 16bit integer or 32bit float)");
        vsapi->freeNode(data->node);
        return;
    }

    data->parameters.passes = static_cast<int>(vsapi->propGetInt(in, "passes", 0, &err));
    if (err)
        data->parameters.passes = 2;

    data->parameters.pushColorCount = static_cast<int>(vsapi->propGetInt(in, "pushColorCount", 0, &err));
    if (err)
        data->parameters.pushColorCount = 2;

    data->parameters.strengthColor = vsapi->propGetFloat(in, "strengthColor", 0, &err);
    if (err)
        data->parameters.strengthColor = 0.3;
    else if (data->parameters.strengthColor < 0.0 || data->parameters.strengthColor > 1.0)
    {
        vsapi->setError(out, "Anime4KCPP: strengthColor must range from 0 to 1");
        vsapi->freeNode(data->node);
        return;
    }

    data->parameters.strengthGradient = vsapi->propGetFloat(in, "strengthGradient", 0, &err);
    if (err)
        data->parameters.strengthGradient = 1.0;
    else if (data->parameters.strengthGradient < 0.0 || data->parameters.strengthGradient > 1.0)
    {
        vsapi->setError(out, "Anime4KCPP: strengthGradient must range from 0 to 1");
        vsapi->freeNode(data->node);
        return;
    }

    data->parameters.zoomFactor = vsapi->propGetFloat(in, "zoomFactor", 0, &err);
    if (err)
        data->parameters.zoomFactor = 2.0;
    else if (data->parameters.zoomFactor < 1.0)
    {
        vsapi->setError(out, "Anime4KCPP: zoomFactor must >= 1.0");
        vsapi->freeNode(data->node);
        return;
    }

    data->CNN = vsapi->propGetInt(in, "ACNet", 0, &err);
    if (err)
        data->CNN = true;

    if (data->vi.format->id != pfGray16 &&
        data->vi.format->id != pfGray8 &&
        data->vi.format->id != pfGrayS &&
        data->vi.format->id != pfRGB24 &&
        data->vi.format->id != pfRGB48 &&
        data->vi.format->id != pfRGBS &&
        data->vi.format->id != pfYUV444P8 &&
        data->vi.format->id != pfYUV444P16 &&
        data->vi.format->id != pfYUV444PS &&
        data->CNN == false)
    {
        vsapi->setError(out, "Anime4KCPP: RGB or YUV444 or Grayscale only for Anime4K09");
        vsapi->freeNode(data->node);
        return;
    }

    data->parameters.HDN = vsapi->propGetInt(in, "HDN", 0, &err);
    if (err)
        data->parameters.HDN = false;

    data->parameters.HDNLevel = static_cast<int>(vsapi->propGetInt(in, "HDNLevel", 0, &err));
    if (err)
        data->parameters.HDNLevel = 1;
    else if (data->parameters.HDNLevel < 1 || data->parameters.HDNLevel > 3)
    {
        vsapi->setError(out, "Anime4KCPP: HDNLevel must range from 1 to 3");
        vsapi->freeNode(data->node);
        return;
    }

    data->GPU = vsapi->propGetInt(in, "GPUMode", 0, &err);
    if (err)
        data->GPU = false;

    std::string GPGPUModel;
    const char* tmpStr = vsapi->propGetData(in, "GPGPUModel", 0, &err);
    if (err)
#ifdef ENABLE_OPENCL
        GPGPUModel = "opencl";
#elif defined(ENABLE_CUDA)
        GPGPUModel = "cuda";
#else
        GPGPUModel = "cpu";
#endif

    else
        GPGPUModel = tmpStr;
    std::transform(GPGPUModel.begin(), GPGPUModel.end(), GPGPUModel.begin(), ::tolower);

    if (GPGPUModel == "opencl")
    {
#ifndef ENABLE_OPENCL
        vsapi->setError(out, "Anime4KCPP: OpenCL is unsupported");
        vsapi->freeNode(data->node);
        return;
#endif // !ENABLE_OPENCL
        data->GPGPUModel = GPGPU::OpenCL;
    }
    else if (GPGPUModel == "cuda")
    {

#ifndef ENABLE_CUDA
        vsapi->setError(out, "Anime4KCPP: CUDA is unsupported");
        vsapi->freeNode(data->node);
        return;
#endif // !ENABLE_CUDA
        data->GPGPUModel = GPGPU::CUDA;
    }
    else if (GPGPUModel == "cpu")
    {
        if (data->GPU)
        {
            vsapi->setError(out, "Anime4KCPP: OpenCL or CUDA is unsupported");
            vsapi->freeNode(data->node);
            return;
        }
    }
    else
    {
        vsapi->setError(out, R"(Anime4KCPP: GPGPUModel must be "cuda" or "opencl")");
        vsapi->freeNode(data->node);
        return;
    }

    data->pID = static_cast<int>(vsapi->propGetInt(in, "platformID", 0, &err));
    if (err || !data->GPU)
        data->pID = 0;

    data->dID = static_cast<int>(vsapi->propGetInt(in, "deviceID", 0, &err));
    if (err || !data->GPU)
        data->dID = 0;

    data->OpenCLQueueNum = static_cast<int>(vsapi->propGetInt(in, "OpenCLQueueNum", 0, &err));
    if (err)
    {
        int currentThreads = static_cast<int>(std::thread::hardware_concurrency());
        data->OpenCLQueueNum = (currentThreads < 1) ? 1 : currentThreads;
    }
    else if (data->OpenCLQueueNum < 1)
    {
        vsapi->setError(out, "Anime4KCPP: OpenCLQueueNum must >= 1");
        vsapi->freeNode(data->node);
        return;
    }

    data->OpenCLParallelIO = vsapi->propGetInt(in, "OpenCLParallelIO", 0, &err);
    if (err)
        data->OpenCLParallelIO = false;

    if (data->GPU)
    {
        std::string info;
        switch (data->GPGPUModel)
        {
        case GPGPU::OpenCL:
#ifdef ENABLE_OPENCL
        {
            Anime4KCPP::OpenCL::GPUList list = Anime4KCPP::OpenCL::listGPUs();
            if (data->pID < 0 || data->dID < 0 ||
                data->pID >= list.platforms || data->dID >= list[data->pID])
            {
                std::ostringstream err;
                err <<
                    "Anime4KCPP: Platform ID or device ID index out of range\n"
                    "Run core.anim4kcpp.listGPUs() for available platforms and devices\n"
                    "Your input is: \n"
                    "    platform ID: " << data->pID << "\n"
                    "    device ID: " << data->dID << '\n';
                vsapi->setError(out, err.str().c_str());
                vsapi->freeNode(data->node);
                return;
            }
            Anime4KCPP::OpenCL::GPUInfo ret =
                Anime4KCPP::OpenCL::checkGPUSupport(data->pID, data->dID);
            if (!ret)
            {
                std::ostringstream err;
                err <<
                    "Anime4KCPP: The current device is unavailable\n"
                    "Your input is: \n"
                    "    platform ID: " << data->pID << "\n"
                    "    device ID: " << data->dID << "\n"
                    "Error: \n"
                    "    " + ret() << '\n';
                vsapi->setError(out, err.str().c_str());
                vsapi->freeNode(data->node);
                return;
            }
            info = ret();
        }
#endif
        break;
        case GPGPU::CUDA:
#ifdef ENABLE_CUDA
        {
            Anime4KCPP::Cuda::GPUList list = Anime4KCPP::Cuda::listGPUs();
            if (data->dID < 0 || data->dID >= list.devices)
            {
                std::ostringstream err;
                err <<
                    "Anime4KCPP: Device ID index out of range\n"
                    "Run core.anim4kcpp.listGPUs() for available CUDA devices\n"
                    "Your input is: \n"
                    "    device ID: " << data->dID << '\n';
                vsapi->setError(out, err.str().c_str());
                vsapi->freeNode(data->node);
                return;
            }
            Anime4KCPP::Cuda::GPUInfo ret =
                Anime4KCPP::Cuda::checkGPUSupport(data->dID);
            if (!ret)
            {
                std::ostringstream err;
                err <<
                    "Anime4KCPP: The current device is unavailable\n"
                    "Your input is: \n"
                    "    device ID: " << data->dID << "\n"
                    "Error: \n"
                    "    " + ret() << '\n';
                vsapi->setError(out, err.str().c_str());
                vsapi->freeNode(data->node);
                return;
            }
            info = ret();
        }
#endif
        break;
        }
        vsapi->logMessage(mtDebug, ("Current GPU information: \n" + info).c_str());
    }

    if (data->parameters.zoomFactor != 1.0)
    {
        data->vi.width = static_cast<int>(std::round(data->vi.width * data->parameters.zoomFactor));
        data->vi.height = static_cast<int>(std::round(data->vi.height * data->parameters.zoomFactor));
    }

    if (data->vi.format->colorFamily == cmYUV)
    {
        if (data->vi.format->sampleType == stFloat)
            vsapi->createFilter(in, out, "Anime4KCPP", Anime4KCPPInit, Anime4KCPPGetFrameYUV<float>, Anime4KCPPFree, fmParallel, 0, data.release(), core);
        else
            if (data->vi.format->bitsPerSample == 8)
                vsapi->createFilter(in, out, "Anime4KCPP", Anime4KCPPInit, Anime4KCPPGetFrameYUV<std::uint8_t>, Anime4KCPPFree, fmParallel, 0, data.release(), core);
            else
                vsapi->createFilter(in, out, "Anime4KCPP", Anime4KCPPInit, Anime4KCPPGetFrameYUV<std::uint16_t>, Anime4KCPPFree, fmParallel, 0, data.release(), core);
    }
    else if (data->vi.format->colorFamily == cmGray)
    {
        if (data->vi.format->sampleType == stFloat)
            vsapi->createFilter(in, out, "Anime4KCPP", Anime4KCPPInit, Anime4KCPPGetFrameGrayscale<float>, Anime4KCPPFree, fmParallel, 0, data.release(), core);
        else
            if (data->vi.format->bitsPerSample == 8)
                vsapi->createFilter(in, out, "Anime4KCPP", Anime4KCPPInit, Anime4KCPPGetFrameGrayscale<std::uint8_t>, Anime4KCPPFree, fmParallel, 0, data.release(), core);
            else
                vsapi->createFilter(in, out, "Anime4KCPP", Anime4KCPPInit, Anime4KCPPGetFrameGrayscale<std::uint16_t>, Anime4KCPPFree, fmParallel, 0, data.release(), core);
    }
    else
    {
        if (data->vi.format->sampleType == stFloat)
            vsapi->createFilter(in, out, "Anime4KCPP", Anime4KCPPInit, Anime4KCPPGetFrame<float>, Anime4KCPPFree, fmParallel, 0, data.release(), core);
        else
            if (data->vi.format->bitsPerSample == 8)
                vsapi->createFilter(in, out, "Anime4KCPP", Anime4KCPPInit, Anime4KCPPGetFrame<std::uint8_t>, Anime4KCPPFree, fmParallel, 0, data.release(), core);
            else
                vsapi->createFilter(in, out, "Anime4KCPP", Anime4KCPPInit, Anime4KCPPGetFrame<std::uint16_t>, Anime4KCPPFree, fmParallel, 0, data.release(), core);
    }
}

static void VS_CC Anime4KCPPListGPUs(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi)
{
    int err;
    std::string GPGPUModel;
    const char* tmpStr = vsapi->propGetData(in, "GPGPUModel", 0, &err);
    if (err)
        GPGPUModel = "opencl";
    else
        GPGPUModel = tmpStr;
    std::transform(GPGPUModel.begin(), GPGPUModel.end(), GPGPUModel.begin(), ::tolower);

#ifdef ENABLE_OPENCL
    if (GPGPUModel == "opencl")
        vsapi->logMessage(mtDebug, Anime4KCPP::OpenCL::listGPUs()().c_str());
    else
#endif // ENABLE_OPENCL
#ifdef ENABLE_CUDA
        if (GPGPUModel == "cuda")
            vsapi->logMessage(mtDebug, Anime4KCPP::Cuda::listGPUs()().c_str());
        else
#endif // ENABLE_CUDA
            vsapi->logMessage(mtWarning, "unkonwn GPGPUModel");
}

static void VS_CC Anime4KCPPBenchmark(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi)
{
    int err = 0;
    int pID = static_cast<int>(vsapi->propGetInt(in, "platformID", 0, &err));
    if (err || !pID)
        pID = 0;

    int dID = static_cast<int>(vsapi->propGetInt(in, "deviceID", 0, &err));
    if (err || !dID)
        dID = 0;

    double CPUScoreDVD = Anime4KCPP::benchmark<Anime4KCPP::CPU::ACNet, 720, 480>();
    double CPUScoreHD = Anime4KCPP::benchmark<Anime4KCPP::CPU::ACNet, 1280, 720>();
    double CPUScoreFHD = Anime4KCPP::benchmark<Anime4KCPP::CPU::ACNet, 1920, 1080>();

#ifdef ENABLE_OPENCL
    double OpenCLScoreDVD = Anime4KCPP::benchmark<Anime4KCPP::OpenCL::ACNet, 720, 480>(pID, dID, Anime4KCPP::CNNType::ACNetHDNL0);
    double OpenCLScoreHD = Anime4KCPP::benchmark<Anime4KCPP::OpenCL::ACNet, 1280, 720>(pID, dID, Anime4KCPP::CNNType::ACNetHDNL0);
    double OpenCLScoreFHD = Anime4KCPP::benchmark<Anime4KCPP::OpenCL::ACNet, 1920, 1080>(pID, dID, Anime4KCPP::CNNType::ACNetHDNL0);
#endif 

#ifdef ENABLE_CUDA
    double CudaScoreDVD = Anime4KCPP::benchmark<Anime4KCPP::Cuda::ACNet, 720, 480>(dID);
    double CudaScoreHD = Anime4KCPP::benchmark<Anime4KCPP::Cuda::ACNet, 1280, 720>(dID);
    double CudaScoreFHD = Anime4KCPP::benchmark<Anime4KCPP::Cuda::ACNet, 1920, 1080>(dID);
#endif 

    std::ostringstream oss;

    oss << "Benchmark test under 8-bit integer input and serial processing...\n\n";

    oss
        << "CPU score:\n"
        << " DVD(480P->960P): " << CPUScoreDVD << " FPS\n"
        << " HD(720P->1440P): " << CPUScoreHD << " FPS\n"
        << " FHD(1080P->2160P): " << CPUScoreFHD << " FPS\n\n";

#ifdef ENABLE_OPENCL
    oss
        << "OpenCL score: (pID = " << pID << ", dID = " << dID << ")\n"
        << " DVD(480P->960P): " << OpenCLScoreDVD << " FPS\n"
        << " HD(720P->1440P): " << OpenCLScoreHD << " FPS\n"
        << " FHD(1080P->2160P): " << OpenCLScoreFHD << " FPS\n\n";
#endif 

#ifdef ENABLE_CUDA
    oss
        << "CUDA score: (dID = " << dID << ")\n"
        << " DVD(480P->960P): " << CudaScoreDVD << " FPS\n"
        << " HD(720P->1440P): " << CudaScoreHD << " FPS\n"
        << " FHD(1080P->2160P): " << CudaScoreFHD << " FPS\n\n";
#endif 

    vsapi->logMessage(mtDebug, oss.str().c_str());
}

static void VS_CC Anime4KCPPInfo(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi)
{
    std::ostringstream oss;
    oss << "Anime4KCPP VapourSynth Plugin" << '\n'
        << '\n'
        << "Anime4KCPP core information:\n"
        << "  Version: " << Anime4KCPP::CoreInfo::version() << '\n'
        << "  Parallel library: " << ANIME4KCPP_CORE_PARALLEL_LIBRARY << '\n'
        << "  Compiler: " << ANIME4KCPP_CORE_COMPILER << '\n'
        << "  Processors: " << Anime4KCPP::CoreInfo::supportedProcessors() << '\n'
        << "  CPU Optimization: " << Anime4KCPP::CoreInfo::CPUOptimizationMode() << '\n'
        << '\n'
        << "GitHub: https://github.com/TianZerL/Anime4KCPP" << std::endl;

    vsapi->logMessage(mtDebug, oss.str().c_str());
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
        "zoomFactor:float:opt;"
        "ACNet:int:opt;"
        "GPUMode:int:opt;"
        "GPGPUModel:data:opt;"
        "HDN:int:opt;"
        "HDNLevel:int:opt;"
        "platformID:int:opt;"
        "deviceID:int:opt;"
        "OpenCLQueueNum:int:opt;"
        "OpenCLParallelIO:int:opt;",
        Anime4KCPPCreate, nullptr, plugin);

    registerFunc("listGPUs", "GPGPUModel:data:opt", Anime4KCPPListGPUs, nullptr, plugin);

    registerFunc("info", "", Anime4KCPPInfo, nullptr, plugin);

    registerFunc("benchmark",
        "platformID:int:opt;"
        "deviceID:int:opt;",
        Anime4KCPPBenchmark, nullptr, plugin);
}
