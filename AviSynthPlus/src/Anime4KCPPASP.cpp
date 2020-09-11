#include <avisynth.h>
#include "Anime4KCPP.h"

#ifdef _MSC_VER
#define AC_STDCALL __stdcall
#define AC_CDECL __cdecl
#define AC_DLL __declspec(dllexport)
#else
#define AC_STDCALL __attribute__((__stdcall__))
#define AC_CDECL __attribute__((__cdecl__))
#define AC_DLL
#endif

enum AC_Parameters
{
    AC_passes = 1,
    AC_pushColorCount = 2,
    AC_strengthColor = 3,
    AC_strengthGradient = 4,
    AC_zoomFactor = 5,
    AC_ACNet = 6,
    AC_GPUMode = 7,
    AC_HDN = 8,
    AC_HDNLevel = 9,
    AC_platformID = 10,
    AC_deviceID = 11
};

class Anime4KCPPF : public GenericVideoFilter
{
public:
    Anime4KCPPF(
        PClip _child,
        Anime4KCPP::Parameters& inputs,
        bool CNN,
        bool GPUMode,
        unsigned int pID,
        unsigned int dID,
        IScriptEnvironment* env
    );
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
private:
    Anime4KCPP::Parameters parameters;
    Anime4KCPP::Anime4KCreator anime4KCreator;
    bool GPUMode;
    bool CNN;
};

Anime4KCPPF::Anime4KCPPF(
    PClip _child,
    Anime4KCPP::Parameters& inputs,
    bool CNN,
    bool GPUMode,
    unsigned int pID,
    unsigned int dID,
    IScriptEnvironment* env
) :
    GenericVideoFilter(_child),
    parameters(inputs),
    anime4KCreator(GPUMode, CNN, pID, dID),
    GPUMode(GPUMode),
    CNN(CNN)
{
    if (!vi.IsRGB24() && (!vi.IsYUV() || vi.BitsPerComponent() != 8 || !vi.IsPlanar()))
    {
        env->ThrowError("Anime4KCPP: RGB24 or planar YUV 8bit data only");
    }

    if (!vi.IsRGB24() && !vi.Is444() && !CNN)
    {
        env->ThrowError("Anime4KCPP: RGB24 or YUV444P8 is needed for Anime4K09");
    }

    vi.height *= inputs.zoomFactor;
    vi.width *= inputs.zoomFactor;
}

PVideoFrame AC_STDCALL Anime4KCPPF::GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrameP(vi, &src);

    if (vi.IsYUV())
    {
        size_t srcPitchY = src->GetPitch(PLANAR_Y);
        size_t dstPitchY = dst->GetPitch(PLANAR_Y);
        size_t srcPitchU = src->GetPitch(PLANAR_U);
        size_t dstPitchU = dst->GetPitch(PLANAR_U);
        size_t srcPitchV = src->GetPitch(PLANAR_V);
        size_t dstPitchV = dst->GetPitch(PLANAR_V);

        size_t srcHY = src->GetHeight(PLANAR_Y);
        size_t srcLY = src->GetRowSize(PLANAR_Y);
        size_t srcHU = src->GetHeight(PLANAR_U);
        size_t srcLU = src->GetRowSize(PLANAR_U);
        size_t srcHV = src->GetHeight(PLANAR_V);
        size_t srcLV = src->GetRowSize(PLANAR_V);

        size_t dstHY = dst->GetHeight(PLANAR_Y);
        size_t dstLY = dst->GetRowSize(PLANAR_Y);
        size_t dstHU = dst->GetHeight(PLANAR_U);
        size_t dstLU = dst->GetRowSize(PLANAR_U);
        size_t dstHV = dst->GetHeight(PLANAR_V);
        size_t dstLV = dst->GetRowSize(PLANAR_V);

        const unsigned char* srcpY = src->GetReadPtr(PLANAR_Y);
        const unsigned char* srcpU = src->GetReadPtr(PLANAR_U);
        const unsigned char* srcpV = src->GetReadPtr(PLANAR_V);

        unsigned char* dstpY = dst->GetWritePtr(PLANAR_Y);
        unsigned char* dstpU = dst->GetWritePtr(PLANAR_U);
        unsigned char* dstpV = dst->GetWritePtr(PLANAR_V);

        unsigned char* srcDataY = new unsigned char[srcHY * srcLY];
        unsigned char* srcDataU = new unsigned char[srcHU * srcLU];
        unsigned char* srcDataV = new unsigned char[srcHV * srcLV];

        cv::Mat dstDataY;
        cv::Mat dstDataU;
        cv::Mat dstDataV;

        for (size_t y = 0; y < srcHY; y++)
        {
            memcpy(srcDataY + y * srcLY, srcpY, srcLY);
            srcpY += srcPitchY;
            if (y < srcHU)
            {
                memcpy(srcDataU + y * srcLU, srcpU, srcLU);
                srcpU += srcPitchU;
            }
            if (y < srcHV)
            {
                memcpy(srcDataV + y * srcLV, srcpV, srcLV);
                srcpV += srcPitchV;
            }
        }

        Anime4KCPP::Anime4K* anime4K;

        if (CNN)
            if (GPUMode)
                anime4K = anime4KCreator.create(parameters, Anime4KCPP::ProcessorType::GPUCNN);
            else
                anime4K = anime4KCreator.create(parameters, Anime4KCPP::ProcessorType::CPUCNN);
        else
            if (GPUMode)
                anime4K = anime4KCreator.create(parameters, Anime4KCPP::ProcessorType::GPU);
            else
                anime4K = anime4KCreator.create(parameters, Anime4KCPP::ProcessorType::CPU);

        anime4K->loadImage(srcHY, srcLY, srcDataY, srcHU, srcLU, srcDataU, srcHV, srcLV, srcDataV);
        anime4K->process();
        anime4K->saveImage(dstDataY, dstDataU, dstDataV);

        for (size_t y = 0; y < dstHY; y++)
        {
            memcpy(dstpY, dstDataY.data + y * dstLY, dstLY);
            dstpY += dstPitchY;
            if (y < dstHU)
            {
                memcpy(dstpU, dstDataU.data + y * dstLU, dstLU);
                dstpU += dstPitchU;
            }
            if (y < dstHV)
            {
                memcpy(dstpV, dstDataV.data + y * dstLV, dstLV);
                dstpV += dstPitchV;
            }
        }

        anime4KCreator.release(anime4K);
        delete[] srcDataY;
        delete[] srcDataU;
        delete[] srcDataV;

        return dst;
    }
    else
    {
        size_t srcPitch = src->GetPitch();
        size_t dstPitch = dst->GetPitch();

        size_t srcH = src->GetHeight();
        size_t srcL = src->GetRowSize();
        size_t dstH = dst->GetHeight();
        size_t dstL = dst->GetRowSize();

        const unsigned char* srcp = src->GetReadPtr();
        unsigned char* dstp = dst->GetWritePtr();

        unsigned char* srcData = new unsigned char[srcH * srcL];

        cv::Mat dstData;

        for (size_t y = 0; y < srcH; y++)
        {
            memcpy(srcData + y * srcL, srcp, srcL);
            srcp += srcPitch;
        }

        Anime4KCPP::Anime4K* anime4K;

        if (CNN)
            if (GPUMode)
                anime4K = anime4KCreator.create(parameters, Anime4KCPP::ProcessorType::GPUCNN);
            else
                anime4K = anime4KCreator.create(parameters, Anime4KCPP::ProcessorType::CPUCNN);
        else
            if (GPUMode)
                anime4K = anime4KCreator.create(parameters, Anime4KCPP::ProcessorType::GPU);
            else
                anime4K = anime4KCreator.create(parameters, Anime4KCPP::ProcessorType::CPU);

        anime4K->loadImage(srcH, srcL / 3, srcData);
        anime4K->process();
        anime4K->saveImage(dstData);

        for (size_t y = 0; y < dstH; y++)
        {
            memcpy(dstp, dstData.data + y * dstL, dstL);
            dstp += dstPitch;
        }

        anime4KCreator.release(anime4K);
        delete[] srcData;

        return dst;
    }

}

AVSValue AC_CDECL createAnime4KCPP(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    Anime4KCPP::Parameters inputs(
        args[AC_passes].AsInt(),
        args[AC_pushColorCount].AsInt(),
        args[AC_strengthColor].AsFloat(),
        args[AC_strengthGradient].AsFloat(),
        args[AC_zoomFactor].AsInt(),
        false, false, false, false, 4, 40, std::thread::hardware_concurrency(),
        args[AC_HDN].AsBool(),
        args[AC_HDNLevel].AsInt()
    );

    bool CNN = args[AC_ACNet].AsBool();
    bool GPUMode = args[AC_GPUMode].AsBool();
    unsigned int pID = args[AC_platformID].AsInt();
    unsigned int dID = args[AC_deviceID].AsInt();


    if (!args[AC_passes].Defined())
        inputs.passes = 2;
    if (!args[AC_pushColorCount].Defined())
        inputs.pushColorCount = 2;
    if (!args[AC_strengthColor].Defined())
        inputs.strengthColor = 0.3;
    if (!args[AC_strengthGradient].Defined())
        inputs.strengthGradient = 1.0;
    if (!args[AC_zoomFactor].Defined())
        inputs.zoomFactor = 1.0;
    if (!args[AC_ACNet].Defined())
        CNN = false;
    if (!args[AC_GPUMode].Defined())
        GPUMode = false;
    if (!args[AC_platformID].Defined())
        pID = 0;
    if (!args[AC_deviceID].Defined())
        dID = 0;


    if (inputs.strengthColor < 0.0 || inputs.strengthColor > 1.0)
        env->ThrowError("Anime4KCPP: strengthColor must range from 0 to 1!");

    if (inputs.strengthGradient < 0.0 || inputs.strengthGradient > 1.0)
        env->ThrowError("Anime4KCPP: strengthGradient must range from 0 to 1!");

    if (inputs.zoomFactor < 1.0)
        env->ThrowError("Anime4KCPP: zoomFactor must be an integer which >= 1!");

    if (inputs.HDNLevel < 1 || inputs.HDNLevel > 3)
        env->ThrowError("Anime4KCPP: HDNLevel must range from 1 to 3!");

    if (GPUMode)
    {
        std::pair<bool, std::string> ret =
            Anime4KCPP::Anime4KGPU::checkGPUSupport(pID, dID);
        if (!ret.first)
        {
            std::ostringstream err;
            err << "Anime4KCPP: The current device is unavailable" << std::endl
                << "Your input is: " << std::endl
                << "    platform ID: " << pID << std::endl
                << "    device ID: " << dID << std::endl
                << "Error: " << std::endl
                << "    " + ret.second << std::endl;
            env->ThrowError(err.str().c_str());
        }
    }

    return new Anime4KCPPF(
        args[0].AsClip(),
        inputs,
        CNN,
        GPUMode,
        pID,
        dID,
        env
    );
}

AVSValue AC_CDECL listGPUs(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    env->ThrowError(Anime4KCPP::Anime4KGPU::listGPUs().second.c_str());
    return AVSValue();
}

AVSValue AC_CDECL benchmark(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    unsigned int pID = args[AC_platformID].AsInt();
    unsigned int dID = args[AC_deviceID].AsInt();

    std::pair<double, double> ret = Anime4KCPP::benchmark(pID, dID);

    std::ostringstream oss;
    oss << "Benchmark result:" << std::endl
        << "CPU score: " << ret.first << std::endl
        << "GPU score: " << ret.second << std::endl
        << " (pID = " << pID << ", dID = " << dID << ")" << std::endl;

    env->ThrowError(oss.str().c_str());
    return AVSValue();
}

const AVS_Linkage* AVS_linkage = 0;

extern "C" AC_DLL const char* AC_STDCALL AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;

    env->AddFunction("listGPUs", "", listGPUs, 0);

    env->AddFunction("benchmark", 
        "[platformID]i"
        "[deviceID]i", 
        benchmark, 0);

    env->AddFunction("Anime4KCPP",
        "[src]c"
        "[passes]i"
        "[pushColorCount]i"
        "[strengthColor]f"
        "[strengthGradient]f"
        "[zoomFactor]i"
        "[ACNet]b"
        "[GPUMode]b"
        "[HDN]b"
        "[HDNLevel]i"
        "[platformID]i"
        "[deviceID]i",
        createAnime4KCPP, 0);
    return "Anime4KCPP plugin for AviSynthPlus";
}
