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
    AC_platformID = 9,
    AC_deviceID = 10
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
    if (!vi.IsRGB24() && !vi.IsYV24())
    {
        env->ThrowError("Anime4KCPP: RGB24 or YUV444 data only!");
    }

    if (vi.IsYV24() && !CNN)
    {
        env->ThrowError("Anime4KCPP: YUV444 only for ACNet!");
    }

    vi.height *= inputs.zoomFactor;
    vi.width *= inputs.zoomFactor;
}

PVideoFrame AC_STDCALL Anime4KCPPF::GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrameP(vi, &src);

    if (vi.IsYV24())
    {
        int srcPitch = src->GetPitch(PLANAR_Y);
        int dstPitch = dst->GetPitch(PLANAR_Y);

        int srcH = src->GetHeight(PLANAR_Y);
        int srcL = src->GetRowSize(PLANAR_Y);
        int dstH = dst->GetHeight(PLANAR_Y);
        int dstL = dst->GetRowSize(PLANAR_Y);

        const unsigned char* srcpY = src->GetReadPtr(PLANAR_Y);
        const unsigned char* srcpU = src->GetReadPtr(PLANAR_U);
        const unsigned char* srcpV = src->GetReadPtr(PLANAR_V);

        unsigned char* dstpY = dst->GetWritePtr(PLANAR_Y);
        unsigned char* dstpU = dst->GetWritePtr(PLANAR_U);
        unsigned char* dstpV = dst->GetWritePtr(PLANAR_V);

        unsigned char* srcDataY = new unsigned char[static_cast<size_t>(srcH) * static_cast<size_t>(srcL)];
        unsigned char* srcDataU = new unsigned char[static_cast<size_t>(srcH) * static_cast<size_t>(srcL)];
        unsigned char* srcDataV = new unsigned char[static_cast<size_t>(srcH) * static_cast<size_t>(srcL)];

        for (int y = 0; y < srcH; y++)
        {
            memcpy(srcDataY + y * static_cast<size_t>(srcL), srcpY, srcL);
            srcpY += srcPitch;
        }
        for (int y = 0; y < srcH; y++)
        {
            memcpy(srcDataU + y * static_cast<size_t>(srcL), srcpU, srcL);
            srcpU += srcPitch;
        }
        for (int y = 0; y < srcH; y++)
        {
            memcpy(srcDataV + y * static_cast<size_t>(srcL), srcpV, srcL);
            srcpV += srcPitch;
        }

        Anime4KCPP::Anime4K* anime4K;

        if (GPUMode)
            anime4K = anime4KCreator.create(parameters, Anime4KCPP::ProcessorType::GPUCNN);
        else
            anime4K = anime4KCreator.create(parameters, Anime4KCPP::ProcessorType::CPUCNN);

        anime4K->loadImage(srcH, srcL, srcDataY, srcDataU, srcDataV, true);
        anime4K->process();

        size_t dstSize = anime4K->getResultDataPerChannelLength();
        unsigned char* dstDataY = new unsigned char[dstSize];
        unsigned char* dstDataU = new unsigned char[dstSize];
        unsigned char* dstDataV = new unsigned char[dstSize];
        anime4K->saveImage(dstDataY, dstDataU, dstDataV);

        for (int y = 0; y < dstH; y++)
        {
            memcpy(dstpY, dstDataY + y * static_cast<size_t>(dstL), dstL);
            dstpY += dstPitch;
        }
        for (int y = 0; y < dstH; y++)
        {
            memcpy(dstpU, dstDataU + y * static_cast<size_t>(dstL), dstL);
            dstpU += dstPitch;
        }
        for (int y = 0; y < dstH; y++)
        {
            memcpy(dstpV, dstDataV + y * static_cast<size_t>(dstL), dstL);
            dstpV += dstPitch;
        }

        anime4KCreator.release(anime4K);
        delete[] dstDataY;
        delete[] dstDataU;
        delete[] dstDataV;
        delete[] srcDataY;
        delete[] srcDataU;
        delete[] srcDataV;

        return dst;
    }
    else
    {
        int srcPitch = src->GetPitch();
        int dstPitch = dst->GetPitch();

        int srcH = src->GetHeight();
        int srcL = src->GetRowSize();
        int dstH = dst->GetHeight();
        int dstL = dst->GetRowSize();

        const unsigned char* srcp = src->GetReadPtr();
        unsigned char* dstp = dst->GetWritePtr();

        unsigned char* srcData = new unsigned char[static_cast<size_t>(srcH) * static_cast<size_t>(srcL)];

        for (int y = 0; y < srcH; y++)
        {
            memcpy(srcData + y * static_cast<size_t>(srcL), srcp, srcL);
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

        unsigned char* dstData = new unsigned char[anime4K->getResultDataLength()];
        anime4K->saveImage(dstData);

        for (int y = 0; y < dstH; y++)
        {
            memcpy(dstp, dstData + y * static_cast<size_t>(dstL), dstL);
            dstp += dstPitch;
        }

        anime4KCreator.release(anime4K);
        delete[] dstData;
        delete[] srcData;

        return dst;
    }
    
}

AVSValue AC_CDECL createAnime4KCPP(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    Anime4KCPP::Parameters inputs(
        args[AC_passes].AsInt(),
        args[AC_pushColorCount].AsInt(),
        args[AC_strengthColor].AsFloatf(),
        args[AC_strengthGradient].AsFloatf(),
        args[AC_zoomFactor].AsInt(),
        false, false, false, false, 4, 40,std::thread::hardware_concurrency(),
        args[AC_HDN].AsBool()
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


    if (inputs.strengthColor < 0.0 || inputs.strengthColor>1.0)
        env->ThrowError("Anime4KCPP: strengthColor must range from 0 to 1!");

    if (inputs.strengthGradient < 0.0 || inputs.strengthGradient>1.0)
        env->ThrowError("Anime4KCPP: strengthGradient must range from 0 to 1!");

    if (inputs.zoomFactor < 1.0)
        env->ThrowError("Anime4KCPP: zoomFactor must be an integer which >= 1!");

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

const AVS_Linkage* AVS_linkage = 0;

extern "C" AC_DLL const char* AC_STDCALL AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;
    env->AddFunction("listGPUs", "", listGPUs, 0);
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
        "[platformID]i"
        "[deviceID]i",
        createAnime4KCPP, 0);
    return "Anime4KCPP plugin for AviSynthPlus";
}
