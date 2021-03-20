#include <avisynth.h>

#include "Anime4KCPP.hpp"

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
    AC_GPGPUModel = 8,
    AC_HDN = 9,
    AC_HDNLevel = 10,
    AC_platformID = 11,
    AC_deviceID = 12,
    AC_OpenCLQueueNum = 13,
    AC_OpenCLParallelIO = 14,
};

enum class GPGPU
{
    OpenCL, CUDA
};

class Anime4KCPPF : public GenericVideoFilter
{
public:
    Anime4KCPPF(
        PClip _child,
        Anime4KCPP::Parameters& inputs,
        bool CNN,
        bool GPUMode,
        GPGPU GPGPUModel,
        unsigned int pID,
        unsigned int dID,
        int OpenCLQueueNum,
        bool OpenCLParallelIO,
        IScriptEnvironment* env
    );
    
    PVideoFrame AC_STDCALL GetFrame(int n, IScriptEnvironment* env);
    template <typename T>
    PVideoFrame FilterYUV(PVideoFrame& src, PVideoFrame& dst);
    template <typename T>
    PVideoFrame FilterGrayscale(PVideoFrame& src, PVideoFrame& dst);

    PVideoFrame FilterRGB(PVideoFrame& src, PVideoFrame& dst);
private:
    Anime4KCPP::Parameters parameters;
    Anime4KCPP::ACCreator acCreator;
    bool GPUMode;
    bool CNN;
    GPGPU GPGPUModel;
};

Anime4KCPPF::Anime4KCPPF(
    PClip _child,
    Anime4KCPP::Parameters& inputs,
    bool CNN,
    bool GPUMode,
    GPGPU GPGPUModel,
    unsigned int pID,
    unsigned int dID,
    int OpenCLQueueNum,
    bool OpenCLParallelIO,
    IScriptEnvironment* env
) :
    GenericVideoFilter(_child),
    parameters(inputs),
    acCreator(),
    CNN(CNN),
    GPUMode(GPUMode),
    GPGPUModel(GPGPUModel)
{
    if ((!vi.IsRGB24() && (!vi.IsYUV() || !vi.IsPlanar()) && !vi.IsY()) ||
        (vi.BitsPerComponent() != 8 && vi.BitsPerComponent() != 16 && vi.BitsPerComponent() != 32))
    {
        env->ThrowError("Anime4KCPP: supported data type: RGB24 and Grayscale or YUV with 8 or 16bit integer or 32bit float)");
    }

    if (!vi.IsRGB24() && !vi.Is444() && !vi.IsY() && !CNN)
    {
        env->ThrowError("Anime4KCPP: RGB24 or YUV444 or Grayscale is needed for Anime4K09");
    }

    vi.height *= inputs.zoomFactor;
    vi.width *= inputs.zoomFactor;

    if (GPUMode)
    {
        switch (GPGPUModel)
        {
        case GPGPU::OpenCL:
            if (CNN)
                acCreator.pushManager<Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::ACNet>>(
                    pID, dID, 
                    Anime4KCPP::CNNType::Default,
                    OpenCLQueueNum,
                    OpenCLParallelIO);
            else
                acCreator.pushManager<Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::Anime4K09>>(
                    pID, dID, 
                    OpenCLQueueNum,
                    OpenCLParallelIO);
            break;
        case GPGPU::CUDA:
#ifdef ENABLE_CUDA
            acCreator.pushManager<Anime4KCPP::Cuda::Manager>(dID);
#endif // ENABLE_CUDA
            break;
        }
        acCreator.init();
    }
}


PVideoFrame AC_STDCALL Anime4KCPPF::GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrameP(vi, &src);

    if (vi.IsY())
    {
        switch (vi.BitsPerComponent())
        {
        case 8:
            return FilterGrayscale<unsigned char>(src, dst);
        case 16:
            return FilterGrayscale<unsigned short int>(src, dst);
        case 32:
            return FilterGrayscale<float>(src, dst);
        }
    }
    if (vi.IsYUV())
    {
        switch (vi.BitsPerComponent())
        {
        case 8:
            return FilterYUV<unsigned char>(src, dst);
        case 16:
            return FilterYUV<unsigned short int>(src, dst);
        case 32:
            return FilterYUV<float>(src, dst);
        }
    }
    return FilterRGB(src, dst);
}

template<typename T>
PVideoFrame Anime4KCPPF::FilterYUV(PVideoFrame& src, PVideoFrame& dst)
{
    size_t srcPitchY = src->GetPitch(PLANAR_Y);
    size_t dstPitchY = dst->GetPitch(PLANAR_Y);
    size_t srcPitchU = src->GetPitch(PLANAR_U);
    size_t dstPitchU = dst->GetPitch(PLANAR_U);
    size_t srcPitchV = src->GetPitch(PLANAR_V);
    size_t dstPitchV = dst->GetPitch(PLANAR_V);

    size_t srcHY = src->GetHeight(PLANAR_Y);
    size_t srcLY = src->GetRowSize(PLANAR_Y) / sizeof(T);
    size_t srcHU = src->GetHeight(PLANAR_U);
    size_t srcLU = src->GetRowSize(PLANAR_U) / sizeof(T);
    size_t srcHV = src->GetHeight(PLANAR_V);
    size_t srcLV = src->GetRowSize(PLANAR_V) / sizeof(T);

    T* srcpY = const_cast<T*>(reinterpret_cast<const T*>(src->GetReadPtr(PLANAR_Y)));
    T* srcpU = const_cast<T*>(reinterpret_cast<const T*>(src->GetReadPtr(PLANAR_U)));
    T* srcpV = const_cast<T*>(reinterpret_cast<const T*>(src->GetReadPtr(PLANAR_V)));

    unsigned char* dstpY = dst->GetWritePtr(PLANAR_Y);
    unsigned char* dstpU = dst->GetWritePtr(PLANAR_U);
    unsigned char* dstpV = dst->GetWritePtr(PLANAR_V);

    Anime4KCPP::AC* ac = nullptr;

    if (GPUMode)
    {
        switch (GPGPUModel)
        {
        case GPGPU::OpenCL:
            if (CNN)
                ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::OpenCL_ACNet);
            else
                ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::OpenCL_Anime4K09);
            break;
        case GPGPU::CUDA:
#ifdef ENABLE_CUDA
            if (CNN)
                ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::Cuda_ACNet);
            else
                ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::Cuda_Anime4K09);
#endif
            break;
        }
    }
    else
    {
        if (CNN)
            ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::CPU_ACNet);
        else
            ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::CPU_Anime4K09);
    }

    ac->loadImage(
        srcHY, srcLY, srcPitchY, srcpY, 
        srcHU, srcLU, srcPitchU, srcpU,
        srcHV, srcLV, srcPitchV, srcpV);
    ac->process();
    ac->saveImage(dstpY, dstPitchY, dstpU, dstPitchU, dstpV, dstPitchV);

    acCreator.release(ac);

    return dst;
}

template<typename T>
PVideoFrame Anime4KCPPF::FilterGrayscale(PVideoFrame& src, PVideoFrame& dst)
{
    size_t srcPitchY = src->GetPitch(PLANAR_Y);
    size_t dstPitchY = dst->GetPitch(PLANAR_Y);

    size_t srcHY = src->GetHeight(PLANAR_Y);
    size_t srcLY = src->GetRowSize(PLANAR_Y) / sizeof(T);

    T* srcpY = const_cast<T*>(reinterpret_cast<const T*>(src->GetReadPtr(PLANAR_Y)));

    unsigned char* dstpY = dst->GetWritePtr(PLANAR_Y);

    Anime4KCPP::AC* ac = nullptr;

    if (GPUMode)
    {
        switch (GPGPUModel)
        {
        case GPGPU::OpenCL:
            if (CNN)
                ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::OpenCL_ACNet);
            else
                ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::OpenCL_Anime4K09);
            break;
        case GPGPU::CUDA:
#ifdef ENABLE_CUDA
            if (CNN)
                ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::Cuda_ACNet);
            else
                ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::Cuda_Anime4K09);
#endif
            break;
        }
    }
    else
    {
        if (CNN)
            ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::CPU_ACNet);
        else
            ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::CPU_Anime4K09);
    }

    ac->loadImage(srcHY, srcLY, srcPitchY, srcpY, false, false, true);
    ac->process();
    ac->saveImage(dstpY, dstPitchY);

    acCreator.release(ac);

    return dst;
}

PVideoFrame Anime4KCPPF::FilterRGB(PVideoFrame& src, PVideoFrame& dst)
{
    size_t srcPitch = src->GetPitch();
    size_t dstPitch = dst->GetPitch();

    size_t srcH = src->GetHeight();
    size_t srcL = src->GetRowSize();

    unsigned char* srcp = const_cast<unsigned char*>(src->GetReadPtr());
    unsigned char* dstp = dst->GetWritePtr();

    Anime4KCPP::AC* ac = nullptr;

    if (GPUMode)
    {
        switch (GPGPUModel)
        {
        case GPGPU::OpenCL:
            if (CNN)
                ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::OpenCL_ACNet);
            else
                ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::OpenCL_Anime4K09);
            break;
        case GPGPU::CUDA:
#ifdef ENABLE_CUDA
            if (CNN)
                ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::Cuda_ACNet);
            else
                ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::Cuda_Anime4K09);
#endif
            break;
        }
    }
    else
    {
        if (CNN)
            ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::CPU_ACNet);
        else
            ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::CPU_Anime4K09);
    }

    ac->loadImage(srcH, srcL / 3, srcPitch, srcp);
    ac->process();
    ac->saveImage(dstp, dstPitch);

    acCreator.release(ac);

    return dst;
}

AVSValue AC_CDECL createAnime4KCPP(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    PClip clip = args[0].AsClip();

    if (clip == nullptr)
        env->ThrowError("Anime4KCPP: No input clip found, has clip of 'src' paramenter been specified ?");

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
    int OpenCLQueueNum = args[AC_OpenCLQueueNum].AsInt();
    bool OpenCLParallelIO = args[AC_OpenCLParallelIO].AsBool();
    const char* GPGPUModelTmp = args[AC_GPGPUModel].AsString();

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
    if (!args[AC_OpenCLQueueNum].Defined())
        OpenCLQueueNum = 4;
    if (!args[AC_OpenCLParallelIO].Defined())
        OpenCLParallelIO = false;
    if(!args[AC_GPGPUModel].Defined())
        GPGPUModelTmp = "opencl";

    std::string GPGPUModelString = GPGPUModelTmp;
    std::transform(GPGPUModelString.begin(), GPGPUModelString.end(), GPGPUModelString.begin(), ::tolower);

    GPGPU GPGPUModel;
    if (GPGPUModelString == "opencl")
        GPGPUModel = GPGPU::OpenCL;
    else if (GPGPUModelString == "cuda")
    {
#ifndef ENABLE_CUDA
        env->ThrowError("Anime4KCPP: CUDA is unsupported");
#endif // ENABLE_CUDA
        GPGPUModel = GPGPU::CUDA;
    }
    else
        env->ThrowError("Anime4KCPP: GPGPUModel must be \"cuda\" or \"opencl\"");

    if (inputs.strengthColor < 0.0 || inputs.strengthColor > 1.0)
        env->ThrowError("Anime4KCPP: strengthColor must range from 0 to 1!");

    if (inputs.strengthGradient < 0.0 || inputs.strengthGradient > 1.0)
        env->ThrowError("Anime4KCPP: strengthGradient must range from 0 to 1!");

    if (inputs.zoomFactor < 1.0)
        env->ThrowError("Anime4KCPP: zoomFactor must be an integer which >= 1!");

    if (inputs.HDNLevel < 1 || inputs.HDNLevel > 3)
        env->ThrowError("Anime4KCPP: HDNLevel must range from 1 to 3!");

    if (OpenCLQueueNum < 1)
        env->ThrowError("Anime4KCPP: OpenCLQueueNum must >= 1!");

    if (GPUMode)
    {
        try
        {
            std::string info;
            switch (GPGPUModel)
            {
            case GPGPU::OpenCL:
            {
                Anime4KCPP::OpenCL::GPUInfo ret =
                    Anime4KCPP::OpenCL::checkGPUSupport(pID, dID);
                if (!ret)
                {
                    std::ostringstream err;
                    err << "Anime4KCPP: The current device is unavailable" << std::endl
                        << "Your input is: " << std::endl
                        << "    platform ID: " << pID << std::endl
                        << "    device ID: " << dID << std::endl
                        << "Error: " << std::endl
                        << "    " + ret() << std::endl;
                    env->ThrowError(err.str().c_str());
                }
            }
            break;
            case GPGPU::CUDA:
#ifdef ENABLE_CUDA
            {
                Anime4KCPP::Cuda::GPUInfo ret =
                    Anime4KCPP::Cuda::checkGPUSupport(dID);
                if (!ret)
                {
                    std::ostringstream err;
                    err << "Anime4KCPP: The current device is unavailable" << std::endl
                        << "Your input is: " << std::endl
                        << "    platform ID: " << pID << std::endl
                        << "    device ID: " << dID << std::endl
                        << "Error: " << std::endl
                        << "    " + ret() << std::endl;
                    env->ThrowError(err.str().c_str());
                }
            }
#endif // ENABLE_CUDA
            break;
            }
        }
        catch (const std::exception& err)
        {
            env->ThrowError(err.what());
        }
    }

    return new Anime4KCPPF(
        clip,
        inputs,
        CNN,
        GPUMode,
        GPGPUModel,
        pID,
        dID,
        OpenCLQueueNum,
        OpenCLParallelIO,
        env
    );
}

AVSValue AC_CDECL listGPUs(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    std::string GPGPUModel;
    const char* tmpStr = args[0].AsString();
    if (!args[0].Defined())
        GPGPUModel = "opencl";
    else
        GPGPUModel = tmpStr;
    std::transform(GPGPUModel.begin(), GPGPUModel.end(), GPGPUModel.begin(), ::tolower);

    if (GPGPUModel == "opencl")
        env->ThrowError(Anime4KCPP::OpenCL::listGPUs()().c_str());
    else
#ifdef ENABLE_CUDA
        if (GPGPUModel == "cuda")
            env->ThrowError(Anime4KCPP::Cuda::listGPUs()().c_str());
        else
#endif // ENABLE_CUDA
            env->ThrowError("unkonwn GPGPUModel module");
 
    return AVSValue();
}

AVSValue AC_CDECL benchmark(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    unsigned int pID = args[AC_platformID].AsInt();
    unsigned int dID = args[AC_deviceID].AsInt();


    double CPUScore = Anime4KCPP::benchmark<Anime4KCPP::CPU::ACNet>();
    double OpenCLScore = Anime4KCPP::benchmark<Anime4KCPP::OpenCL::ACNet>(pID, dID);
#ifdef ENABLE_CUDA
    double CudaScore = Anime4KCPP::benchmark<Anime4KCPP::Cuda::ACNet>(dID);
#endif 

    std::ostringstream oss;
    oss << "Benchmark result:" << std::endl
        << "CPU score: " << CPUScore << std::endl
        << "OpenCL score: " << OpenCLScore << std::endl
        << " (pID = " << pID << ", dID = " << dID << ")" << std::endl
#ifdef ENABLE_CUDA
        << "CUDA score: " << OpenCLScore << std::endl
        << " (dID = " << dID << ")" << std::endl
#endif 
    ;
    env->ThrowError(oss.str().c_str());
    return AVSValue();
}

const AVS_Linkage* AVS_linkage = 0;

extern "C" AC_DLL const char* AC_STDCALL AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;

    env->AddFunction("listGPUs", "[GPGPUModel]s", listGPUs, nullptr);

    env->AddFunction("benchmark", 
        "[platformID]i"
        "[deviceID]i", 
        benchmark, nullptr);

    env->AddFunction("Anime4KCPP",
        "c"
        "[passes]i"
        "[pushColorCount]i"
        "[strengthColor]f"
        "[strengthGradient]f"
        "[zoomFactor]i"
        "[ACNet]b"
        "[GPUMode]b"
        "[GPGPUModel]s"
        "[HDN]b"
        "[HDNLevel]i"
        "[platformID]i"
        "[deviceID]i"
        "[OpenCLQueueNum]i"
        "[OpenCLParallelIO]b",
        createAnime4KCPP, nullptr);

    env->AddFunction("Anime4KCPP2",
        "[src]c"
        "[passes]i"
        "[pushColorCount]i"
        "[strengthColor]f"
        "[strengthGradient]f"
        "[zoomFactor]i"
        "[ACNet]b"
        "[GPUMode]b"
        "[GPGPUModel]s"
        "[HDN]b"
        "[HDNLevel]i"
        "[platformID]i"
        "[deviceID]i"
        "[OpenCLQueueNum]i"
        "[OpenCLParallelIO]b",
        createAnime4KCPP, nullptr);

    return "Anime4KCPP plugin for AviSynthPlus";
}
