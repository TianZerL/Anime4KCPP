#include <avisynth.h>

#include "Anime4KCPP.hpp"

#ifdef _MSC_VER
#define AC_STDCALL __stdcall
#define AC_CDECL __cdecl
#define AC_EXPORT __declspec(dllexport)
#else
#define AC_STDCALL __attribute__((__stdcall__))
#define AC_CDECL __attribute__((__cdecl__))
#define AC_EXPORT
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

class Anime4KCPPFilter : public GenericVideoFilter
{
public:
    Anime4KCPPFilter(
        PClip _child,
        const Anime4KCPP::Parameters& parameters,
        bool CNN,
        bool GPUMode,
        GPGPU GPGPUModel,
        int pID,
        int dID,
        int OpenCLQueueNum,
        bool OpenCLParallelIO,
        IScriptEnvironment* env
    );
    
    PVideoFrame AC_STDCALL GetFrame(int n, IScriptEnvironment* env);
    template <typename T>
    PVideoFrame FilterYUV(int n, IScriptEnvironment* env);
    template <typename T>
    PVideoFrame FilterGrayscale(int n, IScriptEnvironment* env);

    PVideoFrame FilterRGB(int n, IScriptEnvironment* env);
private:
    Anime4KCPP::Parameters parameters;
    Anime4KCPP::ACInitializer initializer;
    bool GPUMode;
    bool CNN;
    GPGPU GPGPUModel;
};

Anime4KCPPFilter::Anime4KCPPFilter(
    PClip _child,
    const Anime4KCPP::Parameters& parameters,
    bool CNN,
    bool GPUMode,
    GPGPU GPGPUModel,
    int pID,
    int dID,
    int OpenCLQueueNum,
    bool OpenCLParallelIO,
    IScriptEnvironment* env
) :
    GenericVideoFilter(_child),
    parameters(parameters),
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

    vi.height = std::round(vi.height * parameters.zoomFactor);
    vi.width = std::round(vi.width * parameters.zoomFactor);

    if (GPUMode)
    {
        switch (GPGPUModel)
        {
        case GPGPU::OpenCL:
#ifdef ENABLE_OPENCL
            if (CNN)
                initializer.pushManager<Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::ACNet>>(
                    pID, dID, 
                    Anime4KCPP::CNNType::Default,
                    OpenCLQueueNum,
                    OpenCLParallelIO);
            else
                initializer.pushManager<Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::Anime4K09>>(
                    pID, dID, 
                    OpenCLQueueNum,
                    OpenCLParallelIO);
#endif // ENABLE_OPENCL
            break;
        case GPGPU::CUDA:
#ifdef ENABLE_CUDA
            acCreator.pushManager<Anime4KCPP::Cuda::Manager>(dID);
#endif // ENABLE_CUDA
            break;
        }
        try
        {
            initializer.init();
        }
        catch (const std::exception& e)
        {
            env->ThrowError(e.what());
        }
    }
}


PVideoFrame AC_STDCALL Anime4KCPPFilter::GetFrame(int n, IScriptEnvironment* env)
{
    if (vi.IsY())
    {
        switch (vi.BitsPerComponent())
        {
        case 8:
            return FilterGrayscale<unsigned char>(n, env);
        case 16:
            return FilterGrayscale<unsigned short>(n, env);
        case 32:
            return FilterGrayscale<float>(n, env);
        }
    }
    if (vi.IsYUV())
    {
        switch (vi.BitsPerComponent())
        {
        case 8:
            return FilterYUV<unsigned char>(n, env);
        case 16:
            return FilterYUV<unsigned short>(n, env);
        case 32:
            return  FilterYUV<float>(n, env);
        }
    }
    return FilterRGB(n, env);
}

template<typename T>
PVideoFrame Anime4KCPPFilter::FilterYUV(int n, IScriptEnvironment* env)
{
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrameP(vi, &src);

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

    std::unique_ptr<Anime4KCPP::AC> ac;

    if (GPUMode)
    {
        switch (GPGPUModel)
        {
        case GPGPU::OpenCL:
#ifdef ENABLE_OPENCL
            if (CNN)
                ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::OpenCL_ACNet);
            else
                ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::OpenCL_Anime4K09);
#endif
            break;
        case GPGPU::CUDA:
#ifdef ENABLE_CUDA
            if (CNN)
                ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::Cuda_ACNet);
            else
                ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::Cuda_Anime4K09);
#endif
            break;
        }
    }
    else
    {
        if (CNN)
            ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::CPU_ACNet);
        else
            ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::CPU_Anime4K09);
    }

    try
    {
        ac->loadImage(
            srcHY, srcLY, srcPitchY, srcpY,
            srcHU, srcLU, srcPitchU, srcpU,
            srcHV, srcLV, srcPitchV, srcpV);
        ac->process();
        ac->saveImage(dstpY, dstPitchY, dstpU, dstPitchU, dstpV, dstPitchV);
    }
    catch (const std::exception& e)
    {
        env->ThrowError(e.what());
    }

    return dst;
}

template<typename T>
PVideoFrame Anime4KCPPFilter::FilterGrayscale(int n, IScriptEnvironment* env)
{
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrameP(vi, &src);

    size_t srcPitchY = src->GetPitch(PLANAR_Y);
    size_t dstPitchY = dst->GetPitch(PLANAR_Y);

    size_t srcHY = src->GetHeight(PLANAR_Y);
    size_t srcLY = src->GetRowSize(PLANAR_Y) / sizeof(T);

    T* srcpY = const_cast<T*>(reinterpret_cast<const T*>(src->GetReadPtr(PLANAR_Y)));

    unsigned char* dstpY = dst->GetWritePtr(PLANAR_Y);

    std::unique_ptr<Anime4KCPP::AC> ac;

    if (GPUMode)
    {
        switch (GPGPUModel)
        {
        case GPGPU::OpenCL:
#ifdef ENABLE_OPENCL
            if (CNN)
                ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::OpenCL_ACNet);
            else
                ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::OpenCL_Anime4K09);
#endif
            break;
        case GPGPU::CUDA:
#ifdef ENABLE_CUDA
            if (CNN)
                ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::Cuda_ACNet);
            else
                ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::Cuda_Anime4K09);
#endif
            break;
        }
    }
    else
    {
        if (CNN)
            ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::CPU_ACNet);
        else
            ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::CPU_Anime4K09);
    }

    try
    {
        ac->loadImage(srcHY, srcLY, srcPitchY, srcpY, false, false, true);
        ac->process();
        ac->saveImage(dstpY, dstPitchY);
    }
    catch (const std::exception& e)
    {
        env->ThrowError(e.what());
    }

    return dst;
}

PVideoFrame Anime4KCPPFilter::FilterRGB(int n, IScriptEnvironment* env)
{
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrameP(vi, &src);

    size_t srcPitch = src->GetPitch();
    size_t dstPitch = dst->GetPitch();

    size_t srcH = src->GetHeight();
    size_t srcL = src->GetRowSize();

    unsigned char* srcp = const_cast<unsigned char*>(src->GetReadPtr());
    unsigned char* dstp = dst->GetWritePtr();

    std::unique_ptr<Anime4KCPP::AC> ac;

    if (GPUMode)
    {
        switch (GPGPUModel)
        {
        case GPGPU::OpenCL:
#ifdef ENABLE_OPENCL
            if (CNN)
                ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::OpenCL_ACNet);
            else
                ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::OpenCL_Anime4K09);
#endif
            break;
        case GPGPU::CUDA:
#ifdef ENABLE_CUDA
            if (CNN)
                ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::Cuda_ACNet);
            else
                ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::Cuda_Anime4K09);
#endif
            break;
        }
    }
    else
    {
        if (CNN)
            ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::CPU_ACNet);
        else
            ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::CPU_Anime4K09);
    }

    try
    {
        ac->loadImage(srcH, srcL / 3, srcPitch, srcp);
        ac->process();
        ac->saveImage(dstp, dstPitch);
    }
    catch (const std::exception& e)
    {
        env->ThrowError(e.what());
    }

    return dst;
}

AVSValue AC_CDECL createAnime4KCPP(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    PClip clip = args[0].AsClip();

    if (clip == nullptr)
        env->ThrowError("Anime4KCPP: No input clip found, has clip of 'src' paramenter been specified ?");

    Anime4KCPP::Parameters parameters(
        args[AC_passes].AsInt(),
        args[AC_pushColorCount].AsInt(),
        args[AC_strengthColor].AsFloat(),
        args[AC_strengthGradient].AsFloat(),
        args[AC_zoomFactor].AsFloat(),
        false, false, false, 4, 40, std::thread::hardware_concurrency(),
        args[AC_HDN].AsBool(),
        args[AC_HDNLevel].AsInt()
    );

    bool CNN = args[AC_ACNet].AsBool();
    bool GPUMode = args[AC_GPUMode].AsBool();
    int pID = args[AC_platformID].AsInt();
    int dID = args[AC_deviceID].AsInt();
    int OpenCLQueueNum = args[AC_OpenCLQueueNum].AsInt();
    bool OpenCLParallelIO = args[AC_OpenCLParallelIO].AsBool();
    const char* GPGPUModelTmp = args[AC_GPGPUModel].AsString();

    if (!args[AC_passes].Defined())
        parameters.passes = 2;
    if (!args[AC_pushColorCount].Defined())
        parameters.pushColorCount = 2;
    if (!args[AC_strengthColor].Defined())
        parameters.strengthColor = 0.3;
    if (!args[AC_strengthGradient].Defined())
        parameters.strengthGradient = 1.0;
    if (!args[AC_zoomFactor].Defined())
        parameters.zoomFactor = 2.0;
    if (!args[AC_HDN].Defined())
        parameters.HDN = false;
    if (!args[AC_HDNLevel].Defined())
        parameters.HDNLevel = 1;
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
    if (!args[AC_GPGPUModel].Defined())
#ifdef ENABLE_OPENCL
        GPGPUModelTmp = "opencl";
#elif defined(ENABLE_CUDA)
        GPGPUModelTmp = "cuda";
#else
        GPGPUModelTmp = "cpu";
#endif 

    std::string GPGPUModelString = GPGPUModelTmp;
    std::transform(GPGPUModelString.begin(), GPGPUModelString.end(), GPGPUModelString.begin(), ::tolower);

    GPGPU GPGPUModel;
    if (GPGPUModelString == "opencl")
    {
#ifndef ENABLE_OPENCL
        env->ThrowError("Anime4KCPP: OpenCL is unsupported");
#endif // ENABLE_OPENCL
        GPGPUModel = GPGPU::OpenCL;
    }
    else if (GPGPUModelString == "cuda")
    {
#ifndef ENABLE_CUDA
        env->ThrowError("Anime4KCPP: CUDA is unsupported");
#endif // ENABLE_CUDA
        GPGPUModel = GPGPU::CUDA;
    }
    else if (GPGPUModelString == "cpu")
    {
        if (GPUMode)
        {
            env->ThrowError("Anime4KCPP: OpenCL or CUDA is unsupported");
        }
    }
    else
        env->ThrowError("Anime4KCPP: GPGPUModel must be \"cuda\" or \"opencl\"");

    if (parameters.strengthColor < 0.0 || parameters.strengthColor > 1.0)
        env->ThrowError("Anime4KCPP: strengthColor must range from 0 to 1!");

    if (parameters.strengthGradient < 0.0 || parameters.strengthGradient > 1.0)
        env->ThrowError("Anime4KCPP: strengthGradient must range from 0 to 1!");

    if (parameters.zoomFactor < 1.0)
        env->ThrowError("Anime4KCPP: zoomFactor must >= 1.0!");

    if (parameters.HDNLevel < 1 || parameters.HDNLevel > 3)
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
#ifdef ENABLE_OPENCL
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
#endif
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
#endif
            break;
            }
        }
        catch (const std::exception& err)
        {
            env->ThrowError(err.what());
        }
    }

    return new Anime4KCPPFilter(
        clip,
        parameters,
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

#ifdef ENABLE_OPENCL
    if (GPGPUModel == "opencl")
        env->ThrowError(Anime4KCPP::OpenCL::listGPUs()().c_str());
    else
#endif
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
    int pID = 0;
    int dID = 0;

    if (args[0].Defined())
        pID = args[0].AsInt();
    if (args[1].Defined())
        dID = args[1].AsInt();

    double CPUScoreDVD = Anime4KCPP::benchmark<Anime4KCPP::CPU::ACNet, 720, 480>();
    double CPUScoreHD = Anime4KCPP::benchmark<Anime4KCPP::CPU::ACNet, 1280, 720>();
    double CPUScoreFHD = Anime4KCPP::benchmark<Anime4KCPP::CPU::ACNet, 1920, 1080>();

#ifdef ENABLE_OPENCL
    double OpenCLScoreDVD = Anime4KCPP::benchmark<Anime4KCPP::OpenCL::ACNet, 720, 480>(pID, dID);
    double OpenCLScoreHD = Anime4KCPP::benchmark<Anime4KCPP::OpenCL::ACNet, 1280, 720>(pID, dID);
    double OpenCLScoreFHD = Anime4KCPP::benchmark<Anime4KCPP::OpenCL::ACNet, 1920, 1080>(pID, dID);
#endif

#ifdef ENABLE_CUDA
    double CudaScoreDVD = Anime4KCPP::benchmark<Anime4KCPP::Cuda::ACNet, 720, 480>(dID);
    double CudaScoreHD = Anime4KCPP::benchmark<Anime4KCPP::Cuda::ACNet, 1280, 720>(dID);
    double CudaScoreFHD = Anime4KCPP::benchmark<Anime4KCPP::Cuda::ACNet, 1920, 1080>(dID);
#endif 

    std::ostringstream oss;

    oss << "Benchmark test under 8-bit integer input and serial processing..." << std::endl << std::endl;

    oss
        << "CPU score:" << std::endl
        << " DVD(480P->960P): " << CPUScoreDVD << " FPS" << std::endl
        << " HD(720P->1440P): " << CPUScoreHD << " FPS" << std::endl
        << " FHD(1080P->2160P): " << CPUScoreFHD << " FPS" << std::endl << std::endl;

#ifdef ENABLE_OPENCL
    oss
        << "OpenCL score:" << " (pID = " << pID << ", dID = " << dID << ")" << std::endl
        << " DVD(480P->960P): " << OpenCLScoreDVD << " FPS" << std::endl
        << " HD(720P->1440P): " << OpenCLScoreHD << " FPS" << std::endl
        << " FHD(1080P->2160P): " << OpenCLScoreFHD << " FPS" << std::endl << std::endl;
#endif

#ifdef ENABLE_CUDA
    oss
        << "CUDA score:" << " (dID = " << dID << ")" << std::endl
        << " DVD(480P->960P): " << CudaScoreDVD << " FPS" << std::endl
        << " HD(720P->1440P): " << CudaScoreHD << " FPS" << std::endl
        << " FHD(1080P->2160P): " << CudaScoreFHD << " FPS" << std::endl << std::endl;
#endif 

    env->ThrowError(oss.str().c_str());
    return AVSValue();
}

const AVS_Linkage* AVS_linkage = 0;

extern "C" AC_EXPORT const char* AC_STDCALL AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors)
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
        "[zoomFactor]f"
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
        "[zoomFactor]f"
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
