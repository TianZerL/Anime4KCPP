#include <cstdint>
#include <memory>

#include <avisynth.h>

#include "AC/Core.hpp"

#ifdef _WIN32
#   define EXPORT_API extern "C" __declspec(dllexport)
#   define STDCALL __stdcall
#else
#   define EXPORT_API extern "C"
#   define STDCALL
#endif

class Filter : public GenericVideoFilter
{
public:
    Filter(const AVSValue& args, IScriptEnvironment* env);
    PVideoFrame STDCALL GetFrame(int n, IScriptEnvironment* env) override;
private:
    int type;
    double factor;
    std::shared_ptr<ac::core::Processor> processor;
};

Filter::Filter(const AVSValue& args, IScriptEnvironment* const env) : GenericVideoFilter(args[0].AsClip())
{
    type = [&]() -> int {
        if (vi.IsPlanar() && (vi.IsYUV() || vi.IsY() || vi.IsYUVA()))
        {
            if (vi.BitsPerComponent() == 8) return ac::core::Image::UInt8;
            if (vi.BitsPerComponent() == 16) return ac::core::Image::UInt16;
            if (vi.BitsPerComponent() == 32) return ac::core::Image::Float32;
        }
        return 0;
    }();
    if (!type) env->ThrowError("Anime4KCPP: %s", "only planar YUV uint8, uint16 and float32 input supported");

    factor = args[1].Defined() ? args[1].AsFloat() : 2.0;
    if (factor <= 1.0) env->ThrowError("Anime4KCPP: %s", "this is a upscaler, so make sure factor > 1.0");
    vi.width = static_cast<decltype(vi.width)>(vi.width * factor);
    vi.height = static_cast<decltype(vi.height)>(vi.height * factor);

    auto processorType = args[2].Defined() ? args[2].AsString() : "cpu";

    auto device = args[3].Defined() ? args[3].AsInt() : 0;
    if (device < 0) env->ThrowError("Anime4KCPP: %s", "the device index cannot be negative");

    auto model = args[4].Defined() ? args[4].AsString() : "acnet-hdn0";

    processor = ac::core::Processor::create(processorType, device, model);
    if (!processor->ok()) env->ThrowError("Anime4KCPP: %s", processor->error());
}

PVideoFrame STDCALL Filter::GetFrame(const int n, IScriptEnvironment* const env)
{
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrameP(vi, &src);
    //y
    ac::core::Image srcy{ src->GetRowSize(PLANAR_Y) / vi.ComponentSize(), src->GetHeight(PLANAR_Y), 1, type, const_cast<std::uint8_t*>(src->GetReadPtr(PLANAR_Y)), src->GetPitch(PLANAR_Y) };
    ac::core::Image dsty{ dst->GetRowSize(PLANAR_Y) / vi.ComponentSize(), dst->GetHeight(PLANAR_Y), 1, type, dst->GetWritePtr(PLANAR_Y), dst->GetPitch(PLANAR_Y) };
    processor->process(srcy, dsty, factor);
    if (!processor->ok()) env->ThrowError("Anime4KCPP: %s", processor->error());
    //uv
    int planes[] = { PLANAR_U, PLANAR_V, PLANAR_A };
    for (int n = 0; n < vi.NumComponents() - 1; n++)
    {
        ac::core::Image srcp{ src->GetRowSize(planes[n]) / vi.ComponentSize(), src->GetHeight(planes[n]), 1, type, const_cast<std::uint8_t*>(src->GetReadPtr(planes[n])), src->GetPitch(planes[n]) };
        ac::core::Image dstp{ dst->GetRowSize(planes[n]) / vi.ComponentSize(), dst->GetHeight(planes[n]), 1, type, dst->GetWritePtr(planes[n]), dst->GetPitch(planes[n]) };
        ac::core::resize(srcp, dstp, 0.0, 0.0);
    }

    return dst;
}

const AVS_Linkage* AVS_linkage = 0;

EXPORT_API const char* STDCALL AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;
    env->AddFunction("ACUpscale",
        "c"
        "[factor]f"
        "[processor]s"
        "[device]i"
        "[model]s",
        [] (AVSValue args, void* /*userdata*/, IScriptEnvironment* env) -> AVSValue {
            return new Filter(args, env);
        }, nullptr);
    env->AddFunction("ACInfoList",
        "",
        [] (AVSValue /*args*/, void* /*userdata*/, IScriptEnvironment* /*env*/) -> AVSValue {
            AVSValue info[] = {
                ac::core::Processor::info<ac::core::Processor::CPU>(),
#           ifdef AC_CORE_WITH_OPENCL
                ac::core::Processor::info<ac::core::Processor::OpenCL>(),
#           endif
#           ifdef AC_CORE_WITH_CUDA
                ac::core::Processor::info<ac::core::Processor::CUDA>(),
#           endif
            };
            return AVSValue{ info, sizeof(info) / sizeof(AVSValue) };
        }, nullptr);
    return "Anime4KCPP for AviSynth";
}
