#include <cassert>
#include <cstddef>
#include <memory>
#include <string>
#include <sstream>
#include <vector>

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "AC/Core/Processor.hpp"
#include "AC/Core/Model/ACNet.hpp"
#include "AC/Util/ThreadLocal.hpp"

#include "ACExport.hpp" // Generated by CMake

#define ContextList (ac::core::hip::getContextList())

namespace ac::core::hip
{
    void conv3x3_1to8_hip(
        hipTextureObject_t src,
        void* dst,
        std::size_t dstPitch,
        unsigned int width,
        unsigned int height,
        const float* kernels,
        const float* biases,
        hipStream_t stream = 0
    ) noexcept;
    void conv3x3_8to8_hip(
        const void* src,
        std::size_t srcPitch,
        void* dst,
        std::size_t dstPitch,
        unsigned int width,
        unsigned int height,
        const float* kernels,
        const float* biases,
        hipStream_t stream = 0
    ) noexcept;
    void deconv2x2_8to1_hip(
        const void* src,
        std::size_t srcPitch,
        hipSurfaceObject_t dst,
        unsigned int width,
        unsigned int height,
        const float* kernels,
        Image::ElementType type,
        hipStream_t stream = 0
    ) noexcept;

    struct Context
    {
        std::string name;
        std::size_t vram;
    };

    //lazy load, god knows if it's safe to call the hip function during DLL initialization
    inline static std::vector<Context>& getContextList() noexcept
    {
        static auto contextList = []() -> std::vector<Context> {
            std::vector<Context> contexts{};
            int deviceCount = 0;
            hipGetDeviceCount(&deviceCount);
            for (int i = 0; i < deviceCount; i++)
            {
                hipDeviceProp_t deviceProp{};
                hipGetDeviceProperties(&deviceProp, i);
                contexts.emplace_back(Context{ deviceProp.name, (deviceProp.totalGlobalMem >> 20) });
            }
            return contexts;
        }();
        return contextList;
    }

    inline static hipChannelFormatDesc channelType(const Image::ElementType elementType) noexcept
    {
        switch (elementType)
        {
        case Image::UInt8: return hipCreateChannelDesc<std::uint8_t>();
        case Image::UInt16: return hipCreateChannelDesc<std::uint16_t>();
        case Image::Float32: return hipCreateChannelDesc<float>();
        default: return assert(elementType == Image::UInt8 || elementType == Image::UInt16 || elementType == Image::Float32), hipChannelFormatDesc{};
        }
    }

    class HIPProcessorBase : public Processor
    {
    public:
        HIPProcessorBase(const int device) noexcept
        {
            idx = (device >= 0 && static_cast<decltype(ContextList.size())>(device) < ContextList.size()) ? device : 0;
        };
        ~HIPProcessorBase() noexcept override = default;

        bool ok() noexcept override
        {
            return errors.local() == hipSuccess;
        }
        const char* error() noexcept override
        {
            return hipGetErrorString(errors.local());
        }
        const char* name() const noexcept override
        {
            return ContextList[idx].name.c_str();
        }
    protected:
        util::ThreadLocal<hipError_t> errors;
    };

    template<typename Model>
    class HIPProcessor;
}

template<>
class ac::core::hip::HIPProcessor<ac::core::model::ACNet> : public ac::core::hip::HIPProcessorBase
{
public:
    HIPProcessor(int device, const model::ACNet& model) noexcept;
    ~HIPProcessor() noexcept override;
private:
    void process(const Image& src, Image& dst) override;
private:
    float* kernels = nullptr;
    float* biases = nullptr;
};

ac::core::hip::HIPProcessor<ac::core::model::ACNet>::HIPProcessor(const int device, const model::ACNet& model) noexcept : HIPProcessorBase(device)
{
    auto& err = errors.local();
    err = hipSetDevice(idx); if (err != hipSuccess) return;
    err = hipMalloc(&kernels, model.kernelSize()); if (err != hipSuccess) return;
    err = hipMalloc(&biases, model.biasSize()); if (err != hipSuccess) return;
    err = hipMemcpy(kernels, model.kernels(), model.kernelSize(), hipMemcpyHostToDevice); if (err != hipSuccess) return;
    err = hipMemcpy(biases, model.biases(), model.biasSize(), hipMemcpyHostToDevice);
}
ac::core::hip::HIPProcessor<ac::core::model::ACNet>::~HIPProcessor() noexcept
{
    hipSetDevice(idx);

    if (kernels) hipFree(kernels);
    if (biases) hipFree(biases);
}

void ac::core::hip::HIPProcessor<ac::core::model::ACNet>::process(const Image& src, Image& dst)
{
    hipSetDevice(idx);

    auto stream = hipStreamPerThread;

    auto srcW = src.width(), srcH = src.height();
    auto dstW = dst.width(), dstH = dst.height();
    auto srcWBytes = srcW * src.channelSize();
    auto dstWBytes = dstW * dst.channelSize();

    auto imgDesc = channelType(src.type());

    hipArray_t inArray{};
    hipArray_t outArray{};
    void* tmp1{};
    void* tmp2{};
    std::size_t tmp1Pitch{};
    std::size_t tmp2Pitch{};

    hipMallocArray(&inArray, &imgDesc, srcW, srcH);
    hipMallocArray(&outArray, &imgDesc, dstW, dstH, hipArraySurfaceLoadStore);
    hipMallocPitch(&tmp1, &tmp1Pitch, srcW * 8 * sizeof(unsigned short), srcH);
    hipMallocPitch(&tmp2, &tmp2Pitch, srcW * 8 * sizeof(unsigned short), srcH);

    hipResourceDesc resDesc{};
    hipTextureDesc texDesc{};

    resDesc.resType = hipResourceTypeArray;
    texDesc.addressMode[0] = hipAddressModeBorder;
    texDesc.addressMode[1] = hipAddressModeBorder;
    texDesc.addressMode[2] = hipAddressModeBorder;
    texDesc.filterMode = hipFilterModePoint;
    texDesc.normalizedCoords = false;
    texDesc.readMode = src.isFloat() ? hipReadModeElementType : hipReadModeNormalizedFloat;

    hipTextureObject_t in{};
    resDesc.res.array.array = inArray;
    hipCreateTextureObject(&in, &resDesc, &texDesc, nullptr);

    hipSurfaceObject_t out{};
    resDesc.res.array.array = outArray;
    hipCreateSurfaceObject(&out, &resDesc);

    hipMemcpy2DToArrayAsync(inArray, 0, 0, src.ptr(), src.stride(), srcWBytes, srcH, hipMemcpyHostToDevice, stream);

    conv3x3_1to8_hip(in, tmp1, tmp1Pitch, srcW, srcH, kernels + model::ACNet::kernelOffset[0], biases + model::ACNet::baisOffset[0], stream);
    conv3x3_8to8_hip(tmp1, tmp1Pitch, tmp2, tmp2Pitch, srcW, srcH, kernels + model::ACNet::kernelOffset[1], biases + model::ACNet::baisOffset[1], stream);
    conv3x3_8to8_hip(tmp2, tmp2Pitch, tmp1, tmp1Pitch, srcW, srcH, kernels + model::ACNet::kernelOffset[2], biases + model::ACNet::baisOffset[2], stream);
    conv3x3_8to8_hip(tmp1, tmp1Pitch, tmp2, tmp2Pitch, srcW, srcH, kernels + model::ACNet::kernelOffset[3], biases + model::ACNet::baisOffset[3], stream);
    conv3x3_8to8_hip(tmp2, tmp2Pitch, tmp1, tmp1Pitch, srcW, srcH, kernels + model::ACNet::kernelOffset[4], biases + model::ACNet::baisOffset[4], stream);
    conv3x3_8to8_hip(tmp1, tmp1Pitch, tmp2, tmp2Pitch, srcW, srcH, kernels + model::ACNet::kernelOffset[5], biases + model::ACNet::baisOffset[5], stream);
    conv3x3_8to8_hip(tmp2, tmp2Pitch, tmp1, tmp1Pitch, srcW, srcH, kernels + model::ACNet::kernelOffset[6], biases + model::ACNet::baisOffset[6], stream);
    conv3x3_8to8_hip(tmp1, tmp1Pitch, tmp2, tmp2Pitch, srcW, srcH, kernels + model::ACNet::kernelOffset[7], biases + model::ACNet::baisOffset[7], stream);
    conv3x3_8to8_hip(tmp2, tmp2Pitch, tmp1, tmp1Pitch, srcW, srcH, kernels + model::ACNet::kernelOffset[8], biases + model::ACNet::baisOffset[8], stream);
    deconv2x2_8to1_hip(tmp1, tmp1Pitch, out, dstW, dstH, kernels + model::ACNet::kernelOffset[9], dst.type(), stream);

    hipMemcpy2DFromArrayAsync(dst.ptr(), dst.stride(), outArray, 0, 0, dstWBytes, dstH, hipMemcpyDeviceToHost, stream);

    hipStreamSynchronize(stream);

    hipDestroyTextureObject(in);
    hipDestroySurfaceObject(out);

    hipFreeArray(inArray);
    hipFreeArray(outArray);
    if (tmp1) hipFree(tmp1);
    if (tmp2) hipFree(tmp2);

    errors.local() = hipGetLastError();
}

template<>
AC_EXPORT std::shared_ptr<ac::core::Processor> ac::core::Processor::create<ac::core::Processor::HIP, ac::core::model::ACNet>(const int idx, const model::ACNet& model)
{
    return std::make_shared<hip::HIPProcessor<model::ACNet>>(idx, model);
}
template<>
AC_EXPORT const char* ac::core::Processor::info<ac::core::Processor::HIP>()
{
    static auto infoBuffer = []() -> std::string {
        std::ostringstream buffer{ "HIP:\n", std::ios_base::ate };
        for (int i = 0; i < ContextList.size(); i++)
            buffer << "  [" << i << "] " << ContextList[i].name << " (" << ContextList[i].vram << "MB)" << '\n';
        return buffer.str();
    }();
    return infoBuffer.c_str();
}
