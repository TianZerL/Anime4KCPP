#include <cassert>
#include <cstddef>
#include <memory>
#include <string>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>

#include "AC/Core/Processor.hpp"
#include "AC/Core/Model/ACNet.hpp"

#include "ACExport.hpp" // Generated by CMake

#define ContextList (ac::core::cuda::getContextList())

namespace ac::core::cuda
{
    void conv3x3_1to8_cuda(
        cudaTextureObject_t src,
        cudaSurfaceObject_t dst,
        const unsigned int width,
        const unsigned int height,
        const float* kernels,
        const float* biases,
        cudaStream_t stream = 0
    ) noexcept;
    void conv3x3_8to8_cuda(
        cudaTextureObject_t src,
        cudaSurfaceObject_t dst,
        const unsigned int width,
        const unsigned int height,
        const float* kernels,
        const float* biases,
        cudaStream_t stream = 0
    ) noexcept;
    void deconv2x2_8to1_cuda(
        cudaTextureObject_t src,
        cudaSurfaceObject_t dst,
        const unsigned int width,
        const unsigned int height,
        const float* kernels,
        Image::ElementType type,
        cudaStream_t stream = 0
    ) noexcept;

    struct Context
    {
        std::string name;
        std::size_t vram;
    };

    //Just like what we do for OpenCL, lazy load for DLL safe
    inline static std::vector<Context>& getContextList() noexcept
    {
        static auto contextList = []() -> std::vector<Context> {
            std::vector<Context> contexts{};
            int deviceCount = 0;
            cudaGetDeviceCount(&deviceCount);
            for (int idx = 0; idx < deviceCount; idx++)
            {
                cudaDeviceProp deviceProp{};
                cudaGetDeviceProperties(&deviceProp, idx);
                contexts.emplace_back(Context{ deviceProp.name, (deviceProp.totalGlobalMem >> 20) });
            }
            return contexts;
        }();
        return contextList;
    }

    inline static cudaChannelFormatDesc channelType(const Image::ElementType elementType) noexcept
    {
        switch (elementType)
        {
        case Image::UInt8: return cudaCreateChannelDesc<std::uint8_t>();
        case Image::UInt16: return cudaCreateChannelDesc<std::uint16_t>();
        case Image::Float32: return cudaCreateChannelDesc<float>();
        default: return assert(elementType == Image::UInt8 || elementType == Image::UInt16 || elementType == Image::Float32), cudaChannelFormatDesc{};
        }
    }

    class CUDAProcessorBase : public Processor
    {
    public:
        CUDAProcessorBase(const int idx) noexcept : Processor(idx), err(cudaSuccess) {};
        ~CUDAProcessorBase() noexcept override = default;

        bool ok() const noexcept override
        {
            return err == cudaSuccess;
        }
        const char* error() const noexcept override
        {
            return err == cudaSuccess ? Processor::error() : cudaGetErrorString(err);
        }
        const char* name() const noexcept override
        {
            return ContextList[idx].name.c_str();
        }
    protected:
        cudaError_t err;
    };

    template<typename Model>
    class CUDAProcessor;
}

template<>
class ac::core::cuda::CUDAProcessor<ac::core::model::ACNet> : public ac::core::cuda::CUDAProcessorBase
{
public:
    CUDAProcessor(int idx, const model::ACNet& model) noexcept;
    ~CUDAProcessor() noexcept override;
private:
    void process(const Image& src, Image& dst) noexcept override;
private:
    float* kernels = nullptr;
    float* biases = nullptr;
};

ac::core::cuda::CUDAProcessor<ac::core::model::ACNet>::CUDAProcessor(const int idx, const model::ACNet& model) noexcept : CUDAProcessorBase(idx)
{
    err = cudaSetDevice(idx); if (err != cudaSuccess) return;
    err = cudaMalloc(&kernels, model.kernelSize()); if (err != cudaSuccess) return;
    err = cudaMalloc(&biases, model.biasSize()); if (err != cudaSuccess) return;
    err = cudaMemcpy(kernels, model.kernels(), model.kernelSize(), cudaMemcpyHostToDevice); if (err != cudaSuccess) return;
    err = cudaMemcpy(biases, model.biases(), model.biasSize(), cudaMemcpyHostToDevice);
}
ac::core::cuda::CUDAProcessor<ac::core::model::ACNet>::~CUDAProcessor() noexcept
{
    if (kernels) cudaFree(kernels);
    if (biases) cudaFree(biases);
}

void ac::core::cuda::CUDAProcessor<ac::core::model::ACNet>::process(const Image& src, Image& dst) noexcept
{
    auto srcW = src.width(), srcH = src.height();
    auto dstW = dst.width(), dstH = dst.height();
    auto srcWBytes = srcW * src.channelSize();
    auto dstWBytes = dstW * dst.channelSize();

    cudaStream_t stream{};
    cudaStreamCreate(&stream);

    auto imgDesc = channelType(src.type());
    auto tmpDesc = cudaCreateChannelDescHalf4();

    cudaArray_t inArray{};
    cudaArray_t tmp1Array{};
    cudaArray_t tmp2Array{};
    cudaArray_t outArray{};

    cudaMallocArray(&inArray, &imgDesc, srcW, srcH);
    cudaMalloc3DArray(&tmp1Array, &tmpDesc, make_cudaExtent(srcW, srcH, 2), cudaArraySurfaceLoadStore | cudaArrayLayered);
    cudaMalloc3DArray(&tmp2Array, &tmpDesc, make_cudaExtent(srcW, srcH, 2), cudaArraySurfaceLoadStore | cudaArrayLayered);
    cudaMallocArray(&outArray, &imgDesc, dstW, dstH, cudaArraySurfaceLoadStore);

    cudaResourceDesc resDesc{};
    cudaTextureDesc texDesc{};

    resDesc.resType = cudaResourceTypeArray;
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.normalizedCoords = false;

    texDesc.readMode = src.isFloat() ? cudaReadModeElementType : cudaReadModeNormalizedFloat;

    cudaTextureObject_t in{};
    resDesc.res.array.array = inArray;
    cudaCreateTextureObject(&in, &resDesc, &texDesc, nullptr);

    texDesc.readMode = cudaReadModeElementType;

    cudaSurfaceObject_t tmp1Store{};
    cudaTextureObject_t tmp1Load{};
    resDesc.res.array.array = tmp1Array;
    cudaCreateSurfaceObject(&tmp1Store, &resDesc);
    cudaCreateTextureObject(&tmp1Load, &resDesc, &texDesc, nullptr);

    cudaSurfaceObject_t tmp2Store{};
    cudaTextureObject_t tmp2Load{};
    resDesc.res.array.array = tmp2Array;
    cudaCreateSurfaceObject(&tmp2Store, &resDesc);
    cudaCreateTextureObject(&tmp2Load, &resDesc, &texDesc, nullptr);

    cudaSurfaceObject_t out{};
    resDesc.res.array.array = outArray;
    cudaCreateSurfaceObject(&out, &resDesc);

    cudaMemcpy2DToArrayAsync(inArray, 0, 0, src.ptr(), src.stride(), srcWBytes, srcH, cudaMemcpyHostToDevice, stream);

    conv3x3_1to8_cuda(in, tmp1Store, srcW, srcH, kernels + model::ACNet::kernelOffset[0], biases + model::ACNet::baisOffset[0], stream);
    conv3x3_8to8_cuda(tmp1Load, tmp2Store, srcW, srcH, kernels + model::ACNet::kernelOffset[1], biases + model::ACNet::baisOffset[1], stream);
    conv3x3_8to8_cuda(tmp2Load, tmp1Store, srcW, srcH, kernels + model::ACNet::kernelOffset[2], biases + model::ACNet::baisOffset[2], stream);
    conv3x3_8to8_cuda(tmp1Load, tmp2Store, srcW, srcH, kernels + model::ACNet::kernelOffset[3], biases + model::ACNet::baisOffset[3], stream);
    conv3x3_8to8_cuda(tmp2Load, tmp1Store, srcW, srcH, kernels + model::ACNet::kernelOffset[4], biases + model::ACNet::baisOffset[4], stream);
    conv3x3_8to8_cuda(tmp1Load, tmp2Store, srcW, srcH, kernels + model::ACNet::kernelOffset[5], biases + model::ACNet::baisOffset[5], stream);
    conv3x3_8to8_cuda(tmp2Load, tmp1Store, srcW, srcH, kernels + model::ACNet::kernelOffset[6], biases + model::ACNet::baisOffset[6], stream);
    conv3x3_8to8_cuda(tmp1Load, tmp2Store, srcW, srcH, kernels + model::ACNet::kernelOffset[7], biases + model::ACNet::baisOffset[7], stream);
    conv3x3_8to8_cuda(tmp2Load, tmp1Store, srcW, srcH, kernels + model::ACNet::kernelOffset[8], biases + model::ACNet::baisOffset[8], stream);
    deconv2x2_8to1_cuda(tmp1Load, out, dstW, dstH, kernels + model::ACNet::kernelOffset[9], dst.type(), stream);

    cudaMemcpy2DFromArrayAsync(dst.ptr(), dst.stride(), outArray, 0, 0, dstWBytes, dstH, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cudaDestroyTextureObject(in);
    cudaDestroySurfaceObject(tmp1Store);
    cudaDestroyTextureObject(tmp1Load);
    cudaDestroySurfaceObject(tmp2Store);
    cudaDestroyTextureObject(tmp2Load);
    cudaDestroySurfaceObject(out);

    cudaFreeArray(inArray);
    cudaFreeArray(tmp1Array);
    cudaFreeArray(tmp2Array);
    cudaFreeArray(outArray);

    cudaStreamDestroy(stream);

    err = cudaGetLastError();
}

template<>
AC_EXPORT std::shared_ptr<ac::core::Processor> ac::core::Processor::create<ac::core::Processor::CUDA, ac::core::model::ACNet>(const int idx, const model::ACNet& model)
{
    return std::make_shared<cuda::CUDAProcessor<model::ACNet>>(idx, model);
}
template<>
AC_EXPORT const char* ac::core::Processor::info<ac::core::Processor::CUDA>()
{
    static auto infoBuffer = []() -> std::string {
        std::ostringstream buffer{ "CUDA:\n", std::ios_base::ate };
        for (int idx = 0; idx < ContextList.size(); idx++)
            buffer << "  [" << idx << "] " << ContextList[idx].name << " (" << ContextList[idx].vram << "MB)" << '\n';
        return buffer.str();
    }();
    return infoBuffer.c_str();
}
