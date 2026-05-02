#ifndef AC_CORE_INTERNAL_PROCESSOR_CUDA_COMMON_HPP
#define AC_CORE_INTERNAL_PROCESSOR_CUDA_COMMON_HPP

#include <cuda_runtime.h>

namespace ac::core::cuda
{
    struct DeviceImage
    {
        void* ptr = nullptr;
        int w = 0;
        int h = 0;
        int c = 0;
        int pitch = 0;
        int elementSize = 0;

        __host__ void create(const int w, const int h, const int c, const int elementSize) noexcept
        {
            this->w = w;
            this->h = h;
            this->c = c;
            this->elementSize = elementSize;
            pitch = align(w * c * elementSize, 128);
        }

        __host__ cudaError_t fromHost(const Image& hostImage, const cudaStream_t stream) const noexcept
        {
            auto lineSize = hostImage.width() * hostImage.pixelSize();
            return cudaMemcpy2DAsync(ptr, pitch, hostImage.ptr(), hostImage.stride(), lineSize, hostImage.height(), cudaMemcpyHostToDevice, stream);
        }
        __host__ cudaError_t toHost(Image& hostImage, const cudaStream_t stream) const noexcept
        {
            auto lineSize = w * c * elementSize;
            return cudaMemcpy2DAsync(hostImage.ptr(), hostImage.stride(), ptr, pitch, lineSize, h, cudaMemcpyDeviceToHost, stream);
        }
    };    
}


#endif
