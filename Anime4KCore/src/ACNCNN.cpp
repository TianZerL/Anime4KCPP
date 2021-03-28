#ifdef ENABLE_NCNN

#define DLL

#include "ACNCNN.hpp"

Anime4KCPP::NCNN::GPUList Anime4KCPP::NCNN::listGPUs()
{
    int gpuCount = ncnn::get_gpu_count();
    std::string gpuInfo;
    for (int i = 0; i < gpuCount; i++)
    {
        const ncnn::GpuInfo& info = ncnn::get_gpu_info(i);
        gpuInfo += ("Device id: " + std::to_string(i) + "\n  Type: " + info.device_name() + "\n");
    }
    if (gpuCount)
        ncnn::destroy_gpu_instance();
    return GPUList{
        gpuCount,
        gpuInfo.empty() ? "No Vulkan device found, CPU is available for NCNN" : gpuInfo
    };
}

#endif