#define DLL

#include "Anime4KCPP.h"

Anime4KCPP::Anime4KCreator::Anime4KCreator(bool initGPU, unsigned int platformID, unsigned int deviceID)
{
    if (initGPU && !Anime4KGPU::isInitializedGPU())
        Anime4KGPU::initGPU(platformID, deviceID);
}

Anime4KCPP::Anime4KCreator::Anime4KCreator(bool initGPU, bool initCNN, unsigned int platformID, unsigned int deviceID, const CNNType type)
{
    if (initGPU && initCNN && !Anime4KGPUCNN::isInitializedGPU())
        Anime4KGPUCNN::initGPU(platformID, deviceID, type);
    else if (initGPU && !Anime4KGPU::isInitializedGPU())
        Anime4KGPU::initGPU(platformID, deviceID);
}

Anime4KCPP::Anime4KCreator::~Anime4KCreator()
{
    if (Anime4KGPU::isInitializedGPU())
        Anime4KGPU::releaseGPU();
    if (Anime4KGPUCNN::isInitializedGPU())
        Anime4KGPUCNN::releaseGPU();
}

Anime4KCPP::Anime4K* Anime4KCPP::Anime4KCreator::create(
    const Parameters& parameters, const ProcessorType type
)
{
    switch (type)
    {
    case ProcessorType::CPU:
        return new Anime4KCPU(parameters);
        break;
    case ProcessorType::GPU:
        return new Anime4KGPU(parameters);
        break;
    case ProcessorType::CPUCNN:
        return new Anime4KCPUCNN(parameters);
        break;
    case ProcessorType::GPUCNN:
        return new Anime4KGPUCNN(parameters);
        break;
    default:
        return nullptr;
        break;
    }
}

void Anime4KCPP::Anime4KCreator::release(Anime4K*& anime4K)
{
    if (anime4K != nullptr)
    {
        delete anime4K;
        anime4K = nullptr;
    }
}
