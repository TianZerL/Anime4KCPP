#define DLL

#include "Anime4KCPP.h"

Anime4KCPP::Anime4KCreator::Anime4KCreator(bool initGPU, unsigned int platformID, unsigned int deviceID)
{
    if (initGPU && !Anime4KGPU::isInitializedGPU())
        Anime4KGPU::initGPU(platformID, deviceID);
}

Anime4KCPP::Anime4KCreator::~Anime4KCreator()
{
    if (Anime4KGPU::isInitializedGPU())
        Anime4KGPU::releaseGPU();
}

Anime4KCPP::Anime4K* Anime4KCPP::Anime4KCreator::create(
    const Parameters& parameters, const ProcessorType type
)
{
    if (type == ProcessorType::CPU)
        return new Anime4KCPU(parameters);
    else
        return new Anime4KGPU(parameters);
}

void Anime4KCPP::Anime4KCreator::release(Anime4K*& anime4K)
{
    if (anime4K != nullptr)
        delete anime4K;
}
