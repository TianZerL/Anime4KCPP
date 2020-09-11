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

void Anime4KCPP::Anime4KCreator::release(Anime4K*& anime4K) noexcept
{
    if (anime4K != nullptr)
    {
        delete anime4K;
        anime4K = nullptr;
    }
}

std::pair<double, double> Anime4KCPP::benchmark(const unsigned int pID, const unsigned int dID)
{
    std::pair<double, double> ret;

    std::pair<bool, std::string> checkGPUResult = Anime4KCPP::Anime4KGPU::checkGPUSupport(pID, dID);

    Anime4KCPP::Parameters parameters;
    Anime4KCPP::Anime4KCreator creator(checkGPUResult.first, true, pID, dID, Anime4KCPP::CNNType::ACNet);
    Anime4KCPP::Anime4K* acCPU = creator.create(parameters, Anime4KCPP::ProcessorType::CPUCNN);

    const size_t dataSize = 1920 * 1080 * 3;
    uint8_t* testData = new uint8_t[dataSize];

    for (size_t i = 0; i < dataSize; i += 3)
    {
        testData[i + 1] = 255 * cos(i / 3);
        testData[i + 2] = 255 * sin(i / 3);
        testData[i] = (testData[i + 1] + testData[i + 2]) / 2;
    }

    acCPU->loadImage(1920, 1080, testData, 0ULL, true);
    std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();
    acCPU->process();
    std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
    ret.first = 10000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();

    creator.release(acCPU);

    if (checkGPUResult.first)
    {
        Anime4KCPP::Anime4K* acGPU = creator.create(parameters, Anime4KCPP::ProcessorType::GPUCNN);

        acGPU->loadImage(1920, 1080, testData, 0ULL, true);
        s = std::chrono::steady_clock::now();
        acGPU->process();
        e = std::chrono::steady_clock::now();
        ret.second = 10000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();

        creator.release(acGPU);
    }

    delete[] testData;

    return std::move(ret);
}
