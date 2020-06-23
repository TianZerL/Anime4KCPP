#pragma once

#include "Anime4KCPU.h"
#include "Anime4KGPU.h"
#include "Anime4KCPUCNN.h"
#include "Anime4KGPUCNN.h"

#define ANIME4KCPP_CORE_VERSION "2.2.1"

namespace Anime4KCPP
{
    class DLL Anime4KCreator;
}

class Anime4KCPP::Anime4KCreator
{
public:
    Anime4KCreator(bool initGPU = false, unsigned int platformID = 0, unsigned int deviceID = 0);
    Anime4KCreator(bool initGPU, bool initCNN, unsigned int platformID = 0, unsigned int deviceID = 0, const CNNType type = CNNType::Default);
    ~Anime4KCreator();
    Anime4K* create(const Parameters& parameters, const ProcessorType type);
    void release(Anime4K*& anime4K);
};
