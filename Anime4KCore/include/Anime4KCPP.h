#pragma once

#include "Anime4KCPU.h"
#include "Anime4KGPU.h"
#include "Anime4KCPUCNN.h"
#include "Anime4KGPUCNN.h"

#define ANIME4KCPP_CORE_VERSION "2.2.2"

namespace Anime4KCPP
{
    class DLL Anime4KCreator;
}

//Anime4KCPP Processor Factory
class Anime4KCPP::Anime4KCreator
{
public:
    //initialize GPU (This constructor is only for compatibility)
    Anime4KCreator(bool initGPU = false, unsigned int platformID = 0, unsigned int deviceID = 0);
    /*
    initialize GPU or GPUCNN
    initGPU = true and initCNN = true will initialize GPUCNN
    initGPU = true and initCNN = false will initialize GPU
    initGPU = false and initCNN = false will initialize nothing
    type = CNNType::Default will initialize ACNet and ACNet HDN model
    */
    Anime4KCreator(bool initGPU, bool initCNN, unsigned int platformID = 0, unsigned int deviceID = 0, const CNNType type = CNNType::Default);
    ~Anime4KCreator();
    Anime4K* create(const Parameters& parameters, const ProcessorType type);
    void release(Anime4K*& anime4K);
};
