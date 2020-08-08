#pragma once
#include "ACNet.h"
#include "ACNetHDN.h"

namespace Anime4KCPP
{
    class CNNCreator;
}

//CNN Processor Factory
class Anime4KCPP::CNNCreator
{
public:
    static CNNProcessor* create(const CNNType& type);
    static void release(CNNProcessor*& processor) noexcept;
};
