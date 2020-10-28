#pragma once
#include "ACNet.h"
#include "ACNetHDN.h"

namespace Anime4KCPP
{
    class CNNCreator;

    enum class CNNType;
}

enum class Anime4KCPP::CNNType
{
    Default, ACNet, ACNetHDNL1, ACNetHDNL2, ACNetHDNL3
};

//CNN Processor Factory
class Anime4KCPP::CNNCreator
{
public:
    static CNNProcessor* create(const CNNType& type);
    static void release(CNNProcessor*& processor) noexcept;
};
