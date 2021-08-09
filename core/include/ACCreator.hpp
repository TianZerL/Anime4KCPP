#pragma once

#include"ACInitializer.hpp"

namespace Anime4KCPP
{
    class AC_EXPORT ACCreator;
}

//Anime4KCPP Processor Factory
class Anime4KCPP::ACCreator
{
public:
    static std::unique_ptr<AC> createUP(const Parameters& parameters, Processor::Type type);
    static AC* create(const Parameters& parameters, Processor::Type type);
    static void release(AC* ac) noexcept;
};
