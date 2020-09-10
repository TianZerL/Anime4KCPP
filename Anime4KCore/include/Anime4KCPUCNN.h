#pragma once
#include "CNN.h"

#define RULE(x) std::max(x, 0.0)
#define NORM(X) (double(X) / 255.0)
#define UNNORM(n) ((n) >= 255.0? uint8_t(255) : ((n) <= 0.0 ? uint8_t(0) : uint8_t(n)))

namespace Anime4KCPP
{
    class DLL Anime4KCPUCNN;
}

class Anime4KCPP::Anime4KCPUCNN :public Anime4K
{
public:
    Anime4KCPUCNN(const Parameters& parameters = Parameters());
    virtual ~Anime4KCPUCNN() = default;
    virtual void process() override;

private:
    virtual ProcessorType getProcessorType() noexcept override;
};
