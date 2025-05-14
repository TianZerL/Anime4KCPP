#ifndef AC_CLI_PROGRESS_BAR_CPP
#define AC_CLI_PROGRESS_BAR_CPP

#include "AC/Util/Stopwatch.hpp"

class ProgressBar
{
public:
    ProgressBar(const ac::util::Stopwatch& stopwatch = {}) noexcept;

    void reset() noexcept;
    void print(double p) noexcept;
    void finish() noexcept;
private:
    ac::util::Stopwatch stopwatch;
};

#endif
