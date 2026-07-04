#ifndef AC_CLI_PROGRESS_BAR_HPP
#define AC_CLI_PROGRESS_BAR_HPP

#include "AC/Util/Stopwatch.hpp"

class ProgressBar
{
public:
    ProgressBar(const ac::util::Stopwatch& stopwatch = {}) noexcept;

    void reset() noexcept;
    void print(double p) noexcept;
    void clear() noexcept;
    void finish() noexcept;
private:
    ac::util::Stopwatch stopwatch;
};

#endif
