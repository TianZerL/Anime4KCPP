#ifndef AC_UTIL_STOPWATCH_HPP
#define AC_UTIL_STOPWATCH_HPP

#include <chrono>

namespace ac::util
{
    class Stopwatch;
}

class ac::util::Stopwatch
{
public:
    Stopwatch() noexcept;
    void reset() noexcept;
    void stop() noexcept;
    double elapsed() noexcept;
private:
    std::chrono::time_point<std::chrono::steady_clock> start, end;
};

inline ac::util::Stopwatch::Stopwatch() noexcept
{
    reset();
}

inline void ac::util::Stopwatch::reset() noexcept
{
    start = std::chrono::steady_clock::now();
}
inline void ac::util::Stopwatch::stop() noexcept
{
    end = std::chrono::steady_clock::now();
}
inline double ac::util::Stopwatch::elapsed() noexcept
{
    return std::chrono::duration<double>{ end - start }.count();
}
#endif
