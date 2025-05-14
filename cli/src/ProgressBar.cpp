#include <cstdio>

#include "ProgressBar.hpp"

#define PROGRESS_BAR_TOKEN "=================================================="

ProgressBar::ProgressBar(const ac::util::Stopwatch& stopwatch) noexcept : stopwatch(stopwatch) {}

void ProgressBar::reset() noexcept
{
    print(0.0);
    stopwatch.reset();
}
void ProgressBar::print(const double p) noexcept
{
    constexpr int width = sizeof(PROGRESS_BAR_TOKEN) - 1;
    int done = static_cast<int>(p * width);
    int left = width - done;
    double elapsed = p > 0.0 ? stopwatch.elapsed() : 0.0;
    double remaining = p > 0.0 ? (elapsed / p - elapsed) : 0.0;
    ac::util::Stopwatch::FormatBuffer elapsedBuffer{}, remainingBuffer{};
    std::printf("\r[%.*s%-*s] %6.2lf%% [%s < %s]", done, PROGRESS_BAR_TOKEN, left, ">", p * 100.0, ac::util::Stopwatch::formatDuration(elapsedBuffer, elapsed), ac::util::Stopwatch::formatDuration(remainingBuffer, remaining));
    std::fflush(stdout);
}
void ProgressBar::finish() noexcept
{
    stopwatch.stop();
    print(1.0);
    std::printf("\n");
}
