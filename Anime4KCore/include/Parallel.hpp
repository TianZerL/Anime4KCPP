#pragma once

#include <utility>

#ifndef DISABLE_PARALLEL
#if defined(_MSC_VER) && !defined(USE_TBB)
#include <ppl.h>
namespace Parallel = Concurrency;
#elif defined(USE_TBB)
#include <tbb/parallel_for.h>
namespace Parallel = tbb;
#elif defined(USE_OPENMP)
#include <omp.h>
#else
#include "ThreadPool.hpp"
#endif
#endif // !DISABLE_PARALLEL

namespace Anime4KCPP::Utils
{
    template <typename F>
    void ParallelFor(const int first, const int last, F&& func);
}

template <typename F>
inline void Anime4KCPP::Utils::ParallelFor(const int first, const int last, F&& func)
{
#ifndef DISABLE_PARALLEL
#if defined(_MSC_VER) || defined(USE_TBB)
    Parallel::parallel_for(first, last, std::forward<F>(func));
#elif defined(USE_OPENMP)
#pragma omp parallel for
    for (int i = first; i < last; i++)
    {
        func(i);
    }
#else
    static const size_t threadNum = std::thread::hardware_concurrency();
    if (threadNum > 1)
    {
        static Anime4KCPP::Utils::ThreadPool pool(threadNum);
        std::mutex mtx;
        std::condition_variable cnd;
        int count = last - first;

        for (int i = first; i < last; i++)
        {
            pool.exec([&mtx, &cnd, &count, &func, i]()
                {
                    func(i);
                    std::unique_lock<std::mutex> lock(mtx);
                    if (--count == 0)
                        cnd.notify_one();
                });
        }

        std::unique_lock<std::mutex> lock(mtx);
        while (count != 0)
            cnd.wait(lock);
    }
    else
    {
        for (int i = first; i < last; i++)
        {
            func(i);
        }
    }
#endif
#else
    for (int i = first; i < last; i++)
    {
        func(i);
    }
#endif // !DISABLE_PARALLEL
}
