#pragma once

#include <utility>

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

namespace Anime4KCPP
{
    namespace Utils
    {
        template <typename F>
        void ParallelFor(const int first, const int last, F&& func);
    }
}

template <typename F>
inline void Anime4KCPP::Utils::ParallelFor(const int first, const int last, F&& func)
{
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

        std::atomic_int count = 0;

        for (int i = first; i < last; i++)
        {
            pool.exec([&count, &func, i]()
                {
                    func(i);
                    count++;
                });
        }

        while (count != last)
        {
            std::this_thread::yield();
        }
    }
    else
    {
        for (int i = first; i < last; i++)
        {
            func(i);
        }
    }
#endif
}
