#pragma once

#include <utility>

#if defined(_MSC_VER) && !defined(USE_TBB)
#include <ppl.h>
namespace Parallel = Concurrency;
#elif defined(USE_TBB)
#include <tbb/parallel_for.h>
namespace Parallel = tbb;
#else
#include <omp.h>
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
#else
#pragma omp parallel for
    for (int i = first; i < last; i++)
    {
        func(i);
    }
#endif
}
