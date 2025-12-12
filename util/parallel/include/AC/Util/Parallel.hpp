#ifndef AC_UTIL_PARALLEL_HPP
#define AC_UTIL_PARALLEL_HPP

#if defined(AC_DEP_PARALLEL_PPL)
#   include <ppl.h>
#elif defined(AC_DEP_PARALLEL_OPENMP)
#else
#   include "AC/Util/ThreadPool.hpp"
#endif

namespace ac::util
{
    template <typename IndexType, typename F>
    void parallelFor(IndexType first, IndexType last, F&& func);
}

template <typename IndexType, typename F>
inline void ac::util::parallelFor(const IndexType first, const IndexType last, F&& func)
{
#if defined(AC_DEP_PARALLEL_PPL)
    Concurrency::parallel_for(first, last, std::forward<F>(func));
#elif defined(AC_DEP_PARALLEL_OPENMP)
#   pragma omp parallel for
    for (IndexType i = first; i < last; i++) func(i);
#else
    static const std::size_t threads = ThreadPool::hardwareThreads();
    auto range = last - first;
    if (range > 0)
    {
        if (threads > 1 && range > 1)
        {
            static ThreadPool pool{ threads + 1 };

            auto blocks = range / threads;
            auto remain = range % threads;

            std::vector<std::future<void>> tasks{};
            tasks.reserve(static_cast<decltype(tasks.size())>(threads + remain));

            IndexType i = first;
            if (blocks)
                while (i < (last - remain))
                {
                    auto next = i + static_cast<IndexType>(blocks);
                    tasks.emplace_back(pool.exec([&](const IndexType startIdx, const IndexType endIdx) { for (IndexType idx = startIdx; idx < endIdx; idx++) func(idx); }, i, next));
                    i = next;
                }
            if (remain)
                while (i < last)
                {
                    tasks.emplace_back(pool.exec(func, i));
                    i++;
                }

            for (auto&& task : tasks) task.wait();
        }
        else for (IndexType i = first; i < last; i++) func(i);
    }
#endif
}

#endif
