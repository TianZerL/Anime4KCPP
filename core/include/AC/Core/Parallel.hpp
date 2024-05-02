#ifndef AC_CORE_PARALLEL_HPP
#define AC_CORE_PARALLEL_HPP

#if defined(AC_CORE_PARALLEL_PPL)
#   include <ppl.h>
#elif defined(AC_CORE_PARALLEL_OPENMP)
#   include <omp.h>
#else
#   include "AC/Util/ThreadPool.hpp"
#endif

namespace ac::core
{
    template <typename IndexType, typename F>
    void parallelFor(IndexType first, IndexType last, F&& func);
}

template <typename IndexType, typename F>
inline void ac::core::parallelFor(const IndexType first, const IndexType last, F&& func)
{
#   if defined(AC_CORE_PARALLEL_PPL)
        Concurrency::parallel_for(first, last, std::forward<F>(func));
#   elif defined(AC_CORE_PARALLEL_OPENMP)
#       pragma omp parallel for
        for (IndexType i = first; i < last; i++) func(i);
#   else
        static const std::size_t threads = ac::util::ThreadPool::hardwareThreads();
        if (threads > 1)
        {
            static ac::util::ThreadPool pool(threads + 1);
            std::vector<std::future<void>> tasks{};
            tasks.reserve(static_cast<decltype(tasks.size())>(last) - static_cast<decltype(tasks.size())>(first));
            for (IndexType i = first; i < last; i++) tasks.emplace_back(pool.exec([&](IndexType idx) { func(idx); }, i));
            for (auto&& task : tasks) task.wait();
        }
        else for (IndexType i = first; i < last; i++) func(i);
#   endif
}

#endif
