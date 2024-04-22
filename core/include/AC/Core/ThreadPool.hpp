#ifndef AC_CORE_THREADPOOL_HPP
#define AC_CORE_THREADPOOL_HPP

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

namespace ac::core
{
    class ThreadPool;
}

class ac::core::ThreadPool
{
public:
    explicit ThreadPool(std::size_t size);
    ~ThreadPool();

    template<typename F> void exec(F&& f);
    template<typename F, typename... Args> auto exec(F&& f, Args&&... args);

    static unsigned int concurrentThreads() noexcept;

private:
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> tasks;
    std::condition_variable cnd;
    std::mutex mtx;
    bool stop;
};

inline ac::core::ThreadPool::ThreadPool(std::size_t size) :stop(false)
{
    threads.reserve(size);

    for (int i = 0; i < size; i++)
        threads.emplace_back([this]() {
            for (;;)
            {
                std::unique_lock<std::mutex> lock(mtx);
                cnd.wait(lock, [this] { return stop || !tasks.empty(); });
                if (stop && tasks.empty()) return;
                auto task = std::move(tasks.front());
                tasks.pop();
                lock.unlock();
                task();
            }
        });
}

inline ac::core::ThreadPool::~ThreadPool()
{
    {
        const std::lock_guard<std::mutex> lock(mtx);
        stop = true;
    }
    cnd.notify_all();
    std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
}

template<typename F>
inline void ac::core::ThreadPool::exec(F&& f)
{
    {
        const std::lock_guard<std::mutex> lock(mtx);
        tasks.emplace(std::forward<F>(f));
    }
    cnd.notify_one();
}

template<typename F, typename ...Args>
inline auto ac::core::ThreadPool::exec(F&& f, Args && ...args)
{
    auto task = std::make_shared<std::packaged_task<decltype(std::declval<F>()(std::declval<Args>()...))()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    auto ret = task->get_future();
    {
        const std::lock_guard<std::mutex> lock(mtx);
        tasks.emplace([=]() { (*task)(); });
    }
    cnd.notify_one();
    return ret;
}

inline unsigned int ac::core::ThreadPool::concurrentThreads() noexcept
{
    auto num = std::thread::hardware_concurrency();
    return num ? num : 1;
}

#endif
