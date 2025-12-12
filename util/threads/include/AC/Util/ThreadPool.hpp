#ifndef AC_UTIL_THREADPOOL_HPP
#define AC_UTIL_THREADPOOL_HPP

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

namespace ac::util
{
    class ThreadPool;
}

class ac::util::ThreadPool
{
public:
    explicit ThreadPool(std::size_t size);
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;
    ~ThreadPool();

    template<typename F> void exec(F&& f);
    template<typename F, typename... Args> auto exec(F&& f, Args&&... args);
    void wait();

public:
    static unsigned int hardwareThreads() noexcept;

private:
    bool stop = false;
    std::size_t busy = 0;
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> tasks;
    std::condition_variable cvt;
    std::condition_variable cvw;
    std::mutex mtx;
};

inline ac::util::ThreadPool::ThreadPool(const std::size_t size)
{
    threads.reserve(size);

    for (std::size_t i = 0; i < size; i++)
        threads.emplace_back([this]() {
            for (;;)
            {
                std::unique_lock lock{ mtx };
                cvt.wait(lock, [this] { return stop || !tasks.empty(); });
                if (stop && tasks.empty()) return;
                busy++;
                auto task = std::move(tasks.front());
                tasks.pop();
                lock.unlock();
                task();
                lock.lock();
                busy--;
                lock.unlock();
                cvw.notify_all();
            }
        });
}

inline ac::util::ThreadPool::~ThreadPool()
{
    {
        const std::lock_guard lock{ mtx };
        stop = true;
    }
    cvt.notify_all();
    cvw.notify_all();
    for (auto&& thread : threads) thread.join();
}

template<typename F>
inline void ac::util::ThreadPool::exec(F&& f)
{
    {
        const std::lock_guard lock{ mtx };
        tasks.emplace(std::forward<F>(f));
    }
    cvt.notify_one();
}

template<typename F, typename ...Args>
inline auto ac::util::ThreadPool::exec(F&& f, Args&& ...args)
{
    auto task = std::make_shared<std::packaged_task<decltype(std::declval<F>()(std::declval<Args>()...))()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    auto ret = task->get_future();
    {
        const std::lock_guard lock{ mtx };
        tasks.emplace([=]() { (*task)(); });
    }
    cvt.notify_one();
    return ret;
}

inline void ac::util::ThreadPool::wait()
{
    std::unique_lock lock{ mtx };
    cvw.wait(lock, [this] { return stop || (tasks.empty() && (busy == 0)); });
}

inline unsigned int ac::util::ThreadPool::hardwareThreads() noexcept
{
    auto num = std::thread::hardware_concurrency();
    return num ? num : 1;
}

#endif
