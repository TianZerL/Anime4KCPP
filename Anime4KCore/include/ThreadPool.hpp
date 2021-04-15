#pragma once

#include<thread>
#include<mutex>
#include<condition_variable>
#include<functional>
#include<queue>
#include<vector>

namespace Anime4KCPP
{
    namespace Utils
    {
        class ThreadPool;
    }
}

class Anime4KCPP::Utils::ThreadPool
{
public:
    ThreadPool(size_t maxThreadCount);
    ~ThreadPool();
    template<typename F>
    void exec(F&& task);
private:
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> tasks;
    std::condition_variable cnd;
    std::mutex mtx;
    bool stop;
};

inline Anime4KCPP::Utils::ThreadPool::ThreadPool(size_t maxThreadCount)
    :stop(false)
{
    for (int i = 0; i < maxThreadCount; i++)
        threads.emplace_back([this]()
            {
                for (;;)
                {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        cnd.wait(lock, [this]
                            {
                                return stop || !tasks.empty();
                            });
                        if (stop && tasks.empty())
                            return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
}

inline Anime4KCPP::Utils::ThreadPool::~ThreadPool()
{
    {
        std::lock_guard<std::mutex> lock(mtx);
        stop = true;
    }
    cnd.notify_all();
    std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
}

template<typename F>
void Anime4KCPP::Utils::ThreadPool::exec(F&& f)
{
    {
        std::lock_guard<std::mutex> lock(mtx);
        tasks.emplace(std::forward<F>(f));
    }
    cnd.notify_one();
}
