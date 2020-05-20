#pragma once
#include<thread>
#include<mutex>
#include<condition_variable>
#include<functional>
#include<queue>

class ThreadPool
{
    typedef std::function<void()> Task;

public:
    explicit ThreadPool(size_t maxThreadCount) :pool(std::make_shared<Pool>())
    {
        for (size_t i = 0; i < maxThreadCount; i++)
        {
            std::thread(
                [pool = this->pool]{
                    std::unique_lock<std::mutex> lock(pool->mtx);
                    while (true)
                    {
                        if (!pool->tasks.empty())
                        {
                            Task curTask = pool->tasks.front();
                            pool->tasks.pop();
                            lock.unlock();
                            curTask();
                            lock.lock();
                        }
                        else if (pool->isShutdown)
                            break;
                        else
                            pool->cv.wait(lock);
                    }
                }
            ).detach();
        }
    }

    ~ThreadPool()
    {
        if (bool(pool))
        {
            {
                std::lock_guard<std::mutex> lock(pool->mtx);
                pool->isShutdown = true;
            }
            pool->cv.notify_all();
        }
    }

    template<typename F>
    void exec(F&& task) {
        {
            std::lock_guard<std::mutex> lock(pool->mtx);
            pool->tasks.emplace(std::forward<F>(task));
        }
        pool->cv.notify_one();
    }
private:
    struct Pool
    {
        std::mutex mtx;
        std::condition_variable cv;
        bool isShutdown = false;
        std::queue<Task> tasks;
    };

    std::shared_ptr<Pool> pool;
};
