#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "AC/Util/Channel.hpp"
#include "AC/Util/ThreadPool.hpp"

int main()
{
    std::mutex mtx{};

    ac::util::Channel<int> chan{3};
    ac::util::ThreadPool pool{3};

    for (int i = 0; i < 5; i++)
    {
        pool.exec([&](){
            int n{};
            chan >> n;
            std::lock_guard<std::mutex> lock(mtx);
            std::cout << "worker take: " << n << std::endl;
        });
    }

    for (int i = 0; i < 5; i++)
    {
        chan << i;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}
