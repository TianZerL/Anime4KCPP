#ifndef AC_UTIL_CHANNEL_HPP
#define AC_UTIL_CHANNEL_HPP

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <type_traits>
#include <utility>

namespace ac::util
{
    template<typename Queue, typename = void>
    constexpr bool HasTopFunction = false;
    template<typename Queue>
    constexpr bool HasTopFunction<Queue, std::void_t<decltype(std::declval<Queue>().top())>> = true;

    template <typename T, typename Queue = std::queue<T>>
    class Channel;

    template<typename T>
    using AscendingChannel = Channel<T, std::priority_queue<T, std::vector<T>, std::greater<T>>>;
    template<typename T>
    using DescendingChannel = Channel<T, std::priority_queue<T, std::vector<T>, std::less<T>>>;
}

template <typename T, typename Queue>
class ac::util::Channel
{
public:
    explicit Channel(std::size_t capacity);
    Channel(const Channel<T, Queue>&) = delete;
    Channel(Channel<T, Queue>&&) = delete;
    Channel<T, Queue>& operator=(const Channel<T, Queue>&) = delete;
    Channel<T, Queue>& operator=(Channel<T, Queue>&&) = delete;
    ~Channel() = default;

    bool operator<<(const T& obj);
    bool operator>>(T& obj);
    std::size_t size();
    bool empty();
    void close();
    bool isClosed();
private:
    bool stop = false;
    const std::size_t capacity;
    Queue queue;
    std::condition_variable consumer, producer;
    std::mutex mtx;
};

template<typename T, typename Queue>
inline ac::util::Channel<T, Queue>::Channel(const std::size_t capacity) : capacity(capacity) {}
template<typename T, typename Queue>
inline bool ac::util::Channel<T, Queue>::operator<<(const T& obj)
{
    std::unique_lock lock{ mtx };
    producer.wait(lock, [&](){ return stop || queue.size() < capacity; });
    if (stop) return false;

    queue.emplace(obj);
    lock.unlock();
    consumer.notify_one();

    return true;
}
template<typename T, typename Queue>
inline bool ac::util::Channel<T, Queue>::operator>>(T& obj)
{
    std::unique_lock lock{ mtx };
    consumer.wait(lock, [&](){ return stop || !queue.empty(); });
    if (queue.empty()) return false;

    if constexpr (HasTopFunction<Queue>) obj = queue.top();
    else obj = std::move(queue.front());
    queue.pop();

    lock.unlock();
    producer.notify_one();

    return true;
}
template<typename T, typename Queue>
inline std::size_t ac::util::Channel<T, Queue>::size()
{
    const std::lock_guard lock{ mtx };
    return queue.size();
}
template<typename T, typename Queue>
inline bool ac::util::Channel<T, Queue>::empty()
{
    const std::lock_guard lock{ mtx };
    return queue.empty();
}
template<typename T, typename Queue>
inline void ac::util::Channel<T, Queue>::close()
{
    {
        const std::lock_guard lock{ mtx };
        stop = true;
    }
    consumer.notify_all();
    producer.notify_all();
}
template<typename T, typename Queue>
inline bool ac::util::Channel<T, Queue>::isClosed()
{
    const std::lock_guard lock{ mtx };
    return stop;
}

#endif
