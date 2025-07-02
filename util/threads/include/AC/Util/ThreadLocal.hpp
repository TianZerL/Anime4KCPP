#ifndef AC_UTIL_THREAD_LOCAL_HPP
#define AC_UTIL_THREAD_LOCAL_HPP

#include <shared_mutex>
#include <thread>
#include <unordered_map>

namespace ac::util
{
    template<typename T, template<typename...> typename Map = std::unordered_map>
    class ThreadLocal;
}

template<typename T, template<typename...> typename Map>
class ac::util::ThreadLocal
{
public:
    T& local();

private:
    Map<std::thread::id, T> map;
    std::shared_mutex mtx;
};

template<typename T, template<typename...> typename Map>
inline T& ac::util::ThreadLocal<T, Map>::local()
{
    std::shared_lock sharedLock{ mtx };
    auto it = map.find(std::this_thread::get_id());
    sharedLock.unlock();
    if (it == map.end())
    {
        const std::lock_guard lock{ mtx };
        return map.emplace(std::this_thread::get_id(), T{}).first->second;
    }
    return it->second;
}

#endif
