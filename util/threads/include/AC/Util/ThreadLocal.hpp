#ifndef AC_UTIL_THREAD_LOCAL_HPP
#define AC_UTIL_THREAD_LOCAL_HPP

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <utility>

namespace ac::util
{
    template<typename T, template<typename...> typename Map = std::unordered_map>
    class ThreadLocal;
}

template<typename T, template<typename...> typename Map>
class ac::util::ThreadLocal
{
public:
    template<typename... Args>
    T& local(Args&&... args);

private:
    Map<std::thread::id, std::unique_ptr<T>> map;
    std::shared_mutex mtx;
};

template<typename T, template<typename...> typename Map>
template<typename... Args>
inline T& ac::util::ThreadLocal<T, Map>::local(Args&&... args)
{
    {
        std::shared_lock sharedLock{ mtx };
        auto it = map.find(std::this_thread::get_id());
        if (it != map.end()) return *it->second;
    }

    {
        std::unique_lock uniqueLock{ mtx };
        return *map.try_emplace(std::this_thread::get_id(), std::make_unique<T>(std::forward<Args>(args)...)).first->second;
    }
}

#endif
