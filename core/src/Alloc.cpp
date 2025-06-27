#if defined(AC_CORE_HAVE_WIN32_ALIGNED_MALLOC)
#   include <malloc.h>
#else
#   include <cstdlib>
#endif

#include "AC/Core/Util.hpp"

#ifndef AC_CORE_MALLOC_ALIGN
#   define AC_CORE_MALLOC_ALIGN 32
#endif

#if !defined(AC_CORE_HAVE_WIN32_ALIGNED_MALLOC) && !defined(AC_CORE_HAVE_STD_ALIGNED_ALLOC)
namespace ac::core::detail
{
    template <typename T>
    inline static constexpr T* alignPtr(T* const ptr, const int n) noexcept
    {
        return reinterpret_cast<T*>(align(reinterpret_cast<std::uintptr_t>(ptr), n));
    }

    inline static void* alignedAlloc(const std::size_t size) noexcept
    {
        auto buffer = static_cast<void**>(std::malloc(size + sizeof(void*) + AC_CORE_MALLOC_ALIGN));
        if (!buffer) return nullptr;
        void** ptr = alignPtr(buffer + 1, AC_CORE_MALLOC_ALIGN);
        ptr[-1] = static_cast<void*>(buffer);
        return ptr;
    }
    inline static void alignedFree(void* const ptr) noexcept
    {
        if (ptr) std::free(static_cast<void**>(ptr)[-1]);
    }
}
#endif

void* ac::core::fastMalloc(const std::size_t size) noexcept
{
#if defined(AC_CORE_HAVE_WIN32_ALIGNED_MALLOC)
    return _aligned_malloc(size, AC_CORE_MALLOC_ALIGN);
#elif defined(AC_CORE_HAVE_STD_ALIGNED_ALLOC)
    return std::aligned_alloc(AC_CORE_MALLOC_ALIGN, size);
#else
    return detail::alignedAlloc(size);
#endif
}

void ac::core::fastFree(void* const ptr) noexcept
{
#if defined(AC_CORE_HAVE_WIN32_ALIGNED_MALLOC)
    _aligned_free(ptr);
#elif defined(AC_CORE_HAVE_STD_ALIGNED_ALLOC)
    std::free(ptr);
#else
    detail::alignedFree(ptr);
#endif
}
