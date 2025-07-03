#if defined(AC_CORE_HAVE_STD_ALIGNED_ALLOC)
#   include <cstdlib>
#elif defined(AC_CORE_HAVE_WIN32_ALIGNED_MALLOC)
#   include <malloc.h>
#elif defined(AC_CORE_HAVE_POSIX_MEMALIGN)
#   include <stdlib.h>
#elif defined(AC_CORE_HAVE_BSD_MEMALIGN)
#   include <malloc.h>
#else
#   include <cstdlib>
#endif

#include "AC/Core/Util.hpp"

#ifndef AC_CORE_MALLOC_ALIGN
#   define AC_CORE_MALLOC_ALIGN 32
#endif

#if !defined(AC_CORE_HAVE_STD_ALIGNED_ALLOC) && \
    !defined(AC_CORE_HAVE_WIN32_ALIGNED_MALLOC) && \
    !defined(AC_CORE_HAVE_POSIX_MEMALIGN) && \
    !defined(AC_CORE_HAVE_BSD_MEMALIGN)
namespace ac::core::detail
{
    template <typename T>
    static inline constexpr T* alignPtr(T* const ptr, const int n) noexcept
    {
        return reinterpret_cast<T*>(align(reinterpret_cast<std::uintptr_t>(ptr), n));
    }

    static inline void* alignedAlloc(const std::size_t size, const int alignment) noexcept
    {
        auto buffer = static_cast<void**>(std::malloc(size + sizeof(void*) + alignment));
        if (!buffer) return nullptr;
        void** ptr = alignPtr(buffer + 1, alignment);
        ptr[-1] = static_cast<void*>(buffer);
        return ptr;
    }
    static inline void alignedFree(void* const ptr) noexcept
    {
        if (ptr) std::free(static_cast<void**>(ptr)[-1]);
    }
}
#endif

void* ac::core::fastMalloc(const std::size_t size) noexcept
{
    auto alignedSize = align(size, AC_CORE_MALLOC_ALIGN); // size must be an integral multiple of alignment.
#if defined(AC_CORE_HAVE_STD_ALIGNED_ALLOC)
    return std::aligned_alloc(AC_CORE_MALLOC_ALIGN, alignedSize);
#elif defined(AC_CORE_HAVE_WIN32_ALIGNED_MALLOC)
    return _aligned_malloc(alignedSize, AC_CORE_MALLOC_ALIGN);
#elif defined(AC_CORE_HAVE_POSIX_MEMALIGN)
    void* ptr = nullptr;
    return posix_memalign(&ptr, AC_CORE_MALLOC_ALIGN, alignedSize) ? nullptr : ptr;
#elif defined(AC_CORE_HAVE_BSD_MEMALIGN)
    return memalign(AC_CORE_MALLOC_ALIGN, alignedSize);
#else
    return detail::alignedAlloc(alignedSize, AC_CORE_MALLOC_ALIGN);
#endif
}

void ac::core::fastFree(void* const ptr) noexcept
{
#if defined(AC_CORE_HAVE_STD_ALIGNED_ALLOC)
    std::free(ptr);
#elif defined(AC_CORE_HAVE_WIN32_ALIGNED_MALLOC)
    _aligned_free(ptr);
#elif defined(AC_CORE_HAVE_POSIX_MEMALIGN) || defined(AC_CORE_HAVE_BSD_MEMALIGN)
    std::free(ptr);
#else
    detail::alignedFree(ptr);
#endif
}
