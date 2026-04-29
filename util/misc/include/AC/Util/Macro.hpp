#ifndef AC_UTIL_MACRO_HPP
#define AC_UTIL_MACRO_HPP

#if defined(_MSC_VER)
#   define AC_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__INTEL_COMPILER)
#   define AC_FORCE_INLINE inline __attribute__((__always_inline__))
#elif defined(__CLANG__)
#   if __has_attribute(__always_inline__)
#       define AC_FORCE_INLINE inline __attribute__((__always_inline__))
#   else
#       define AC_FORCE_INLINE inline
#   endif
#else
#   define AC_FORCE_INLINE inline
#endif

#endif
