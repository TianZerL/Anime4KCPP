#ifndef AC_UTIL_MISC_HPP
#define AC_UTIL_MISC_HPP

namespace ac::util
{
    /**
     * @brief Align a value to a multiple of a power-of-two boundary.
     *
     * This function rounds up the input value `v` to the nearest multiple of `n`.
     * The alignment boundary `n` must be a power of two (1, 2, 4, 8, 16, ...).
     *
     * @tparam Integer Integer type (e.g., int, size_t, uint32_t).
     * @param v Integer value to align.
     * @param n Alignment boundary (must be a power of two).
     * @return Integer Aligned value (multiple of n).
     */
    template <typename Integer>
    constexpr Integer align(Integer v, int n) noexcept;
}

template <typename Integer>
inline constexpr Integer ac::util::align(const Integer v, const int n) noexcept
{
    return (v + n - 1) & -n;
}

#endif
