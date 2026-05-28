#ifndef AC_CORE_UTIL_HPP
#define AC_CORE_UTIL_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace ac::core
{
    /**
     * @brief Convert a value to normalized float.
     * @param v Input value.
     * @return A normalized float value.
     */
    template<typename T>
    constexpr float toFloat(T v) noexcept;

    /**
     * @brief Clamp the float value between 0.0f and 1.0f, then convert it to type `T`, denormalizing if necessary.
     * @param v Input float value.
     * @return A unnormalized value.
     */
    template<typename T>
    constexpr T fromFloat(float v) noexcept;

    /**
     * @brief Compute `ceil(log2(v))` as fast as possible.
     * @param v Input.
     * @return result of `ceil(log2(v))`.
     */
    int ceilLog2(double v) noexcept;

    /**
     * @brief allocate memory with internal alignment.
     * @param size Size to allocate.
     * @return A pointer to allocated memory.
     */
    void* fastMalloc(std::size_t size) noexcept;
    /**
     * @brief deallocate memory witch allocated by `fastMalloc`.
     * @param ptr A pointer from `fastMalloc`.
     */
    void fastFree(void* ptr) noexcept;
}

template<typename T>
inline constexpr float ac::core::toFloat(const T v) noexcept
{
    if constexpr (std::is_unsigned_v<T>)
        return static_cast<float>(v) / static_cast<float>(std::numeric_limits<T>::max());
    else if constexpr (std::is_integral_v<T>)
    {
        constexpr float l = static_cast<float>(std::numeric_limits<T>::min());
        constexpr float r = static_cast<float>(std::numeric_limits<T>::max());
        return (static_cast<float>(v) - l) / (r - l);
    }
    else return static_cast<float>(v);
}

template<typename T>
inline constexpr T ac::core::fromFloat(const float v) noexcept
{
    float saturated = v < 0.0f ? 0.0f : (v < 1.0f ? v : 1.0f);
    if constexpr (std::is_unsigned_v<T>)
        return static_cast<T>(saturated * std::numeric_limits<T>::max() + 0.5f);
    else if constexpr (std::is_integral_v<T>)
    {
        constexpr float l = static_cast<float>(std::numeric_limits<T>::min());
        constexpr float r = static_cast<float>(std::numeric_limits<T>::max());
        float x = saturated * (r - l) + l;
        x += (x >= 0.0f) ? 0.5f : -0.5f;
        return static_cast<T>(x);
    }
    else return static_cast<T>(saturated);
}

inline int ac::core::ceilLog2(const double v) noexcept
{
    if constexpr (std::numeric_limits<double>::is_iec559 && (sizeof(double) == sizeof(std::uint64_t)))
    {
        std::uint64_t data{};
        std::memcpy(&data, &v, sizeof(double));
        return static_cast<int>((((data >> 52) & 0x7ff) - 1023) + ((data << 12) != 0));
    }
    else return static_cast<int>(std::ceil(std::log2(v)));
}

#endif
