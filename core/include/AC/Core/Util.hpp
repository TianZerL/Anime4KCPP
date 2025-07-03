#ifndef AC_CORE_UTIL_HPP
#define AC_CORE_UTIL_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <type_traits>

#include "AC/Core/Image.hpp"
#include "AC/Util/Parallel.hpp"

namespace ac::core
{
    // align v to a multiple of n, and n must a power of 2
    template <typename Integer>
    constexpr Integer align(Integer v, int n) noexcept;

    // convert value to float
    template<typename Float, std::enable_if_t<std::is_floating_point_v<Float>, bool> = true>
    constexpr float toFloat(Float v) noexcept;
    // convert value to normalized float
    template<typename Unsigned, std::enable_if_t<std::is_unsigned_v<Unsigned>, bool> = true>
    constexpr float toFloat(Unsigned v) noexcept;

    // clamp value between 0.0f and 1.0f
    template<typename Float, std::enable_if_t<std::is_floating_point_v<Float>, bool> = true>
    constexpr Float fromFloat(float v) noexcept;
    // clamp value between 0 and Unsigned's max
    template<typename Unsigned, std::enable_if_t<std::is_unsigned_v<Unsigned>, bool> = true>
    constexpr Unsigned fromFloat(float v) noexcept;

    // clamp value between 0.0f and value
    template<typename Float, std::enable_if_t<std::is_floating_point_v<Float>, bool> = true>
    constexpr Float relu(float v) noexcept;
    // clamp value between 0 and Unsigned's max
    template<typename Unsigned, std::enable_if_t<std::is_unsigned_v<Unsigned>, bool> = true>
    constexpr Unsigned relu(float v) noexcept;

    // compute `ceil(log2(v))` as fast as possible
    int ceilLog2(double v) noexcept;

    // filter images
    // source images should be passed by `const Image&` and destination images by `Image&`.
    // all source images should have the same width and height, and so should all destination images.
    // the width and height of the destination images will be used for the filtering process.
    // the scale ratio from source images to destination images will be computed as `dst.width() / src.width()` and this ratio will be applied to both the width and height.
    // the scale ratio is assumed to be an integer, so size of destination image should an integral multiple of source image, ohterwise it's an undefined behavior.
    template<typename F, typename ...Images>
    auto filter(F&& f, Images& ...images) ->
        std::enable_if_t<(
            (sizeof...(Images) > 1) &&
            (std::is_const_v<std::tuple_element_t<0, std::tuple<Images...>>>) &&
            (!std::is_const_v<std::tuple_element_t<sizeof...(Images) - 1, std::tuple<Images...>>>) &&
            (std::is_same_v<ac::core::Image, std::remove_cv_t<Images>> && ...)),
        void>;

    void* fastMalloc(std::size_t size) noexcept;
    void fastFree(void* ptr) noexcept;
}

template <typename Integer>
inline constexpr Integer ac::core::align(const Integer v, const int n) noexcept
{
    return (v + n - 1) & -n;
}

template<typename Float, std::enable_if_t<std::is_floating_point_v<Float>, bool>>
inline constexpr float ac::core::toFloat(const Float v) noexcept
{
    return static_cast<float>(v);
}
template<typename Unsigned, std::enable_if_t<std::is_unsigned_v<Unsigned>, bool>>
inline constexpr float ac::core::toFloat(const Unsigned v) noexcept
{
    return static_cast<float>(v) / static_cast<float>(std::numeric_limits<Unsigned>::max());
}

template<typename Float, std::enable_if_t<std::is_floating_point_v<Float>, bool>>
inline constexpr Float ac::core::fromFloat(const float v) noexcept
{
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}
template<typename Unsigned, std::enable_if_t<std::is_unsigned_v<Unsigned>, bool>>
inline constexpr Unsigned ac::core::fromFloat(const float v) noexcept
{
    return v < 0.0f ? 0 : (v > 1.0f ? std::numeric_limits<Unsigned>::max() : static_cast<Unsigned>(v * std::numeric_limits<Unsigned>::max() + 0.5f));
}

template<typename Float, std::enable_if_t<std::is_floating_point_v<Float>, bool>>
inline constexpr Float ac::core::relu(const float v) noexcept
{
    return v < 0.0f ? 0.0f : v;
}
template<typename Unsigned, std::enable_if_t<std::is_unsigned_v<Unsigned>, bool>>
inline constexpr Unsigned ac::core::relu(const float v) noexcept
{
    return fromFloat<Unsigned>(v);
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

template<typename F, typename ...Images>
inline auto ac::core::filter(F&& f, Images& ...images) ->
    std::enable_if_t<(
        (sizeof...(Images) > 1) &&
        (std::is_const_v<std::tuple_element_t<0, std::tuple<Images...>>>) &&
        (!std::is_const_v<std::tuple_element_t<sizeof...(Images) - 1, std::tuple<Images...>>>) &&
        (std::is_same_v<ac::core::Image, std::remove_cv_t<Images>> && ...)),
    void>
{
    const auto& src = std::get<0>(std::forward_as_tuple(images...));
    const auto& dst = std::get<sizeof...(Images) - 1>(std::forward_as_tuple(images...));

    const int w = dst.width(), h = dst.height();
    const int scale = dst.width() / src.width();

    util::parallelFor(0, h,
        [&](const int i) {
            for (int j = 0; j < w; j++) f(i, j, (std::is_const_v<Images> ? images.ptr(j / scale, i / scale) : images.ptr(j, i))...);
        });
}

#endif
