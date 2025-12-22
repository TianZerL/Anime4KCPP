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
    struct ResidualArg
    {
        const Image& image;
        float scale;
    };

    constexpr float identity(float v) noexcept;
    constexpr float relu(float v) noexcept;
    constexpr float lrelu(float v, float n) noexcept;

    class Identity
    {
    public:
        constexpr Identity() noexcept = default;
        float operator() (const float v) const noexcept { return identity(v); }
    };
    class ReLU
    {
    public:
        constexpr ReLU() noexcept = default;
        float operator() (const float v) const noexcept { return relu(v); }
    };
    class LReLU
    {
    public:
        constexpr LReLU(const float negativeSlope) noexcept : negativeSlope(negativeSlope) {}
        float operator() (const float v) const noexcept { return lrelu(v, negativeSlope); }

    private:
        const float negativeSlope;
    };

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

    /**
     * @brief Convert a value to float.
     * @param v Input float point value.
     * @return A float value.
     */
    template<typename Float, std::enable_if_t<std::is_floating_point_v<Float>, bool> = true>
    constexpr float toFloat(Float v) noexcept;
    /**
     * @brief Convert a value to normalized float.
     * @param v Input unsigned integer value.
     * @return A normalized float value.
     */
    template<typename Unsigned, std::enable_if_t<std::is_unsigned_v<Unsigned>, bool> = true>
    constexpr float toFloat(Unsigned v) noexcept;

    /**
     * @brief Convert a float value to `Float` type and clamp the value between 0.0f and 1.0f.
     * @param v Input float value.
     * @return A value between 0.0f and 1.0f.
     */
    template<typename Float, std::enable_if_t<std::is_floating_point_v<Float>, bool> = true>
    constexpr Float fromFloat(float v) noexcept;
    /**
     * @brief Convert a float value to `Unsigned` type and clamp the value between 0 and the max of `Unsigned`.
     * @param v Input float value.
     * @return A value between 0 and the max of `Unsigned`.
     */
    template<typename Unsigned, std::enable_if_t<std::is_unsigned_v<Unsigned>, bool> = true>
    constexpr Unsigned fromFloat(float v) noexcept;

    /**
     * @brief Compute `ceil(log2(v))` as fast as possible.
     * @param v Input.
     * @return result of `ceil(log2(v))`.
     */
    int ceilLog2(double v) noexcept;

    /**
     * @brief Apply a parallel pixel-wise operation to multiple images with integer scaling.
     *
     * This function processes multiple images in parallel, applying a user-defined operation
     * to each pixel position in the destination image(s). It supports integer upscaling from
     * source to destination images.
     *
     * This function operates on destination image dimensions (w x h). For each destination
     * pixel at (j, i), source pixel coordinates are computed as (j/scale, i/scale) where
     * `scale = dst.width() / src.width()`.
     * 
     * @tparam F Type of the callable operation function.
     * @tparam Images Variadic template parameter pack for image references.
     *
     * @param f Callable operation function that will be called for each pixel in the destination.
     * @param images Variadic list of image references, the first image must be `const Image&`
     *        (`src`) and last image must be `Image&` (`dst`), images in between can be
     *        either const (sources) or non-const (destinations)
     *
     * @note All images must be valid (non-empty) `ac::core::Image` (with const/ref qualifiers).
     * @note Processing is parallelized over rows.
     * @note The function assumes uniform integer scaling, the scale factor must be an integer > 0,
     *       and same in bothdimensions; non-integer scaling or different horizontal/vertical scaling 
     *       factors result in undefined behavior.
     */
    template<typename F, typename ...Images>
    auto filter(F&& f, Images& ...images) ->
        std::enable_if_t<(
            (sizeof...(Images) > 1) &&
            (std::is_const_v<std::tuple_element_t<0, std::tuple<Images...>>>) &&
            (!std::is_const_v<std::tuple_element_t<sizeof...(Images) - 1, std::tuple<Images...>>>) &&
            (std::is_same_v<ac::core::Image, std::remove_cv_t<Images>> && ...)),
        void>;

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

inline constexpr float ac::core::identity(const float v) noexcept
{
    return v;
}
inline constexpr float ac::core::relu(const float v) noexcept
{
    return v < 0.0f ? 0.0f : v;
}
inline constexpr float ac::core::lrelu(const float v, const float n) noexcept
{
    return v < v * n ? v * n : v;
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
    return v < 0.0f ? 0.0f : (v < 1.0f ? v : 1.0f);
}
template<typename Unsigned, std::enable_if_t<std::is_unsigned_v<Unsigned>, bool>>
inline constexpr Unsigned ac::core::fromFloat(const float v) noexcept
{
    return static_cast<Unsigned>(fromFloat<float>(v) * std::numeric_limits<Unsigned>::max() + 0.5f);
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
