#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "AC/Core/Image.hpp"
#include "AC/Core/Util.hpp"

namespace ac::core::detail
{
    template<typename IN, typename OUT = IN>
    static inline void rgb2yuv(const Image& src, Image& dst)
    {
        filter([](const int /*i*/, const int /*j*/, const void* const sptr, void* const dptr) {
            auto in = static_cast<const IN*>(sptr);
            auto out = static_cast<OUT*>(dptr);

            float r = toFloat(in[0]);
            float g = toFloat(in[1]);
            float b = toFloat(in[2]);

            float y = 0.299f * r + 0.587f * g + 0.114f * b;
            float u = 0.564f * (b - y) + 0.5f;
            float v = 0.713f * (r - y) + 0.5f;

            out[0] = fromFloat<OUT>(y);
            out[1] = fromFloat<OUT>(u);
            out[2] = fromFloat<OUT>(v);
        }, src, dst);
    }
    template<typename IN, typename OUT = IN>
    static inline void rgb2yuv(const Image& src, Image& dsty, Image& dstuv)
    {
        filter([](const int /*i*/, const int /*j*/, const void* const sptr, void* const yptr, void* const uvptr) {
            auto in = static_cast<const IN*>(sptr);
            auto yout = static_cast<OUT*>(yptr);
            auto uvout = static_cast<OUT*>(uvptr);

            float r = toFloat(in[0]);
            float g = toFloat(in[1]);
            float b = toFloat(in[2]);

            float y = 0.299f * r + 0.587f * g + 0.114f * b;
            float u = 0.564f * (b - y) + 0.5f;
            float v = 0.713f * (r - y) + 0.5f;

            yout[0] = fromFloat<OUT>(y);
            uvout[0] = fromFloat<OUT>(u);
            uvout[1] = fromFloat<OUT>(v);
        }, src, dsty, dstuv);
    }
    template<typename IN, typename OUT = IN>
    static inline void rgb2yuv(const Image& src, Image& dsty, Image& dstu, Image& dstv)
    {
        filter([](const int /*i*/, const int /*j*/, const void* const sptr, void* const yptr, void* const uptr, void* const vptr) {
            auto in = static_cast<const IN*>(sptr);
            auto yout = static_cast<OUT*>(yptr);
            auto uout = static_cast<OUT*>(uptr);
            auto vout = static_cast<OUT*>(vptr);

            float r = toFloat(in[0]);
            float g = toFloat(in[1]);
            float b = toFloat(in[2]);

            float y = 0.299f * r + 0.587f * g + 0.114f * b;
            float u = 0.564f * (b - y) + 0.5f;
            float v = 0.713f * (r - y) + 0.5f;

            *yout = fromFloat<OUT>(y);
            *uout = fromFloat<OUT>(u);
            *vout = fromFloat<OUT>(v);
        }, src, dsty, dstu, dstv);
    }

    template<typename IN, typename OUT = IN>
    static inline void rgba2yuva(const Image& src, Image& dst)
    {
        filter([](const int /*i*/, const int /*j*/, const void* const sptr, void* const dptr) {
            auto in = static_cast<const IN*>(sptr);
            auto out = static_cast<OUT*>(dptr);

            float a = toFloat(in[3]);
            float r = toFloat(in[0]) * a;
            float g = toFloat(in[1]) * a;
            float b = toFloat(in[2]) * a;

            float y = 0.299f * r + 0.587f * g + 0.114f * b;
            float u = 0.564f * (b - y) + 0.5f;
            float v = 0.713f * (r - y) + 0.5f;

            out[0] = fromFloat<OUT>(y);
            out[1] = fromFloat<OUT>(u);
            out[2] = fromFloat<OUT>(v);
            out[3] = fromFloat<OUT>(a);
        }, src, dst);
    }
    template<typename IN, typename OUT = IN>
    static inline void rgba2yuva(const Image& src, Image& dsty, Image& dstuva)
    {
        filter([](const int /*i*/, const int /*j*/, const void* const sptr, void* const yptr, void* const uvaptr) {
            auto in = static_cast<const IN*>(sptr);
            auto yout = static_cast<OUT*>(yptr);
            auto uvaout = static_cast<OUT*>(uvaptr);

            float a = toFloat(in[3]);
            float r = toFloat(in[0]) * a;
            float g = toFloat(in[1]) * a;
            float b = toFloat(in[2]) * a;

            float y = 0.299f * r + 0.587f * g + 0.114f * b;
            float u = 0.564f * (b - y) + 0.5f;
            float v = 0.713f * (r - y) + 0.5f;

            yout[0] = fromFloat<OUT>(y);
            uvaout[0] = fromFloat<OUT>(u);
            uvaout[1] = fromFloat<OUT>(v);
            uvaout[2] = fromFloat<OUT>(a);
        }, src, dsty, dstuva);
    }
    template<typename IN, typename OUT = IN>
    static inline void rgba2yuva(const Image& src, Image& dsty, Image& dstu, Image& dstv, Image& dsta)
    {
        filter([](const int /*i*/, const int /*j*/, const void* const sptr, void* const yptr, void* const uptr, void* const vptr, void* const aptr) {
            auto in = static_cast<const IN*>(sptr);
            auto yout = static_cast<OUT*>(yptr);
            auto uout = static_cast<OUT*>(uptr);
            auto vout = static_cast<OUT*>(vptr);
            auto aout = static_cast<OUT*>(aptr);

            float a = toFloat(in[3]);
            float r = toFloat(in[0]) * a;
            float g = toFloat(in[1]) * a;
            float b = toFloat(in[2]) * a;

            float y = 0.299f * r + 0.587f * g + 0.114f * b;
            float u = 0.564f * (b - y) + 0.5f;
            float v = 0.713f * (r - y) + 0.5f;

            *yout = fromFloat<OUT>(y);
            *uout = fromFloat<OUT>(u);
            *vout = fromFloat<OUT>(v);
            *aout = fromFloat<OUT>(a);
        }, src, dsty, dstu, dstv, dsta);
    }

    template<typename IN, typename OUT = IN>
    static inline void yuv2rgb(const Image& src, Image& dst)
    {
        filter([](const int /*i*/, const int /*j*/, const void* const sptr, void* const dptr) {
            auto in = static_cast<const IN*>(sptr);
            auto out = static_cast<OUT*>(dptr);

            float y = toFloat(in[0]);
            float u = toFloat(in[1]) - 0.5f;
            float v = toFloat(in[2]) - 0.5f;

            float r = y + 1.403f * v;
            float g = y - 0.344f * u - 0.714f * v;
            float b = y + 1.773f * u;

            out[0] = fromFloat<OUT>(r);
            out[1] = fromFloat<OUT>(g);
            out[2] = fromFloat<OUT>(b);
        }, src, dst);
    }
    template<typename IN, typename OUT = IN>
    static inline void yuv2rgb(const Image& srcy, const Image& srcuv, Image& dst)
    {
        filter([](const int /*i*/, const int /*j*/, const void* const yptr, const void* const uvptr, void* const dptr) {
            auto yin = static_cast<const IN*>(yptr);
            auto uvin = static_cast<const IN*>(uvptr);
            auto out = static_cast<OUT*>(dptr);

            float y = toFloat(yin[0]);
            float u = toFloat(uvin[0]) - 0.5f;
            float v = toFloat(uvin[1]) - 0.5f;

            float r = y + 1.403f * v;
            float g = y - 0.344f * u - 0.714f * v;
            float b = y + 1.773f * u;

            out[0] = fromFloat<OUT>(r);
            out[1] = fromFloat<OUT>(g);
            out[2] = fromFloat<OUT>(b);
        }, srcy, srcuv, dst);
    }
    template<typename IN, typename OUT = IN>
    static inline void yuv2rgb(const Image& srcy, const Image& srcu, const Image& srcv, Image& dst)
    {
        filter([](const int /*i*/, const int /*j*/, const void* const yptr, const void* const uptr, const void* const vptr, void* const dptr) {
            auto yin = static_cast<const IN*>(yptr);
            auto uin = static_cast<const IN*>(uptr);
            auto vin = static_cast<const IN*>(vptr);
            auto out = static_cast<OUT*>(dptr);

            float y = toFloat(*yin);
            float u = toFloat(*uin) - 0.5f;
            float v = toFloat(*vin) - 0.5f;

            float r = y + 1.403f * v;
            float g = y - 0.344f * u - 0.714f * v;
            float b = y + 1.773f * u;

            out[0] = fromFloat<OUT>(r);
            out[1] = fromFloat<OUT>(g);
            out[2] = fromFloat<OUT>(b);
        }, srcy, srcu, srcv, dst);
    }

    template<typename IN, typename OUT = IN>
    static inline void yuva2rgba(const Image& src, Image& dst)
    {
        filter([](const int /*i*/, const int /*j*/, const void* const sptr, void* const dptr) {
            auto in = static_cast<const IN*>(sptr);
            auto out = static_cast<OUT*>(dptr);

            float y = toFloat(in[0]);
            float u = toFloat(in[1]) - 0.5f;
            float v = toFloat(in[2]) - 0.5f;
            float a = toFloat(in[3]);

            float r = y + 1.403f * v;
            float g = y - 0.344f * u - 0.714f * v;
            float b = y + 1.773f * u;

            if (a > 1e-6f) // 1/65535 ~ 1e-5
            {
                r /= a;
                g /= a;
                b /= a;
            }
            else r = g = b = 0.0f;

            out[0] = fromFloat<OUT>(r);
            out[1] = fromFloat<OUT>(g);
            out[2] = fromFloat<OUT>(b);
            out[3] = fromFloat<OUT>(a);
        }, src, dst);
    }
    template<typename IN, typename OUT = IN>
    static inline void yuva2rgba(const Image& srcy, const Image& srcuva, Image& dst)
    {
        filter([](const int /*i*/, const int /*j*/, const void* const yptr, const void* const uvaptr, void* const dptr) {
            auto yin = static_cast<const IN*>(yptr);
            auto uvain = static_cast<const IN*>(uvaptr);
            auto out = static_cast<OUT*>(dptr);

            float y = toFloat(yin[0]);
            float u = toFloat(uvain[0]) - 0.5f;
            float v = toFloat(uvain[1]) - 0.5f;
            float a = toFloat(uvain[2]);

            float r = y + 1.403f * v;
            float g = y - 0.344f * u - 0.714f * v;
            float b = y + 1.773f * u;

            if (a > 1e-6f)
            {
                r /= a;
                g /= a;
                b /= a;
            }
            else r = g = b = 0.0f;

            out[0] = fromFloat<OUT>(r);
            out[1] = fromFloat<OUT>(g);
            out[2] = fromFloat<OUT>(b);
            out[3] = fromFloat<OUT>(a);
        }, srcy, srcuva, dst);
    }
    template<typename IN, typename OUT = IN>
    static inline void yuva2rgba(const Image& srcy, const Image& srcu, const Image& srcv, const Image& srca, Image& dst)
    {
        filter([](const int /*i*/, const int /*j*/, const void* const yptr, const void* const uptr, const void* const vptr, const void* const aptr, void* const dptr) {
            auto yin = static_cast<const IN*>(yptr);
            auto uin = static_cast<const IN*>(uptr);
            auto vin = static_cast<const IN*>(vptr);
            auto ain = static_cast<const IN*>(aptr);
            auto out = static_cast<OUT*>(dptr);

            float y = toFloat(*yin);
            float u = toFloat(*uin) - 0.5f;
            float v = toFloat(*vin) - 0.5f;
            float a = toFloat(*ain);

            float r = y + 1.403f * v;
            float g = y - 0.344f * u - 0.714f * v;
            float b = y + 1.773f * u;

            if (a > 1e-6f)
            {
                r /= a;
                g /= a;
                b /= a;
            }
            else r = g = b = 0.0f;

            out[0] = fromFloat<OUT>(r);
            out[1] = fromFloat<OUT>(g);
            out[2] = fromFloat<OUT>(b);
            out[3] = fromFloat<OUT>(a);
        }, srcy, srcu, srcv, srca, dst);
    }

    template<typename IN, typename OUT = IN, typename OP>
    static inline void elementwise(const Image& src, Image& dst, const int n, OP&& op) noexcept
    {
        for (int i = 0; i < src.height(); i++)
        {
            auto in = static_cast<const IN*>(src.ptr(i));
            auto out = static_cast<OUT*>(dst.ptr(i));
            for (int j = 0; j < src.width() * src.channels(); j++) *out++ = op(*in++, n);
        }
    }

    template<typename IN = void, typename OUT = IN>
    static inline void copy(const Image& src, Image& dst) noexcept
    {
        for (int i = 0; i < src.height(); i++)
        {
            auto in = static_cast<const IN*>(src.ptr(i));
            auto out = static_cast<OUT*>(dst.ptr(i));
            if constexpr (std::is_same_v<IN, OUT>)
            {
                auto lineSize = src.width() * src.pixelSize();
                std::memcpy(out, in, lineSize);
            }
            else
            {
                for (int j = 0; j < src.width() * src.channels(); j++)
                    *out++ = fromFloat<OUT>(toFloat(*in++));
            }
        }
    }
}

namespace ac::core::impl
{
    template <typename OP>
    void bitwise(Image& image, const int n, OP&& op) noexcept
    {
        if (image.empty() || image.isFloat() || n <= 0) return;
        switch (image.type())
        {
        case Image::UInt8: detail::elementwise<std::uint8_t>(image, image, n, op); break;
        case Image::UInt16: detail::elementwise<std::uint16_t>(image, image, n, op); break;
        }
    }
    template <typename OP>
    void bitwise(const Image& src, Image& dst, const int n, OP&& op) noexcept
    {
        if (src.empty() || src.isFloat() || n <= 0) return;
        Image tmp{};
        if (dst.empty() || (src == dst) || (dst.width() != src.width()) || (dst.height() != src.height()) || (dst.channels() != src.channels()) || (dst.type() != src.type()))
            tmp.create(src.width(), src.height(), src.channels(), src.type());
        else tmp = dst;
        switch (src.type())
        {
        case Image::UInt8: detail::elementwise<std::uint8_t>(src, tmp, n, op); break;
        case Image::UInt16: detail::elementwise<std::uint16_t>(src, tmp, n, op); break;
        }
        if (dst != tmp) dst = tmp;
    }
}

void ac::core::rgb2yuv(const ac::core::Image& rgb, ac::core::Image& yuv)
{
    if (rgb.empty()) return;
    if (yuv.empty()) yuv.create(rgb.width(), rgb.height(), 3, rgb.type());
    switch (rgb.type())
    {
    case Image::UInt8: return detail::rgb2yuv<std::uint8_t>(rgb, yuv);
    case Image::UInt16: return detail::rgb2yuv<std::uint16_t>(rgb, yuv);
    case Image::Float32: return detail::rgb2yuv<float>(rgb, yuv);
    }
}
void ac::core::rgb2yuv(const Image& rgb, Image& y, Image& uv)
{
    if (rgb.empty()) return;
    if (y.empty()) y.create(rgb.width(), rgb.height(), 1, rgb.type());
    if (uv.empty()) uv.create(rgb.width(), rgb.height(), 2, rgb.type());
    switch (rgb.type())
    {
    case Image::UInt8: return detail::rgb2yuv<std::uint8_t>(rgb, y, uv);
    case Image::UInt16: return detail::rgb2yuv<std::uint16_t>(rgb, y, uv);
    case Image::Float32: return detail::rgb2yuv<float>(rgb, y, uv);
    }
}
void ac::core::rgb2yuv(const Image& rgb, Image& y, Image& u, Image& v)
{
    if (rgb.empty()) return;
    if (y.empty()) y.create(rgb.width(), rgb.height(), 1, rgb.type());
    if (u.empty()) u.create(rgb.width(), rgb.height(), 1, rgb.type());
    if (v.empty()) v.create(rgb.width(), rgb.height(), 1, rgb.type());
    switch (rgb.type())
    {
    case Image::UInt8: return detail::rgb2yuv<std::uint8_t>(rgb, y, u, v);
    case Image::UInt16: return detail::rgb2yuv<std::uint16_t>(rgb, y, u, v);
    case Image::Float32: return detail::rgb2yuv<float>(rgb, y, u, v);
    }
}

void ac::core::rgba2yuva(const Image& rgba, Image& yuva)
{
    if (rgba.empty()) return;
    if (yuva.empty()) yuva.create(rgba.width(), rgba.height(), 4, rgba.type());
    switch (rgba.type())
    {
    case Image::UInt8: return detail::rgba2yuva<std::uint8_t>(rgba, yuva);
    case Image::UInt16: return detail::rgba2yuva<std::uint16_t>(rgba, yuva);
    case Image::Float32: return detail::rgba2yuva<float>(rgba, yuva);
    }
}
void ac::core::rgba2yuva(const Image& rgba, Image& y, Image& uva)
{
    if (rgba.empty()) return;
    if (y.empty()) y.create(rgba.width(), rgba.height(), 1, rgba.type());
    if (uva.empty()) uva.create(rgba.width(), rgba.height(), 3, rgba.type());
    switch (rgba.type())
    {
    case Image::UInt8: return detail::rgba2yuva<std::uint8_t>(rgba, y, uva);
    case Image::UInt16: return detail::rgba2yuva<std::uint16_t>(rgba, y, uva);
    case Image::Float32: return detail::rgba2yuva<float>(rgba, y, uva);
    }
}
void ac::core::rgba2yuva(const Image& rgba, Image& y, Image& u, Image& v, Image& a)
{
    if (rgba.empty()) return;
    if (y.empty()) y.create(rgba.width(), rgba.height(), 1, rgba.type());
    if (u.empty()) u.create(rgba.width(), rgba.height(), 1, rgba.type());
    if (v.empty()) v.create(rgba.width(), rgba.height(), 1, rgba.type());
    if (a.empty()) a.create(rgba.width(), rgba.height(), 1, rgba.type());
    switch (rgba.type())
    {
    case Image::UInt8: return detail::rgba2yuva<std::uint8_t>(rgba, y, u, v, a);
    case Image::UInt16: return detail::rgba2yuva<std::uint16_t>(rgba, y, u, v, a);
    case Image::Float32: return detail::rgba2yuva<float>(rgba, y, u, v, a);
    }
}

void ac::core::yuv2rgb(const ac::core::Image& yuv, ac::core::Image& rgb)
{
    if (yuv.empty()) return;
    if (rgb.empty()) rgb.create(yuv.width(), yuv.height(), 3, yuv.type());
    switch (yuv.type())
    {
    case Image::UInt8: return detail::yuv2rgb<std::uint8_t>(yuv, rgb);
    case Image::UInt16: return detail::yuv2rgb<std::uint16_t>(yuv, rgb);
    case Image::Float32: return detail::yuv2rgb<float>(yuv, rgb);
    }
}
void ac::core::yuv2rgb(const ac::core::Image& y, const ac::core::Image& uv, ac::core::Image& rgb)
{
    if (y.empty() || uv.empty()) return;
    if (rgb.empty()) rgb.create(y.width(), y.height(), 3, y.type());
    switch (y.type())
    {
    case Image::UInt8: return detail::yuv2rgb<std::uint8_t>(y, uv, rgb);
    case Image::UInt16: return detail::yuv2rgb<std::uint16_t>(y, uv, rgb);
    case Image::Float32: return detail::yuv2rgb<float>(y, uv, rgb);
    }
}
void ac::core::yuv2rgb(const Image& y, const Image& u, const Image& v, Image& rgb)
{
    if (y.empty() || u.empty() || v.empty()) return;
    if (rgb.empty()) rgb.create(y.width(), y.height(), 3, y.type());
    switch (y.type())
    {
    case Image::UInt8: return detail::yuv2rgb<std::uint8_t>(y, u, v, rgb);
    case Image::UInt16: return detail::yuv2rgb<std::uint16_t>(y, u, v, rgb);
    case Image::Float32: return detail::yuv2rgb<float>(y, u, v, rgb);
    }
}

void ac::core::yuva2rgba(const Image& yuva, Image& rgba)
{
    if (yuva.empty()) return;
    if (rgba.empty()) rgba.create(yuva.width(), yuva.height(), 4, yuva.type());
    switch (yuva.type())
    {
    case Image::UInt8: return detail::yuva2rgba<std::uint8_t>(yuva, rgba);
    case Image::UInt16: return detail::yuva2rgba<std::uint16_t>(yuva, rgba);
    case Image::Float32: return detail::yuva2rgba<float>(yuva, rgba);
    }
}
void ac::core::yuva2rgba(const Image& y, const Image& uva, Image& rgba)
{
    if (y.empty() || uva.empty()) return;
    if (rgba.empty()) rgba.create(y.width(), y.height(), 4, y.type());
    switch (y.type())
    {
    case Image::UInt8: return detail::yuva2rgba<std::uint8_t>(y, uva, rgba);
    case Image::UInt16: return detail::yuva2rgba<std::uint16_t>(y, uva, rgba);
    case Image::Float32: return detail::yuva2rgba<float>(y, uva, rgba);
    }
}
void ac::core::yuva2rgba(const Image& y, const Image& u, const Image& v, const Image& a, Image& rgba)
{
    if (y.empty() || u.empty() || v.empty() || a.empty()) return;
    if (rgba.empty()) rgba.create(y.width(), y.height(), 4, y.type());
    switch (y.type())
    {
    case Image::UInt8: return detail::yuva2rgba<std::uint8_t>(y, u, v, a, rgba);
    case Image::UInt16: return detail::yuva2rgba<std::uint16_t>(y, u, v, a, rgba);
    case Image::Float32: return detail::yuva2rgba<float>(y, u, v, a, rgba);
    }
}
ac::core::Image ac::core::unpadding(const Image& src) noexcept
{
    auto lineSize = src.width() * src.pixelSize();
    if (src.empty() || (src.stride() == lineSize)) return src;

    Image dst{ src.width(), src.height(), src.channels(), src.type(), lineSize };

    for (int i = 0; i < src.height(); i++) std::memcpy(dst.ptr(i), src.ptr(i), lineSize);

    return dst;
}
void ac::core::shl(Image& image, const int n) noexcept
{
    impl::bitwise(image, n, [](auto a, auto b) { return a << b; });
}
void ac::core::shl(const Image& src, Image& dst, const int n) noexcept
{
    impl::bitwise(src, dst, n, [](auto a, auto b) { return a << b; });
}
void ac::core::shr(Image& image, const int n) noexcept
{
    impl::bitwise(image, n, [](auto a, auto b) { return a >> b; });
}
void ac::core::shr(const Image& src, Image& dst, const int n) noexcept
{
    impl::bitwise(src, dst, n, [](auto a, auto b) { return a >> b; });
}

ac::core::Image ac::core::astype(const Image& src, const int type) noexcept
{
    if (src.empty() || src.type() == type) return src;

    Image dst{ src.width(), src.height(), src.channels(), type };

    if (src.type() == Image::UInt8 && dst.type() == Image::Float32)
        detail::copy<std::uint8_t, float>(src, dst);
    else if (src.type() == Image::Float32 && dst.type() == Image::UInt8)
        detail::copy<float, std::uint8_t>(src, dst);
    else if(src.type() == Image::UInt16 && dst.type() == Image::Float32)
        detail::copy<std::uint16_t, float>(src, dst);
    else if (src.type() == Image::Float32 && dst.type() == Image::UInt16)
        detail::copy<float, std::uint16_t>(src, dst);
    else if (src.type() == Image::UInt8 && dst.type() == Image::UInt16)
        detail::copy<std::uint8_t, std::uint16_t>(src, dst);
    else if (src.type() == Image::UInt16 && dst.type() == Image::UInt8)
        detail::copy<std::uint16_t, std::uint8_t>(src, dst);

    return dst;
}
void ac::core::copy(const Image& src, Image& dst) noexcept
{
    if (src.empty())
    {
        dst = src;
        return;
    }

    bool sameShape = (dst.width() == src.width()) && (dst.height() == src.height()) && (dst.channels() == src.channels());

    if ((dst.ptr() == src.ptr()) && (dst.type() == src.type()) && sameShape) return;

    Image tmp{};
    if (dst.empty() || (src == dst) || !sameShape)
        tmp.create(src.width(), src.height(), src.channels(), dst.empty() ? src.type() : dst.type());
    else tmp = dst;

    if (src.type() == tmp.type())
        detail::copy(src, tmp);
    else if (src.type() == Image::UInt8 && tmp.type() == Image::Float32)
        detail::copy<std::uint8_t, float>(src, tmp);
    else if (src.type() == Image::Float32 && tmp.type() == Image::UInt8)
        detail::copy<float, std::uint8_t>(src, tmp);
    else if (src.type() == Image::UInt16 && tmp.type() == Image::Float32)
        detail::copy<std::uint16_t, float>(src, tmp);
    else if (src.type() == Image::Float32 && tmp.type() == Image::UInt16)
        detail::copy<float, std::uint16_t>(src, tmp);
    else if (src.type() == Image::UInt8 && tmp.type() == Image::UInt16)
        detail::copy<std::uint8_t, std::uint16_t>(src, tmp);

    if (dst != tmp) dst = tmp;
}

ac::core::Image ac::core::crop(const Image& src, const int x, const int y, const int w, const int h) noexcept
{
    if (src.empty()) return src;

    int rectX = w < 0 ? x + w : x;
    int rectY = h < 0 ? y + h : y;
    int rectW = std::abs(w);
    int rectH = std::abs(h);

    return src.view(rectX, rectY, rectW, rectH);
}

ac::core::Image ac::core::extract(const Image& src, const int channel, const int n) noexcept
{
    if (src.empty() || channel >= src.channels() || channel < 0 || n <= 0) return Image{};
    if (channel == 0 && n >= src.channels()) return src;

    ac::core::Image dst{ src.width(), src.height(), std::min(src.channels() - channel, n), src.type() };

    for (int i = 0; i < dst.height(); i++)
        for (int j = 0; j < dst.width(); j++)
            std::memcpy(dst.pixel(j, i), src.pixel(j, i) + channel * src.elementSize(), dst.pixelSize());

    return dst;
}

ac::core::Image ac::core::insert(const Image& src, const Image& image, const int channel) noexcept
{
    if (src.empty() || image.empty() || channel > src.channels() || channel < 0 ||
        image.width() != src.width() || image.height() != src.height() || image.type() != src.type())
        return src;

    ac::core::Image dst{ src.width(), src.height(), src.channels() + image.channels(), src.type()};

    auto step1 = channel * src.elementSize();
    auto step2 = image.pixelSize();
    auto step3 = src.pixelSize() - step1;

    for (int i = 0; i < dst.height(); i++)
        for (int j = 0; j < dst.width(); j++)
        {
            auto dp = dst.pixel(j, i);
            auto sp = src.pixel(j, i);
            auto ip = image.pixel(j, i);

            if (step1 > 0)
            {
                std::memcpy(dp, sp, step1);
                dp += step1;
                sp += step1;
            }

            std::memcpy(dp, ip, step2);
            dp += step2;

            if (step3 > 0) std::memcpy(dp, sp, step3);
        }

    return dst;
}
