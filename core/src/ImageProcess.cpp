#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>

#include "AC/Core/Image.hpp"
#include "AC/Util/Parallel.hpp"

#include "AC/Core/Internal/DataType.hpp"
#include "AC/Core/Internal/Util.hpp"

namespace ac::core::detail
{
    template<typename IN, typename OUT = IN>
    static inline void rgb2yuv(const Image& src, Image& dst)
    {
        util::parallelFor(0, src.height(), [&](const int i) {
            for (int j = 0; j < src.width(); j++)
            {
                auto in = static_cast<const IN*>(src.ptr(j, i));
                auto out = static_cast<OUT*>(dst.ptr(j, i));

                float r = toFloat(in[0]);
                float g = toFloat(in[1]);
                float b = toFloat(in[2]);

                float y = 0.299f * r + 0.587f * g + 0.114f * b;
                float u = 0.564f * (b - y) + 0.5f;
                float v = 0.713f * (r - y) + 0.5f;

                out[0] = fromFloat<OUT>(y);
                out[1] = fromFloat<OUT>(u);
                out[2] = fromFloat<OUT>(v);
            }
        });
    }
    template<typename IN, typename OUT = IN>
    static inline void rgb2yuv(const Image& src, Image& dsty, Image& dstuv)
    {
        util::parallelFor(0, src.height(), [&](const int i) {
            for (int j = 0; j < src.width(); j++)
            {
                auto in = static_cast<const IN*>(src.ptr(j, i));
                auto yout = static_cast<OUT*>(dsty.ptr(j, i));
                auto uvout = static_cast<OUT*>(dstuv.ptr(j, i));

                float r = toFloat(in[0]);
                float g = toFloat(in[1]);
                float b = toFloat(in[2]);

                float y = 0.299f * r + 0.587f * g + 0.114f * b;
                float u = 0.564f * (b - y) + 0.5f;
                float v = 0.713f * (r - y) + 0.5f;

                yout[0] = fromFloat<OUT>(y);
                uvout[0] = fromFloat<OUT>(u);
                uvout[1] = fromFloat<OUT>(v);
            }
        });
    }
    template<typename IN, typename OUT = IN>
    static inline void rgb2yuv(const Image& src, Image& dsty, Image& dstu, Image& dstv)
    {
        util::parallelFor(0, src.height(), [&](const int i) {
            for (int j = 0; j < src.width(); j++)
            {
                auto in = static_cast<const IN*>(src.ptr(j, i));
                auto yout = static_cast<OUT*>(dsty.ptr(j, i));
                auto uout = static_cast<OUT*>(dstu.ptr(j, i));
                auto vout = static_cast<OUT*>(dstv.ptr(j, i));

                float r = toFloat(in[0]);
                float g = toFloat(in[1]);
                float b = toFloat(in[2]);

                float y = 0.299f * r + 0.587f * g + 0.114f * b;
                float u = 0.564f * (b - y) + 0.5f;
                float v = 0.713f * (r - y) + 0.5f;

                *yout = fromFloat<OUT>(y);
                *uout = fromFloat<OUT>(u);
                *vout = fromFloat<OUT>(v);
            }
        });
    }

    template<typename IN, typename OUT = IN>
    static inline void rgba2yuva(const Image& src, Image& dst)
    {
        util::parallelFor(0, src.height(), [&](const int i) {
            for (int j = 0; j < src.width(); j++)
            {
                auto in = static_cast<const IN*>(src.ptr(j, i));
                auto out = static_cast<OUT*>(dst.ptr(j, i));

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
            }
        });
    }
    template<typename IN, typename OUT = IN>
    static inline void rgba2yuva(const Image& src, Image& dsty, Image& dstuva)
    {
        util::parallelFor(0, src.height(), [&](const int i) {
            for (int j = 0; j < src.width(); j++)
            {
                auto in = static_cast<const IN*>(src.ptr(j, i));
                auto yout = static_cast<OUT*>(dsty.ptr(j, i));
                auto uvaout = static_cast<OUT*>(dstuva.ptr(j, i));

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
            }
        });
    }
    template<typename IN, typename OUT = IN>
    static inline void rgba2yuva(const Image& src, Image& dsty, Image& dstu, Image& dstv, Image& dsta)
    {
        util::parallelFor(0, src.height(), [&](const int i) {
            for (int j = 0; j < src.width(); j++)
            {
                auto in = static_cast<const IN*>(src.ptr(j, i));
                auto yout = static_cast<OUT*>(dsty.ptr(j, i));
                auto uout = static_cast<OUT*>(dstu.ptr(j, i));
                auto vout = static_cast<OUT*>(dstv.ptr(j, i));
                auto aout = static_cast<OUT*>(dsta.ptr(j, i));

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
            }
        });
    }

    template<typename IN, typename OUT = IN>
    static inline void yuv2rgb(const Image& src, Image& dst)
    {
        util::parallelFor(0, src.height(), [&](const int i) {
            for (int j = 0; j < src.width(); j++)
            {
                auto in = static_cast<const IN*>(src.ptr(j, i));
                auto out = static_cast<OUT*>(dst.ptr(j, i));

                float y = toFloat(in[0]);
                float u = toFloat(in[1]) - 0.5f;
                float v = toFloat(in[2]) - 0.5f;

                float r = y + 1.403f * v;
                float g = y - 0.344f * u - 0.714f * v;
                float b = y + 1.773f * u;

                out[0] = fromFloat<OUT>(r);
                out[1] = fromFloat<OUT>(g);
                out[2] = fromFloat<OUT>(b);
            }
        });
    }
    template<typename IN, typename OUT = IN>
    static inline void yuv2rgb(const Image& srcy, const Image& srcuv, Image& dst)
    {
        const int w = srcy.width();
        util::parallelFor(0, srcy.height(), [&](const int i) {
            for (int j = 0; j < w; j++)
            {
                auto yin = static_cast<const IN*>(srcy.ptr(j, i));
                auto uvin = static_cast<const IN*>(srcuv.ptr(j, i));
                auto out = static_cast<OUT*>(dst.ptr(j, i));

                float y = toFloat(yin[0]);
                float u = toFloat(uvin[0]) - 0.5f;
                float v = toFloat(uvin[1]) - 0.5f;

                float r = y + 1.403f * v;
                float g = y - 0.344f * u - 0.714f * v;
                float b = y + 1.773f * u;

                out[0] = fromFloat<OUT>(r);
                out[1] = fromFloat<OUT>(g);
                out[2] = fromFloat<OUT>(b);
            }
        });
    }
    template<typename IN, typename OUT = IN>
    static inline void yuv2rgb(const Image& srcy, const Image& srcu, const Image& srcv, Image& dst)
    {
        util::parallelFor(0, srcy.height(), [&](const int i) {
            for (int j = 0; j < srcy.width(); j++)
            {
                auto yin = static_cast<const IN*>(srcy.ptr(j, i));
                auto uin = static_cast<const IN*>(srcu.ptr(j, i));
                auto vin = static_cast<const IN*>(srcv.ptr(j, i));
                auto out = static_cast<OUT*>(dst.ptr(j, i));

                float y = toFloat(*yin);
                float u = toFloat(*uin) - 0.5f;
                float v = toFloat(*vin) - 0.5f;

                float r = y + 1.403f * v;
                float g = y - 0.344f * u - 0.714f * v;
                float b = y + 1.773f * u;

                out[0] = fromFloat<OUT>(r);
                out[1] = fromFloat<OUT>(g);
                out[2] = fromFloat<OUT>(b);
            }
        });
    }

    template<typename IN, typename OUT = IN>
    static inline void yuva2rgba(const Image& src, Image& dst)
    {
        util::parallelFor(0, src.height(), [&](const int i) {
            for (int j = 0; j < src.width(); j++)
            {
                auto in = static_cast<const IN*>(src.ptr(j, i));
                auto out = static_cast<OUT*>(dst.ptr(j, i));

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
            }
        });
    }
    template<typename IN, typename OUT = IN>
    static inline void yuva2rgba(const Image& srcy, const Image& srcuva, Image& dst)
    {
        util::parallelFor(0, srcy.height(), [&](const int i) {
            for (int j = 0; j < srcy.width(); j++)
            {
                auto yin = static_cast<const IN*>(srcy.ptr(j, i));
                auto uvain = static_cast<const IN*>(srcuva.ptr(j, i));
                auto out = static_cast<OUT*>(dst.ptr(j, i));

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
            }
        });
    }
    template<typename IN, typename OUT = IN>
    static inline void yuva2rgba(const Image& srcy, const Image& srcu, const Image& srcv, const Image& srca, Image& dst)
    {
        util::parallelFor(0, srcy.height(), [&](const int i) {
            for (int j = 0; j < srcy.width(); j++)
            {
                auto yin = static_cast<const IN*>(srcy.ptr(j, i));
                auto uin = static_cast<const IN*>(srcu.ptr(j, i));
                auto vin = static_cast<const IN*>(srcv.ptr(j, i));
                auto ain = static_cast<const IN*>(srca.ptr(j, i));
                auto out = static_cast<OUT*>(dst.ptr(j, i));

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
            }
        });
    }

    template<typename IN, typename OUT = IN, typename OP>
    static inline void elementwise(const Image& src, Image& dst, OP&& op) noexcept
    {
        for (int i = 0; i < src.height(); i++)
        {
            auto in = static_cast<const IN*>(src.ptr(i));
            auto out = static_cast<OUT*>(dst.ptr(i));
            for (int j = 0; j < src.width() * src.channels(); j++) *out++ = op(*in++);
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

    template <typename IN, typename OUT>
    static inline void pixelshuffle(const Image& src, Image& dst, const int upscale) noexcept
    {
        int group = upscale * upscale;

        for (int i = 0; i < src.height(); i++)
        {
            for (int j = 0; j < src.width(); j++)
            {
                auto dstY = i * upscale;
                auto dstX = j * upscale;

                auto in = static_cast<const IN*>(src.ptr(j, i));

                for (int p = 0; p < group; p++)
                {
                    auto out = static_cast<OUT*>(dst.ptr(dstX + (p % upscale), dstY + (p / upscale)));
                    for (int n = 0; n < dst.channels(); n++)
                    {
                        if constexpr (std::is_same_v<IN, OUT> && std::is_integral_v<IN>)
                            out[n] = in[n * group + p];
                        else
                            out[n] = fromFloat<OUT>(toFloat(in[n * group + p]));
                    }
                }
            }
        }
    }
}

namespace ac::core::impl
{
    template <typename OP>
    void elementwise(Image& image, OP&& op) noexcept
    {
        if (image.empty() || image.isFloat()) return;
        switch (image.type())
        {
        case Image::UInt8: detail::elementwise<DataType::UInt8>(image, image, std::forward<OP>(op)); break;
        case Image::UInt16: detail::elementwise<DataType::UInt16>(image, image, std::forward<OP>(op)); break;
        }
    }
    template <typename OP>
    void elementwise(const Image& src, Image& dst, OP&& op) noexcept
    {
        if (src.empty() || src.isFloat()) return;
        Image tmp{};
        if (dst.empty() || (src == dst) || (dst.width() != src.width()) || (dst.height() != src.height()) || (dst.channels() != src.channels()) || (dst.type() != src.type()))
            tmp.create(src.width(), src.height(), src.channels(), src.type());
        else tmp = dst;
        switch (src.type())
        {
        case Image::UInt8: detail::elementwise<DataType::UInt8>(src, tmp, std::forward<OP>(op)); break;
        case Image::UInt16: detail::elementwise<DataType::UInt16>(src, tmp, std::forward<OP>(op)); break;
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
    case Image::UInt8: return detail::rgb2yuv<DataType::UInt8>(rgb, yuv);
    case Image::UInt16: return detail::rgb2yuv<DataType::UInt16>(rgb, yuv);
    case Image::Float16: return detail::rgb2yuv<DataType::Float16>(rgb, yuv);
    case Image::Float32: return detail::rgb2yuv<DataType::Float32>(rgb, yuv);
    }
}
void ac::core::rgb2yuv(const Image& rgb, Image& y, Image& uv)
{
    if (rgb.empty()) return;
    if (y.empty()) y.create(rgb.width(), rgb.height(), 1, rgb.type());
    if (uv.empty()) uv.create(rgb.width(), rgb.height(), 2, rgb.type());
    switch (rgb.type())
    {
    case Image::UInt8: return detail::rgb2yuv<DataType::UInt8>(rgb, y, uv);
    case Image::UInt16: return detail::rgb2yuv<DataType::UInt16>(rgb, y, uv);
    case Image::Float16: return detail::rgb2yuv<DataType::Float16>(rgb, y, uv);
    case Image::Float32: return detail::rgb2yuv<DataType::Float32>(rgb, y, uv);
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
    case Image::UInt8: return detail::rgb2yuv<DataType::UInt8>(rgb, y, u, v);
    case Image::UInt16: return detail::rgb2yuv<DataType::UInt16>(rgb, y, u, v);
    case Image::Float16: return detail::rgb2yuv<DataType::Float16>(rgb, y, u, v);
    case Image::Float32: return detail::rgb2yuv<DataType::Float32>(rgb, y, u, v);
    }
}

void ac::core::rgba2yuva(const Image& rgba, Image& yuva)
{
    if (rgba.empty()) return;
    if (yuva.empty()) yuva.create(rgba.width(), rgba.height(), 4, rgba.type());
    switch (rgba.type())
    {
    case Image::UInt8: return detail::rgba2yuva<DataType::UInt8>(rgba, yuva);
    case Image::UInt16: return detail::rgba2yuva<DataType::UInt16>(rgba, yuva);
    case Image::Float16: return detail::rgba2yuva<DataType::Float16>(rgba, yuva);
    case Image::Float32: return detail::rgba2yuva<DataType::Float32>(rgba, yuva);
    }
}
void ac::core::rgba2yuva(const Image& rgba, Image& y, Image& uva)
{
    if (rgba.empty()) return;
    if (y.empty()) y.create(rgba.width(), rgba.height(), 1, rgba.type());
    if (uva.empty()) uva.create(rgba.width(), rgba.height(), 3, rgba.type());
    switch (rgba.type())
    {
    case Image::UInt8: return detail::rgba2yuva<DataType::UInt8>(rgba, y, uva);
    case Image::UInt16: return detail::rgba2yuva<DataType::UInt16>(rgba, y, uva);
    case Image::Float16: return detail::rgba2yuva<DataType::Float16>(rgba, y, uva);
    case Image::Float32: return detail::rgba2yuva<DataType::Float32>(rgba, y, uva);
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
    case Image::UInt8: return detail::rgba2yuva<DataType::UInt8>(rgba, y, u, v, a);
    case Image::UInt16: return detail::rgba2yuva<DataType::UInt16>(rgba, y, u, v, a);
    case Image::Float16: return detail::rgba2yuva<DataType::Float16>(rgba, y, u, v, a);
    case Image::Float32: return detail::rgba2yuva<DataType::Float32>(rgba, y, u, v, a);
    }
}

void ac::core::yuv2rgb(const ac::core::Image& yuv, ac::core::Image& rgb)
{
    if (yuv.empty()) return;
    if (rgb.empty()) rgb.create(yuv.width(), yuv.height(), 3, yuv.type());
    switch (yuv.type())
    {
    case Image::UInt8: return detail::yuv2rgb<DataType::UInt8>(yuv, rgb);
    case Image::UInt16: return detail::yuv2rgb<DataType::UInt16>(yuv, rgb);
    case Image::Float16: return detail::yuv2rgb<DataType::Float16>(yuv, rgb);
    case Image::Float32: return detail::yuv2rgb<DataType::Float32>(yuv, rgb);
    }
}
void ac::core::yuv2rgb(const ac::core::Image& y, const ac::core::Image& uv, ac::core::Image& rgb)
{
    if (y.empty() || uv.empty()) return;
    if (rgb.empty()) rgb.create(y.width(), y.height(), 3, y.type());
    switch (y.type())
    {
    case Image::UInt8: return detail::yuv2rgb<DataType::UInt8>(y, uv, rgb);
    case Image::UInt16: return detail::yuv2rgb<DataType::UInt16>(y, uv, rgb);
    case Image::Float16: return detail::yuv2rgb<DataType::Float16>(y, uv, rgb);
    case Image::Float32: return detail::yuv2rgb<DataType::Float32>(y, uv, rgb);
    }
}
void ac::core::yuv2rgb(const Image& y, const Image& u, const Image& v, Image& rgb)
{
    if (y.empty() || u.empty() || v.empty()) return;
    if (rgb.empty()) rgb.create(y.width(), y.height(), 3, y.type());
    switch (y.type())
    {
    case Image::UInt8: return detail::yuv2rgb<DataType::UInt8>(y, u, v, rgb);
    case Image::UInt16: return detail::yuv2rgb<DataType::UInt16>(y, u, v, rgb);
    case Image::Float16: return detail::yuv2rgb<DataType::Float16>(y, u, v, rgb);
    case Image::Float32: return detail::yuv2rgb<DataType::Float32>(y, u, v, rgb);
    }
}

void ac::core::yuva2rgba(const Image& yuva, Image& rgba)
{
    if (yuva.empty()) return;
    if (rgba.empty()) rgba.create(yuva.width(), yuva.height(), 4, yuva.type());
    switch (yuva.type())
    {
    case Image::UInt8: return detail::yuva2rgba<DataType::UInt8>(yuva, rgba);
    case Image::UInt16: return detail::yuva2rgba<DataType::UInt16>(yuva, rgba);
    case Image::Float16: return detail::yuva2rgba<DataType::Float16>(yuva, rgba);
    case Image::Float32: return detail::yuva2rgba<DataType::Float32>(yuva, rgba);
    }
}
void ac::core::yuva2rgba(const Image& y, const Image& uva, Image& rgba)
{
    if (y.empty() || uva.empty()) return;
    if (rgba.empty()) rgba.create(y.width(), y.height(), 4, y.type());
    switch (y.type())
    {
    case Image::UInt8: return detail::yuva2rgba<DataType::UInt8>(y, uva, rgba);
    case Image::UInt16: return detail::yuva2rgba<DataType::UInt16>(y, uva, rgba);
    case Image::Float16: return detail::yuva2rgba<DataType::Float16>(y, uva, rgba);
    case Image::Float32: return detail::yuva2rgba<DataType::Float32>(y, uva, rgba);
    }
}
void ac::core::yuva2rgba(const Image& y, const Image& u, const Image& v, const Image& a, Image& rgba)
{
    if (y.empty() || u.empty() || v.empty() || a.empty()) return;
    if (rgba.empty()) rgba.create(y.width(), y.height(), 4, y.type());
    switch (y.type())
    {
    case Image::UInt8: return detail::yuva2rgba<DataType::UInt8>(y, u, v, a, rgba);
    case Image::UInt16: return detail::yuva2rgba<DataType::UInt16>(y, u, v, a, rgba);
    case Image::Float16: return detail::yuva2rgba<DataType::Float16>(y, u, v, a, rgba);
    case Image::Float32: return detail::yuva2rgba<DataType::Float32>(y, u, v, a, rgba);
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
    impl::elementwise(image, [=](auto a) { return a << n; });
}
void ac::core::shl(const Image& src, Image& dst, const int n) noexcept
{
    impl::elementwise(src, dst, [=](auto a) { return a << n; });
}
void ac::core::shr(Image& image, const int n) noexcept
{
    impl::elementwise(image, [=](auto a) { return a >> n; });
}
void ac::core::shr(const Image& src, Image& dst, const int n) noexcept
{
    impl::elementwise(src, dst, [=](auto a) { return a >> n; });
}

ac::core::Image ac::core::astype(const Image& src, const int type) noexcept
{
    if (src.empty() || src.type() == type) return src;

    Image dst{ src.width(), src.height(), src.channels(), type };

    if (src.type() == Image::UInt8 && dst.type() == Image::Float32)
        detail::copy<DataType::UInt8, DataType::Float32>(src, dst);
    else if (src.type() == Image::Float32 && dst.type() == Image::UInt8)
        detail::copy<DataType::Float32, DataType::UInt8>(src, dst);
    else if(src.type() == Image::UInt16 && dst.type() == Image::Float32)
        detail::copy<DataType::UInt16, DataType::Float32>(src, dst);
    else if (src.type() == Image::Float32 && dst.type() == Image::UInt16)
        detail::copy<DataType::Float32, DataType::UInt16>(src, dst);
    else if (src.type() == Image::UInt8 && dst.type() == Image::UInt16)
        detail::copy<DataType::UInt8, DataType::UInt16>(src, dst);
    else if (src.type() == Image::UInt16 && dst.type() == Image::UInt8)
        detail::copy<DataType::UInt16, DataType::UInt8>(src, dst);
    else if (src.type() == Image::Float16 && dst.type() == Image::Float32)
        detail::copy<DataType::Float16, DataType::Float32>(src, dst);
    else if (src.type() == Image::Float32 && dst.type() == Image::Float16)
        detail::copy<DataType::Float32, DataType::Float16>(src, dst);
    else if (src.type() == Image::UInt8 && dst.type() == Image::Float16)
        detail::copy<DataType::UInt8, DataType::Float16>(src, dst);
    else if (src.type() == Image::Float16 && dst.type() == Image::UInt8)
        detail::copy<DataType::Float16, DataType::UInt8>(src, dst);
    else if (src.type() == Image::UInt16 && dst.type() == Image::Float16)
        detail::copy<DataType::UInt16, DataType::Float16>(src, dst);
    else if (src.type() == Image::Float16 && dst.type() == Image::UInt16)
        detail::copy<DataType::Float16, DataType::UInt16>(src, dst);
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
        detail::copy<DataType::UInt8, DataType::Float32>(src, tmp);
    else if (src.type() == Image::Float32 && tmp.type() == Image::UInt8)
        detail::copy<DataType::Float32, DataType::UInt8>(src, tmp);
    else if (src.type() == Image::UInt16 && tmp.type() == Image::Float32)
        detail::copy<DataType::UInt16, DataType::Float32>(src, tmp);
    else if (src.type() == Image::Float32 && tmp.type() == Image::UInt16)
        detail::copy<DataType::Float32, DataType::UInt16>(src, tmp);
    else if (src.type() == Image::UInt8 && tmp.type() == Image::UInt16)
        detail::copy<DataType::UInt8, DataType::UInt16>(src, tmp);
    else if (src.type() == Image::Float16 && tmp.type() == Image::Float32)
        detail::copy<DataType::Float16, DataType::Float32>(src, tmp);
    else if (src.type() == Image::Float32 && tmp.type() == Image::Float16)
        detail::copy<DataType::Float32, DataType::Float16>(src, tmp);
    else if (src.type() == Image::UInt8 && tmp.type() == Image::Float16)
        detail::copy<DataType::UInt8, DataType::Float16>(src, tmp);
    else if (src.type() == Image::Float16 && tmp.type() == Image::UInt8)
        detail::copy<DataType::Float16, DataType::UInt8>(src, tmp);
    else if (src.type() == Image::UInt16 && tmp.type() == Image::Float16)
        detail::copy<DataType::UInt16, DataType::Float16>(src, tmp);
    else if (src.type() == Image::Float16 && tmp.type() == Image::UInt16)
        detail::copy<DataType::Float16, DataType::UInt16>(src, tmp);

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

void ac::core::pixelShuffle(const Image& src, Image& dst, const int upscale) noexcept
{
    int group = upscale * upscale;

    if (src.empty() || group <= 0 || (src.channels() % (group))) return;

    if (dst.empty() || (dst.width() != src.width() * upscale) || (dst.height() != src.height() * upscale) || (dst.channels() != src.channels() / group))
        dst.create(src.width() * upscale, src.height() * upscale, src.channels() / group, src.type());

    if (src.type() == Image::Float32 && dst.type() == Image::Float32)
        detail::pixelshuffle<DataType::Float32, DataType::Float32>(src, dst, upscale);
    else if (src.type() == Image::Float32 && dst.type() == Image::UInt8)
        detail::pixelshuffle<DataType::Float32, DataType::UInt8>(src, dst, upscale);
    else if (src.type() == Image::Float32 && dst.type() == Image::UInt16)
        detail::pixelshuffle<DataType::Float32, DataType::UInt16>(src, dst, upscale);
    else if (src.type() == Image::Float32 && dst.type() == Image::Float16)
        detail::pixelshuffle<DataType::Float32, DataType::Float16>(src, dst, upscale);
    else if (src.type() == Image::UInt8 && dst.type() == Image::UInt8)
        detail::pixelshuffle<DataType::UInt8, DataType::UInt8>(src, dst, upscale);
    else if (src.type() == Image::UInt8 && dst.type() == Image::Float32)
        detail::pixelshuffle<DataType::UInt8, DataType::Float32>(src, dst, upscale);
    else if (src.type() == Image::UInt8 && dst.type() == Image::UInt16)
        detail::pixelshuffle<DataType::UInt8, DataType::UInt16>(src, dst, upscale);
    else if (src.type() == Image::UInt8 && dst.type() == Image::Float16)
        detail::pixelshuffle<DataType::UInt8, DataType::Float16>(src, dst, upscale);
    else if (src.type() == Image::UInt16 && dst.type() == Image::UInt16)
        detail::pixelshuffle<DataType::UInt16, DataType::UInt16>(src, dst, upscale);
    else if (src.type() == Image::UInt16 && dst.type() == Image::Float32)
        detail::pixelshuffle<DataType::UInt16, DataType::Float32>(src, dst, upscale);
    else if (src.type() == Image::UInt16 && dst.type() == Image::UInt8)
        detail::pixelshuffle<DataType::UInt16, DataType::UInt8>(src, dst, upscale);
    else if (src.type() == Image::UInt16 && dst.type() == Image::Float16)
        detail::pixelshuffle<DataType::UInt16, DataType::Float16>(src, dst, upscale);
    else if (src.type() == Image::Float16 && dst.type() == Image::Float16)
        detail::pixelshuffle<DataType::Float16, DataType::Float16>(src, dst, upscale);
    else if (src.type() == Image::Float16 && dst.type() == Image::Float32)
        detail::pixelshuffle<DataType::Float16, DataType::Float32>(src, dst, upscale);
    else if (src.type() == Image::Float16 && dst.type() == Image::UInt8)
        detail::pixelshuffle<DataType::Float16, DataType::UInt8>(src, dst, upscale);
    else if (src.type() == Image::Float16 && dst.type() == Image::UInt16)
        detail::pixelshuffle<DataType::Float16, DataType::UInt16>(src, dst, upscale);
}
