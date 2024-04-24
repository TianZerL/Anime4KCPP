#include <cassert>

#define STB_IMAGE_RESIZE2_IMPLEMENTATION
#include <stb_image_resize2.h>

#include "AC/Core/Image.hpp"
#include "AC/Core/Util.hpp"

namespace ac::core::detail
{
    inline static void resize(const Image& src, Image& dst, const double fx, const double fy) noexcept
    {
        if (src.empty()) return;

        if (fx > 0.0 && fy > 0.0)
        {
            if (fx == 1.0 && fy == 1.0)
            {
                dst = src;
                return ;
            }

            auto dstW = static_cast<int>(src.width() * fx);
            auto dstH = static_cast<int>(src.height() * fy);

            if ((dst.width() != dstW) || (dst.height() != dstH) || (dst.channels() != src.channels()))
                dst.create(dstW, dstH, src.channels(), src.type());
        }
        else
        {
            if (dst.empty()) return;
            if (dst.width() == src.width() && dst.height() == src.height())
            {
                dst = src;
                return ;
            }

            if (dst.channels() != src.channels())
                dst.create(dst.width(), dst.height(), src.channels(), src.type());
        }

        stbir_resize(
            src.ptr(), src.width(), src.height(), src.stride(),
            dst.ptr(), dst.width(), dst.height(), dst.stride(),
            [&]()->stbir_pixel_layout {
                switch (src.channels())
                {
                case 1 : return STBIR_1CHANNEL;
                case 2 : return STBIR_2CHANNEL;
                case 3 : return STBIR_RGB;
                case 4 : return STBIR_4CHANNEL;
                default: return assert(src.channels() == 1 || src.channels() == 2 || src.channels() == 3 || src.channels() == 4), STBIR_1CHANNEL;
                }
            }(), 
            [&]()->stbir_datatype {
                switch (src.type())
                {
                case Image::UInt8 : return STBIR_TYPE_UINT8;
                case Image::UInt16 : return STBIR_TYPE_UINT16;
                case Image::Float32 : return STBIR_TYPE_FLOAT;
                default: return assert(src.type() == Image::UInt8 || src.type() == Image::UInt16 || src.type() == Image::Float32), STBIR_TYPE_UINT8;
                }
            }(), 
            STBIR_EDGE_CLAMP, STBIR_FILTER_TRIANGLE
        );
    }

    template<typename IN, typename OUT = IN>
    inline static void rgb2yuv(const Image& src, Image& dst)
    {
        filter([](const int /*i*/, const int /*j*/, void* const sptr, void* const dptr) {
            auto in = static_cast<IN*>(sptr);
            auto out = static_cast<IN*>(dptr);

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
    inline static void rgb2yuv(const Image& src, Image& dsty, Image& dstuv)
    {
        filter([](const int /*i*/, const int /*j*/, void* const sptr, void* const yptr, void* const uvptr) {
            auto in = static_cast<IN*>(sptr);
            auto yout = static_cast<IN*>(yptr);
            auto uvout = static_cast<IN*>(uvptr);

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
    inline static void rgb2yuv(const Image& src, Image& dsty, Image& dstu, Image& dstv)
    {
        filter([](const int /*i*/, const int /*j*/, void* const sptr, void* const yptr, void* const uptr, void* const vptr) {
            auto in = static_cast<IN*>(sptr);
            auto yout = static_cast<IN*>(yptr);
            auto uout = static_cast<IN*>(uptr);
            auto vout = static_cast<IN*>(vptr);

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
    inline static void rgba2yuva(const Image& src, Image& dst)
    {
        filter([](const int /*i*/, const int /*j*/, void* const sptr, void* const dptr) {
            auto in = static_cast<IN*>(sptr);
            auto out = static_cast<IN*>(dptr);

            float r = toFloat(in[0]);
            float g = toFloat(in[1]);
            float b = toFloat(in[2]);

            float y = 0.299f * r + 0.587f * g + 0.114f * b;
            float u = 0.564f * (b - y) + 0.5f;
            float v = 0.713f * (r - y) + 0.5f;
            
            out[0] = fromFloat<OUT>(y);
            out[1] = fromFloat<OUT>(u);
            out[2] = fromFloat<OUT>(v);
            out[3] = in[3];
        }, src, dst);
    }
    template<typename IN, typename OUT = IN>
    inline static void rgba2yuva(const Image& src, Image& dsty, Image& dstuva)
    {
        filter([](const int /*i*/, const int /*j*/, void* const sptr, void* const yptr, void* const uvaptr) {
            auto in = static_cast<IN*>(sptr);
            auto yout = static_cast<IN*>(yptr);
            auto uvaout = static_cast<IN*>(uvaptr);

            float r = toFloat(in[0]);
            float g = toFloat(in[1]);
            float b = toFloat(in[2]);

            float y = 0.299f * r + 0.587f * g + 0.114f * b;
            float u = 0.564f * (b - y) + 0.5f;
            float v = 0.713f * (r - y) + 0.5f;

            yout[0] = fromFloat<OUT>(y);
            uvaout[0] = fromFloat<OUT>(u);
            uvaout[1] = fromFloat<OUT>(v);
            uvaout[2] = in[3];
        }, src, dsty, dstuva);
    }
    template<typename IN, typename OUT = IN>
    inline static void rgba2yuva(const Image& src, Image& dsty, Image& dstu, Image& dstv, Image& dsta)
    {
        filter([](const int /*i*/, const int /*j*/, void* const sptr, void* const yptr, void* const uptr, void* const vptr, void* const aptr) {
            auto in = static_cast<IN*>(sptr);
            auto yout = static_cast<IN*>(yptr);
            auto uout = static_cast<IN*>(uptr);
            auto vout = static_cast<IN*>(vptr);
            auto aout = static_cast<IN*>(aptr);

            float r = toFloat(in[0]);
            float g = toFloat(in[1]);
            float b = toFloat(in[2]);

            float y = 0.299f * r + 0.587f * g + 0.114f * b;
            float u = 0.564f * (b - y) + 0.5f;
            float v = 0.713f * (r - y) + 0.5f;

            *yout = fromFloat<OUT>(y);
            *uout = fromFloat<OUT>(u);
            *vout = fromFloat<OUT>(v);
            *aout = in[3];
        }, src, dsty, dstu, dstv, dsta);
    }

    template<typename IN, typename OUT = IN>
    inline static void yuv2rgb(const Image& src, Image& dst)
    {
        filter([](const int /*i*/, const int /*j*/, void* const sptr, void* const dptr) {
            auto in = static_cast<IN*>(sptr);
            auto out = static_cast<IN*>(dptr);

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
    inline static void yuv2rgb(const Image& srcy, const Image& srcuv, Image& dst)
    {
        filter([](const int /*i*/, const int /*j*/, void* const yptr, void* const uvptr, void* const dptr) {
            auto yin = static_cast<IN*>(yptr);
            auto uvin = static_cast<IN*>(uvptr);
            auto out = static_cast<IN*>(dptr);

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
    inline static void yuv2rgb(const Image& srcy, const Image& srcu, const Image& srcv, Image& dst)
    {
        filter([](const int /*i*/, const int /*j*/, void* const yptr, void* const uptr, void* const vptr, void* const dptr) {
            auto yin = static_cast<IN*>(yptr);
            auto uin = static_cast<IN*>(uptr);
            auto vin = static_cast<IN*>(vptr);
            auto out = static_cast<IN*>(dptr);

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
    inline static void yuva2rgba(const Image& src, Image& dst)
    {
        filter([](const int /*i*/, const int /*j*/, void* const sptr, void* const dptr) {
            auto in = static_cast<IN*>(sptr);
            auto out = static_cast<IN*>(dptr);

            float y = toFloat(in[0]);
            float u = toFloat(in[1]) - 0.5f;
            float v = toFloat(in[2]) - 0.5f;

            float r = y + 1.403f * v;
            float g = y - 0.344f * u - 0.714f * v;
            float b = y + 1.773f * u;

            out[0] = fromFloat<OUT>(r);
            out[1] = fromFloat<OUT>(g);
            out[2] = fromFloat<OUT>(b);
            out[3] = in[3];
        }, src, dst);
    }
    template<typename IN, typename OUT = IN>
    inline static void yuva2rgba(const Image& srcy, const Image& srcuva, Image& dst)
    {
        filter([](const int /*i*/, const int /*j*/, void* const yptr, void* const uvaptr, void* const dptr) {
            auto yin = static_cast<IN*>(yptr);
            auto uvain = static_cast<IN*>(uvaptr);
            auto out = static_cast<IN*>(dptr);

            float y = toFloat(yin[0]);
            float u = toFloat(uvain[0]) - 0.5f;
            float v = toFloat(uvain[1]) - 0.5f;

            float r = y + 1.403f * v;
            float g = y - 0.344f * u - 0.714f * v;
            float b = y + 1.773f * u;

            out[0] = fromFloat<OUT>(r);
            out[1] = fromFloat<OUT>(g);
            out[2] = fromFloat<OUT>(b);
            out[3] = uvain[2];
        }, srcy, srcuva, dst);
    }
    template<typename IN, typename OUT = IN>
    inline static void yuva2rgba(const Image& srcy, const Image& srcu, const Image& srcv, const Image& srca, Image& dst)
    {
        filter([](const int /*i*/, const int /*j*/, void* const yptr, void* const uptr, void* const vptr, void* const aptr, void* const dptr) {
            auto yin = static_cast<IN*>(yptr);
            auto uin = static_cast<IN*>(uptr);
            auto vin = static_cast<IN*>(vptr);
            auto ain = static_cast<IN*>(aptr);
            auto out = static_cast<IN*>(dptr);

            float y = toFloat(*yin);
            float u = toFloat(*uin) - 0.5f;
            float v = toFloat(*vin) - 0.5f;

            float r = y + 1.403f * v;
            float g = y - 0.344f * u - 0.714f * v;
            float b = y + 1.773f * u;

            out[0] = fromFloat<OUT>(r);
            out[1] = fromFloat<OUT>(g);
            out[2] = fromFloat<OUT>(b);
            out[3] = *ain;
        }, srcy, srcu, srcv, srca, dst);
    }
}

void ac::core::resize(const ac::core::Image& src, ac::core::Image& dst, const double fx, const double fy) noexcept
{
    if (src == dst)
    {
        Image tmp{};
        detail::resize(src, tmp, fx, fy);
        if (!tmp.empty()) dst = tmp;
    }
    else detail::resize(src, dst, fx, fy);
}
ac::core::Image ac::core::resize(const ac::core::Image& src, const double fx, const double fy) noexcept
{
    if (fx <= 0.0 || fy <= 0.0) return src;
    
    Image dst{};
    detail::resize(src, dst, fx, fy);
    return dst;
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