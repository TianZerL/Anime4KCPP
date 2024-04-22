#include "AC/Core/Processor.hpp"
#include "AC/Core/Util.hpp"

ac::core::Processor::Processor(const int idx) noexcept : idx(idx) {}
ac::core::Processor::~Processor() = default;

ac::core::Image ac::core::Processor::process(const Image& src, const double factor)
{
    Image dst{};
    process(src, dst, factor);
    return dst;
}
void ac::core::Processor::process(const Image& src, Image& dst, const double factor)
{
    Image in{}, out{src};
    Image uv{};

    if (src.channels() > 1)
    {   
        Image y{};
        if (src.channels() == 4) rgba2yuva(src, y, uv);
        else rgb2yuv(src, y, uv);
        out = y;
    }

    int power = factor > 2.0 ? ceilLog2(factor) : 1;
    for (int i = 0; i < power; i++)
    {
        in = out;
        out.create(in.width() * 2, in.height() * 2, 1, in.type());
        process(in, out);
    }

    double fxy = factor / (1 << power);
    resize(out, dst, fxy, fxy);

    if (src.channels() > 1)
    {
        Image rgb{};
        resize(uv, uv, factor, factor);
        if (src.channels() == 4) yuva2rgba(dst, uv, rgb);
        else yuv2rgb(dst, uv, rgb);      
        dst = rgb;
    }
}
bool ac::core::Processor::ok() const noexcept
{
    return true;
}
const char* ac::core::Processor::error() const noexcept
{
    return "NO_ERROR";
}
