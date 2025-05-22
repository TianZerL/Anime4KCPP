#include <cctype>
#include <string>

#include "AC/Core/Model.hpp"
#include "AC/Core/Processor.hpp"
#include "AC/Core/Util.hpp"

ac::core::Processor::Processor() noexcept : idx(0) {}
ac::core::Processor::~Processor() = default;

ac::core::Image ac::core::Processor::process(const Image& src, const double factor)
{
    Image dst{};
    process(src, dst, factor);
    return dst;
}
void ac::core::Processor::process(const Image& src, Image& dst, const double factor)
{
    Image in{}, out{ src };
    Image uv{};

    int power = factor > 2.0 ? ceilLog2(factor) : 1;
    double fxy = factor / static_cast<double>(1 << power);

    if (src.channels() > 1)
    {
        Image y{};
        if (src.channels() == 4) rgba2yuva(src, y, uv);
        else rgb2yuv(src, y, uv);
        out = y;
    }

    if (!dst.empty())
    {
        if (src.channels() == 1) //grey
        {
            if (fxy == 1.0)
            {
                for (int i = 0; i < power - 1; i++)
                {
                    in = out;
                    out.create(in.width() * 2, in.height() * 2, 1, in.type());
                    process(in, out);
                }
                process(out, dst);
            }
            else
            {
                for (int i = 0; i < power; i++)
                {
                    in = out;
                    out.create(in.width() * 2, in.height() * 2, 1, in.type());
                    process(in, out);
                }
                resize(out, dst, 0.0, 0.0);
            }
        }
        else //rgb[a]
        {
            for (int i = 0; i < power; i++)
            {
                in = out;
                out.create(in.width() * 2, in.height() * 2, 1, in.type());
                process(in, out);
            }

            if (fxy != 1.0) resize(out, out, fxy, fxy);

            resize(uv, uv, factor, factor);
            if (src.channels() == 4) yuva2rgba(out, uv, dst);
            else yuv2rgb(out, uv, dst);
        }
    }
    else
    {
        for (int i = 0; i < power; i++)
        {
            in = out;
            out.create(in.width() * 2, in.height() * 2, 1, in.type());
            process(in, out);
        }

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
}
bool ac::core::Processor::ok() noexcept
{
    return true;
}
const char* ac::core::Processor::error() noexcept
{
    return "NO ERROR";
}

int ac::core::Processor::type(const char* const str) noexcept
{
    if (str)
    {
        std::string typeString = str;

        for (char& ch : typeString) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

        if (typeString == "opencl") return OpenCL;
        if (typeString == "cuda") return CUDA;
    }
    return CPU;
}
const char* ac::core::Processor::type(const int id) noexcept
{
    switch (id)
    {
    case OpenCL:
        return "opencl";
    case CUDA:
        return "cuda";
    default:
        return "cpu";
    }
}
std::shared_ptr<ac::core::Processor> ac::core::Processor::create(const int type, const int device, const char* const model)
{
    auto createImpl = [](int type, int device, auto&& model) {
        switch (type)
        {
#   ifdef AC_CORE_WITH_OPENCL
        case ac::core::Processor::OpenCL:
            return ac::core::Processor::create<ac::core::Processor::OpenCL>(device, model);
#   endif
#   ifdef AC_CORE_WITH_CUDA
        case ac::core::Processor::CUDA:
            return ac::core::Processor::create<ac::core::Processor::CUDA>(device, model);
#   endif
        default:
            return ac::core::Processor::create<ac::core::Processor::CPU>(device, model);
        }
    };

    if (model)
    {
        std::string modelString = model;

        for (char& ch : modelString) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

        if (modelString.find("arnet") != std::string::npos) // ARNet
        {
            auto variant = ac::core::model::ARNet::Variant::HDN;
            if (modelString.find("hdn") != std::string::npos) variant = ac::core::model::ARNet::Variant::HDN;
            return createImpl(type, device, ac::core::model::ARNet{ variant });
        }
        else // ACNet
        {
            auto variant = ac::core::model::ACNet::Variant::GAN;
            if (modelString.find("hdn") != std::string::npos) // ACNet-HDN
            {
                variant = ac::core::model::ACNet::Variant::HDN0;
                for (char ch : modelString)
                {
                    if (ch == '0') variant = ac::core::model::ACNet::Variant::HDN0;
                    else if (ch == '1') variant = ac::core::model::ACNet::Variant::HDN1;
                    else if (ch == '2') variant = ac::core::model::ACNet::Variant::HDN2;
                    else if (ch == '3') variant = ac::core::model::ACNet::Variant::HDN3;
                    else continue;

                    break;
                }
            }
            return createImpl(type, device, ac::core::model::ACNet{ variant });
        }
    }
    return createImpl(type, device, ac::core::model::ACNet{ ac::core::model::ACNet::Variant::HDN0 });
}
