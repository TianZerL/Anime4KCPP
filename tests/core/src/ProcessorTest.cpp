#include <cstring>
#include <cmath>
#include <random>
#include <limits>

#include <doctest/doctest.h>

#include "AC/Core.hpp"
#include "AC/Specs.hpp"

static double psnr(const ac::core::Image& image, const ac::core::Image& ref)
{
    if (image.size() == 0 ||
        image.type() != ac::core::Image::UInt8 ||
        image.width() != ref.width() ||
        image.height() != ref.height() ||
        image.channels() != ref.channels() ||
        image.type() != ref.type())
        return 0.0;

    double mse = 0.0;

    for (int y = 0; y < image.height(); y++)
    {
        for (int x = 0; x < image.width(); x++)
        {
            auto pixel = image.pixel(x, y);
            auto pixelRef = ref.pixel(x, y);
            for (int c = 0; c < image.channels(); c++)
            {
                double diff = static_cast<double>(pixel[c]) - static_cast<double>(pixelRef[c]);
                mse += diff * diff;
            }
        }
    }

    mse /= static_cast<double>(image.width()) * image.height() * image.channels();

    if (mse == 0.0) return std::numeric_limits<double>::max();

    constexpr double maxVal = 255.0;
    return 10.0 * std::log10((maxVal * maxVal) / mse);
}

TEST_CASE("Processor create")
{
    auto proc = ac::core::Processor::create("cpu", 1, "acnet-f8b4");
    REQUIRE(static_cast<bool>(proc));
    CHECK(proc->ok());
    CHECK(proc->type() == ac::core::Processor::CPU);
    CHECK(proc->typeName() != nullptr);

    auto name = proc->name();
    CHECK(name != nullptr);
    CHECK(std::strlen(name) > 0);
}

TEST_CASE("Processor process returns correct dimensions")
{
    auto proc = ac::core::Processor::create("cpu", 1, "acnet-f8b4");
    REQUIRE(static_cast<bool>(proc));
    REQUIRE(proc->ok());

    ac::core::Image src{ 32, 32, 3, ac::core::Image::UInt8 };
    std::memset(src.ptr(), 128, src.size());

    auto dst = proc->process(src, 2.0);
    REQUIRE(proc->ok());
    CHECK(dst.width() == src.width() * 2);
    CHECK(dst.height() == src.height() * 2);
    CHECK(dst.channels() == src.channels());
    CHECK(dst.type() == src.type());
    CHECK(!dst.empty());
}

TEST_CASE("Processor process with preallocated dst")
{
    auto proc = ac::core::Processor::create("cpu", 1, "acnet-f8b4");
    REQUIRE(static_cast<bool>(proc));
    REQUIRE(proc->ok());

    ac::core::Image src{ 32, 32, 3, ac::core::Image::UInt8 };
    std::memset(src.ptr(), 128, src.size());

    ac::core::Image dst{ 64, 64, 3, ac::core::Image::UInt8 };
    proc->process(src, dst, 2.0);
    REQUIRE(proc->ok());
    CHECK(!dst.empty());
    CHECK(dst.width() == 64);
    CHECK(dst.height() == 64);
}

TEST_CASE("Processor process with factor 4")
{
    auto proc = ac::core::Processor::create("cpu", 1, "acnet-f8b4");
    REQUIRE(static_cast<bool>(proc));
    REQUIRE(proc->ok());

    ac::core::Image src{ 16, 16, 3, ac::core::Image::UInt8 };
    auto dst = proc->process(src, 4.0);
    REQUIRE(proc->ok());
    CHECK(dst.width() == src.width() * 4);
    CHECK(dst.height() == src.height() * 4);
}

TEST_CASE("Processor listInfo")
{
    auto info = ac::core::Processor::listInfo();
    CHECK(info != nullptr);
    CHECK(std::strlen(info) > 0);
}

TEST_CASE("CPU processor process")
{
    std::random_device rd{};
    std::mt19937 gen{ rd() };

    ac::core::Image src{ 64, 64, 3, ac::core::Image::UInt8 };

    auto length = src.size() & -4;
    for (int i = 0; i < length; i += 4)
    {
        auto pixels = gen();
        std::memcpy(src.data() + i, &pixels, 4);
    }

    SUBCASE("CPU processor process with acnet-legacy-hdn0")
    {
        auto procRef = ac::core::Processor::create("cpu", 1, "acnet-legacy-hdn0");
        REQUIRE(static_cast<bool>(procRef));
        REQUIRE(procRef->ok());

        auto ref = procRef->process(src, 2.0);
        REQUIRE(procRef->ok());

        auto proc = ac::core::Processor::create("cpu", 0, "acnet-legacy-hdn0");
        REQUIRE(static_cast<bool>(proc));
        REQUIRE(proc->ok());

        auto dst = proc->process(src, 2.0);
        REQUIRE(proc->ok());

        CHECK(psnr(dst, ref) > 48.0);
    }
    SUBCASE("CPU processor process with acnet-f8b4")
    {
        auto procRef = ac::core::Processor::create("cpu", 1, "acnet-f8b4");
        REQUIRE(static_cast<bool>(procRef));
        REQUIRE(procRef->ok());

        auto ref = procRef->process(src, 2.0);
        REQUIRE(procRef->ok());

        auto proc = ac::core::Processor::create("cpu", 0, "acnet-f8b4");
        REQUIRE(static_cast<bool>(proc));
        REQUIRE(proc->ok());

        auto dst = proc->process(src, 2.0);
        REQUIRE(proc->ok());

        CHECK(psnr(dst, ref) > 48.0);
    }
    SUBCASE("CPU processor process with arnet-f8b8")
    {
        auto procRef = ac::core::Processor::create("cpu", 1, "arnet-f8b8");
        REQUIRE(static_cast<bool>(procRef));
        REQUIRE(procRef->ok());

        auto ref = procRef->process(src, 2.0);
        REQUIRE(procRef->ok());

        auto proc = ac::core::Processor::create("cpu", 0, "arnet-f8b8");
        REQUIRE(static_cast<bool>(proc));
        REQUIRE(proc->ok());

        auto dst = proc->process(src, 2.0);
        REQUIRE(proc->ok());

        CHECK(psnr(dst, ref) > 48.0);
    }
    SUBCASE("CPU processor process with artcnn-c4f16")
    {
        auto procRef = ac::core::Processor::create("cpu", 1, "artcnn-c4f16");
        REQUIRE(static_cast<bool>(procRef));
        REQUIRE(procRef->ok());

        auto ref = procRef->process(src, 2.0);
        REQUIRE(procRef->ok());

        auto proc = ac::core::Processor::create("cpu", 0, "artcnn-c4f16");
        REQUIRE(static_cast<bool>(proc));
        REQUIRE(proc->ok());

        auto dst = proc->process(src, 2.0);
        REQUIRE(proc->ok());

        CHECK(psnr(dst, ref) > 48.0);
    }
    SUBCASE("CPU processor process with fsrcnnx-f8b4")
    {
        auto procRef = ac::core::Processor::create("cpu", 1, "fsrcnnx-f8b4");
        REQUIRE(static_cast<bool>(procRef));
        REQUIRE(procRef->ok());

        auto ref = procRef->process(src, 2.0);
        REQUIRE(procRef->ok());

        auto proc = ac::core::Processor::create("cpu", 0, "fsrcnnx-f8b4");
        REQUIRE(static_cast<bool>(proc));
        REQUIRE(proc->ok());

        auto dst = proc->process(src, 2.0);
        REQUIRE(proc->ok());

        CHECK(psnr(dst, ref) > 48.0);
    }
}

#ifdef AC_CORE_WITH_OPENCL
TEST_CASE("OpenCL processor process")
{
    std::random_device rd{};
    std::mt19937 gen{ rd() };

    ac::core::Image src{ 64, 64, 3, ac::core::Image::UInt8 };

    auto length = src.size() & -4;
    for (int i = 0; i < length; i += 4)
    {
        auto pixels = gen();
        std::memcpy(src.data() + i, &pixels, 4);
    }

    SUBCASE("OpenCL processor process with acnet-legacy-hdn0")
    {
        auto procRef = ac::core::Processor::create("cpu", 1, "acnet-legacy-hdn0");
        REQUIRE(static_cast<bool>(procRef));
        REQUIRE(procRef->ok());

        auto ref = procRef->process(src, 2.0);
        REQUIRE(procRef->ok());

        auto proc = ac::core::Processor::create("opencl", 0, "acnet-legacy-hdn0");
        REQUIRE(static_cast<bool>(proc));
        if (!proc->ok())
        {
            MESSAGE("OpenCL device not available");
            return;
        }

        auto dst = proc->process(src, 2.0);
        REQUIRE(proc->ok());

        CHECK(psnr(dst, ref) > 48.0);
    }
    SUBCASE("OpenCL processor process with acnet-f8b4")
    {
        auto procRef = ac::core::Processor::create("cpu", 1, "acnet-f8b4");
        REQUIRE(static_cast<bool>(procRef));
        REQUIRE(procRef->ok());

        auto ref = procRef->process(src, 2.0);
        REQUIRE(procRef->ok());

        auto proc = ac::core::Processor::create("opencl", 0, "acnet-f8b4");
        REQUIRE(static_cast<bool>(proc));
        if (!proc->ok())
        {
            MESSAGE("OpenCL device not available");
            return;
        }

        auto dst = proc->process(src, 2.0);
        REQUIRE(proc->ok());

        CHECK(psnr(dst, ref) > 48.0);
    }
    SUBCASE("OpenCL processor process with arnet-f8b8")
    {
        auto procRef = ac::core::Processor::create("cpu", 1, "arnet-f8b8");
        REQUIRE(static_cast<bool>(procRef));
        REQUIRE(procRef->ok());

        auto ref = procRef->process(src, 2.0);
        REQUIRE(procRef->ok());

        auto proc = ac::core::Processor::create("opencl", 0, "arnet-f8b8");
        REQUIRE(static_cast<bool>(proc));
        if (!proc->ok())
        {
            MESSAGE("OpenCL device not available");
            return;
        }

        auto dst = proc->process(src, 2.0);
        CHECK(proc->ok());

        CHECK(psnr(dst, ref) > 48.0);
    }
    SUBCASE("OpenCL processor process with artcnn-c4f16")
    {
        auto procRef = ac::core::Processor::create("cpu", 1, "artcnn-c4f16");
        REQUIRE(static_cast<bool>(procRef));
        REQUIRE(procRef->ok());

        auto ref = procRef->process(src, 2.0);
        REQUIRE(procRef->ok());

        auto proc = ac::core::Processor::create("opencl", 0, "artcnn-c4f16");
        REQUIRE(static_cast<bool>(proc));
        if (!proc->ok())
        {
            MESSAGE("OpenCL device not available");
            return;
        }

        auto dst = proc->process(src, 2.0);
        REQUIRE(proc->ok());

        CHECK(psnr(dst, ref) > 48.0);
    }
    SUBCASE("OpenCL processor process with fsrcnnx-f8b4")
    {
        auto procRef = ac::core::Processor::create("cpu", 1, "fsrcnnx-f8b4");
        REQUIRE(static_cast<bool>(procRef));
        REQUIRE(procRef->ok());

        auto ref = procRef->process(src, 2.0);
        REQUIRE(procRef->ok());

        auto proc = ac::core::Processor::create("opencl", 0, "fsrcnnx-f8b4");
        REQUIRE(static_cast<bool>(proc));
        if (!proc->ok())
        {
            MESSAGE("OpenCL device not available");
            return;
        }

        auto dst = proc->process(src, 2.0);
        REQUIRE(proc->ok());

        CHECK(psnr(dst, ref) > 48.0);
    }
}
#endif

#ifdef AC_CORE_WITH_CUDA
TEST_CASE("CUDA processor process")
{
    std::random_device rd{};
    std::mt19937 gen{ rd() };

    ac::core::Image src{ 64, 64, 3, ac::core::Image::UInt8 };

    auto length = src.size() & -4;
    for (int i = 0; i < length; i += 4)
    {
        auto pixels = gen();
        std::memcpy(src.data() + i, &pixels, 4);
    }

    SUBCASE("CUDA processor process with acnet-legacy-hdn0")
    {
        auto procRef = ac::core::Processor::create("cpu", 1, "acnet-legacy-hdn0");
        REQUIRE(static_cast<bool>(procRef));
        REQUIRE(procRef->ok());

        auto ref = procRef->process(src, 2.0);
        REQUIRE(procRef->ok());

        auto proc = ac::core::Processor::create("cuda", 0, "acnet-legacy-hdn0");
        REQUIRE(static_cast<bool>(proc));
        if (!proc->ok())
        {
            MESSAGE("CUDA device not available");
            return;
        }

        auto dst = proc->process(src, 2.0);
        REQUIRE(proc->ok());

        CHECK(psnr(dst, ref) > 48.0);
    }
    SUBCASE("CUDA processor process with acnet-f8b4")
    {
        auto procRef = ac::core::Processor::create("cpu", 1, "acnet-f8b4");
        REQUIRE(static_cast<bool>(procRef));
        REQUIRE(procRef->ok());

        auto ref = procRef->process(src, 2.0);
        REQUIRE(procRef->ok());

        auto proc = ac::core::Processor::create("cuda", 0, "acnet-f8b4");
        REQUIRE(static_cast<bool>(proc));
        if (!proc->ok())
        {
            MESSAGE("CUDA device not available");
            return;
        }

        auto dst = proc->process(src, 2.0);
        REQUIRE(proc->ok());

        CHECK(psnr(dst, ref) > 48.0);
    }
    SUBCASE("CUDA processor process with arnet-f8b8")
    {
        auto procRef = ac::core::Processor::create("cpu", 1, "arnet-f8b8");
        REQUIRE(static_cast<bool>(procRef));
        REQUIRE(procRef->ok());

        auto ref = procRef->process(src, 2.0);
        REQUIRE(procRef->ok());

        auto proc = ac::core::Processor::create("cuda", 0, "arnet-f8b8");
        REQUIRE(static_cast<bool>(proc));
        if (!proc->ok())
        {
            MESSAGE("CUDA device not available");
            return;
        }

        auto dst = proc->process(src, 2.0);
        CHECK(proc->ok());

        CHECK(psnr(dst, ref) > 48.0);
    }
    SUBCASE("CUDA processor process with artcnn-c4f16")
    {
        auto procRef = ac::core::Processor::create("cpu", 1, "artcnn-c4f16");
        REQUIRE(static_cast<bool>(procRef));
        REQUIRE(procRef->ok());

        auto ref = procRef->process(src, 2.0);
        REQUIRE(procRef->ok());

        auto proc = ac::core::Processor::create("cuda", 0, "artcnn-c4f16");
        REQUIRE(static_cast<bool>(proc));
        if (!proc->ok())
        {
            MESSAGE("CUDA device not available");
            return;
        }

        auto dst = proc->process(src, 2.0);
        REQUIRE(proc->ok());

        CHECK(psnr(dst, ref) > 48.0);
    }
    SUBCASE("CUDA processor process with fsrcnnx-f8b4")
    {
        auto procRef = ac::core::Processor::create("cpu", 1, "fsrcnnx-f8b4");
        REQUIRE(static_cast<bool>(procRef));
        REQUIRE(procRef->ok());

        auto ref = procRef->process(src, 2.0);
        REQUIRE(procRef->ok());

        auto proc = ac::core::Processor::create("cuda", 0, "fsrcnnx-f8b4");
        REQUIRE(static_cast<bool>(proc));
        if (!proc->ok())
        {
            MESSAGE("CUDA device not available");
            return;
        }

        auto dst = proc->process(src, 2.0);
        REQUIRE(proc->ok());

        CHECK(psnr(dst, ref) > 48.0);
    }
}
#endif
