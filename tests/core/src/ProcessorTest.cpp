#include <cstring>

#include <doctest/doctest.h>

#include "AC/Core.hpp"
#include "AC/Specs.hpp"

TEST_CASE("Processor create CPU with string")
{
    auto proc = ac::core::Processor::create("cpu", 0, "acnet-f8b4");
    REQUIRE(static_cast<bool>(proc));
    CHECK(proc->ok());
    CHECK(proc->type() == ac::core::Processor::CPU);
    CHECK(proc->typeName() != nullptr);

    auto name = proc->name();
    CHECK(name != nullptr);
    CHECK(std::strlen(name) > 0);
}

TEST_CASE("Processor create auto with string")
{
    auto proc = ac::core::Processor::create("auto", -1, "acnet-f8b4");
    REQUIRE(static_cast<bool>(proc));
    CHECK(proc->ok());
}

TEST_CASE("Processor process returns correct dimensions")
{
    auto proc = ac::core::Processor::create("cpu", 0, "acnet-f8b4");
    REQUIRE(proc->ok());

    ac::core::Image src{ 32, 32, 3, ac::core::Image::UInt8 };
    std::memset(src.ptr(), 128, src.size());

    auto dst = proc->process(src, 2.0);
    CHECK(dst.width() == src.width() * 2);
    CHECK(dst.height() == src.height() * 2);
    CHECK(dst.channels() == src.channels());
    CHECK(dst.type() == src.type());
    CHECK(!dst.empty());
}

TEST_CASE("Processor process with preallocated dst")
{
    auto proc = ac::core::Processor::create("cpu", 0, "acnet-f8b4");
    REQUIRE(proc->ok());

    ac::core::Image src{ 32, 32, 3, ac::core::Image::UInt8 };
    std::memset(src.ptr(), 128, src.size());

    ac::core::Image dst{ 64, 64, 3, ac::core::Image::UInt8 };
    proc->process(src, dst, 2.0);
    CHECK(!dst.empty());
    CHECK(dst.width() == 64);
    CHECK(dst.height() == 64);
}

TEST_CASE("Processor process ACNetLegacy")
{
    auto proc = ac::core::Processor::create("cpu", 0, "acnet-legacy-gan");
    REQUIRE(proc->ok());

    ac::core::Image src{ 32, 32, 3, ac::core::Image::UInt8 };
    std::memset(src.ptr(), 128, src.size());

    auto dst = proc->process(src, 2.0);
    CHECK(dst.width() == src.width() * 2);
    CHECK(dst.height() == src.height() * 2);
}

TEST_CASE("Processor process ARNet")
{
    auto proc = ac::core::Processor::create("cpu", 0, "arnet-f8b8");
    REQUIRE(proc->ok());

    ac::core::Image src{ 32, 32, 3, ac::core::Image::UInt8 };
    std::memset(src.ptr(), 128, src.size());

    auto dst = proc->process(src, 2.0);
    CHECK(dst.width() == src.width() * 2);
    CHECK(dst.height() == src.height() * 2);
}

TEST_CASE("Processor process ArtCNN")
{
    auto proc = ac::core::Processor::create("cpu", 0, "artcnn-c4f16");
    REQUIRE(proc->ok());

    ac::core::Image src{ 32, 32, 3, ac::core::Image::UInt8 };
    std::memset(src.ptr(), 128, src.size());

    auto dst = proc->process(src, 2.0);
    CHECK(dst.width() == src.width() * 2);
    CHECK(dst.height() == src.height() * 2);
}

TEST_CASE("Processor process FSRCNNX")
{
    auto proc = ac::core::Processor::create("cpu", 0, "fsrcnnx-f8b4");
    REQUIRE(proc->ok());

    ac::core::Image src{ 32, 32, 3, ac::core::Image::UInt8 };
    std::memset(src.ptr(), 128, src.size());

    auto dst = proc->process(src, 2.0);
    CHECK(dst.width() == src.width() * 2);
    CHECK(dst.height() == src.height() * 2);
}

#ifdef AC_CORE_WITH_OPENCL
TEST_CASE("Processor OpenCL")
{
    auto proc = ac::core::Processor::create("opencl", 0, "acnet-f8b4");
    if (!proc->ok())
    {
        MESSAGE("OpenCL device not available");
        return;
    }

    ac::core::Image src{ 32, 32, 3, ac::core::Image::UInt8 };
    std::memset(src.ptr(), 128, src.size());

    auto dst = proc->process(src, 2.0);
    CHECK(dst.width() == src.width() * 2);
    CHECK(dst.height() == src.height() * 2);
}
#endif

#ifdef AC_CORE_WITH_CUDA
TEST_CASE("Processor CUDA")
{
    auto proc = ac::core::Processor::create("cuda", 0, "acnet-f8b4");
    if (!proc->ok())
    {
        MESSAGE("CUDA device not available");
        return;
    }

    ac::core::Image src{ 32, 32, 3, ac::core::Image::UInt8 };
    std::memset(src.ptr(), 128, src.size());

    auto dst = proc->process(src, 2.0);
    CHECK(dst.width() == src.width() * 2);
    CHECK(dst.height() == src.height() * 2);
}
#endif

TEST_CASE("Processor process with factor 4")
{
    auto proc = ac::core::Processor::create("cpu", 0, "acnet-f8b4");
    REQUIRE(proc->ok());

    ac::core::Image src{ 16, 16, 3, ac::core::Image::UInt8 };
    auto dst = proc->process(src, 4.0);

    CHECK(dst.width() == src.width() * 4);
    CHECK(dst.height() == src.height() * 4);
}

TEST_CASE("Processor listInfo")
{
    auto info = ac::core::Processor::listInfo();
    CHECK(info != nullptr);
    CHECK(std::strlen(info) > 0);
}
