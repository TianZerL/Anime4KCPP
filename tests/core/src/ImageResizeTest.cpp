#include <doctest/doctest.h>

#include "AC/Core.hpp"

TEST_CASE("resize factor 1 returns same image")
{
    ac::core::Image src{ 4, 4, 3, ac::core::Image::UInt8 };
    auto dst = ac::core::resize(src, 1.0, 1.0);
    CHECK(dst == src);
}

TEST_CASE("resize factor 2 dimensions")
{
    ac::core::Image src{ 4, 4, 3, ac::core::Image::UInt8 };
    auto dst = ac::core::resize(src, 2.0, 2.0);

    CHECK(dst.width() == 8);
    CHECK(dst.height() == 8);
    CHECK(dst.channels() == 3);
    CHECK(dst.type() == ac::core::Image::UInt8);
}

TEST_CASE("resize factor 0.5 dimensions")
{
    ac::core::Image src{ 4, 4, 3, ac::core::Image::UInt8 };
    auto dst = ac::core::resize(src, 0.5, 0.5);

    CHECK(dst.width() == 2);
    CHECK(dst.height() == 2);
    CHECK(dst.channels() == 3);
    CHECK(dst.type() == ac::core::Image::UInt8);
}

TEST_CASE("resize different horizontal and vertical factors")
{
    ac::core::Image src{ 4, 4, 3, ac::core::Image::UInt8 };
    auto dst = ac::core::resize(src, 2.0, 1.0);

    CHECK(dst.width() == 8);
    CHECK(dst.height() == 4);
}

TEST_CASE("resize point sampling")
{
    ac::core::Image src{ 2, 2, 1, ac::core::Image::UInt8 };
    *src.pixel(0, 0) = 10;
    *src.pixel(1, 0) = 20;
    *src.pixel(0, 1) = 30;
    *src.pixel(1, 1) = 40;

    auto dst = ac::core::resize(src, 2.0, 2.0, ac::core::RESIZE_POINT);

    CHECK(dst.width() == 4);
    CHECK(dst.height() == 4);

    CHECK(*dst.pixel(0, 0) == 10);
    CHECK(*dst.pixel(1, 0) == 10);
    CHECK(*dst.pixel(2, 0) == 20);
    CHECK(*dst.pixel(3, 0) == 20);
    CHECK(*dst.pixel(0, 1) == 10);
    CHECK(*dst.pixel(1, 1) == 10);
    CHECK(*dst.pixel(2, 1) == 20);
    CHECK(*dst.pixel(3, 1) == 20);
    CHECK(*dst.pixel(0, 2) == 30);
    CHECK(*dst.pixel(1, 2) == 30);
    CHECK(*dst.pixel(2, 2) == 40);
    CHECK(*dst.pixel(3, 2) == 40);
    CHECK(*dst.pixel(0, 3) == 30);
    CHECK(*dst.pixel(1, 3) == 30);
    CHECK(*dst.pixel(2, 3) == 40);
    CHECK(*dst.pixel(3, 3) == 40);
}

TEST_CASE("resize dst parameter")
{
    ac::core::Image src{ 4, 4, 3, ac::core::Image::UInt8 };
    ac::core::Image dst{ 8, 8, 3, ac::core::Image::UInt8 };
    ac::core::resize(src, dst, 0.0, 0.0);

    CHECK(dst.width() == 8);
    CHECK(dst.height() == 8);
}

TEST_CASE("resize various modes don't crash")
{
    ac::core::Image src{ 4, 4, 3, ac::core::Image::UInt8 };

    int modes[] = {
        ac::core::RESIZE_CATMULL_ROM, ac::core::RESIZE_MITCHELL_NETRAVALI,
        ac::core::RESIZE_BICUBIC_0_60, ac::core::RESIZE_BICUBIC_0_75, ac::core::RESIZE_BICUBIC_0_100,
        ac::core::RESIZE_BICUBIC_20_50,
        ac::core::RESIZE_SOFTCUBIC50, ac::core::RESIZE_SOFTCUBIC75, ac::core::RESIZE_SOFTCUBIC100,
        ac::core::RESIZE_LANCZOS2, ac::core::RESIZE_LANCZOS3, ac::core::RESIZE_LANCZOS4,
        ac::core::RESIZE_SPLINE16, ac::core::RESIZE_SPLINE36, ac::core::RESIZE_SPLINE64,
        ac::core::RESIZE_BILINEAR,
    };

    for (auto mode : modes)
    {
        auto dst = ac::core::resize(src, 2.0, 2.0, mode);
        CHECK(dst.width() == 8);
        CHECK(dst.height() == 8);
    }
}

TEST_CASE("resize invalid factor returns src")
{
    ac::core::Image src{ 4, 4, 3, ac::core::Image::UInt8 };
    auto dst = ac::core::resize(src, 0.0, 2.0);
    CHECK(dst == src);

    auto dst2 = ac::core::resize(src, -1.0, -1.0);
    CHECK(dst2 == src);
}
