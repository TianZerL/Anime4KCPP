#include <cstring>

#include <doctest/doctest.h>

#include "AC/Core.hpp"

TEST_CASE("rgb2yuv single plane dimensions")
{
    ac::core::Image rgb{ 4, 4, 3, ac::core::Image::UInt8 };
    std::memset(rgb.ptr(), 128, rgb.size());

    ac::core::Image yuv{};
    ac::core::rgb2yuv(rgb, yuv);

    CHECK(yuv.width() == 4);
    CHECK(yuv.height() == 4);
    CHECK(yuv.channels() == 3);
    CHECK(yuv.type() == ac::core::Image::UInt8);
}

TEST_CASE("rgb2yuv two plane dimensions")
{
    ac::core::Image rgb{ 4, 4, 3, ac::core::Image::UInt8 };
    std::memset(rgb.ptr(), 128, rgb.size());

    ac::core::Image y{}, uv{};
    ac::core::rgb2yuv(rgb, y, uv);

    CHECK(y.width() == 4);
    CHECK(y.height() == 4);
    CHECK(y.channels() == 1);
    CHECK(uv.width() == 4);
    CHECK(uv.height() == 4);
    CHECK(uv.channels() == 2);
}

TEST_CASE("rgb2yuv three plane dimensions")
{
    ac::core::Image rgb{ 4, 4, 3, ac::core::Image::UInt8 };
    std::memset(rgb.ptr(), 128, rgb.size());

    ac::core::Image y{}, u{}, v{};
    ac::core::rgb2yuv(rgb, y, u, v);

    CHECK(y.width() == 4);
    CHECK(y.height() == 4);
    CHECK(y.channels() == 1);
    CHECK(u.channels() == 1);
    CHECK(v.channels() == 1);
}

TEST_CASE("gray rgb2yuv then yuv2rgb round-trip single plane")
{
    ac::core::Image rgb{ 4, 4, 3, ac::core::Image::UInt8 };
    for (int i = 0; i < rgb.height(); i++)
        for (int j = 0; j < rgb.width(); j++)
        {
            auto p = rgb.pixel(j, i);
            p[0] = p[1] = p[2] = 100;
        }

    ac::core::Image yuv{};
    ac::core::rgb2yuv(rgb, yuv);

    ac::core::Image back{};
    ac::core::yuv2rgb(yuv, back);

    for (int i = 0; i < back.height(); i++)
        for (int j = 0; j < back.width(); j++)
        {
            auto p = back.pixel(j, i);
            CHECK(p[0] >= 99);
            CHECK(p[0] <= 101);
            CHECK(p[1] >= 99);
            CHECK(p[1] <= 101);
            CHECK(p[2] >= 99);
            CHECK(p[2] <= 101);
        }
}

TEST_CASE("gray rgb2yuv then yuv2rgb round-trip two plane")
{
    ac::core::Image rgb{ 4, 4, 3, ac::core::Image::UInt8 };
    for (int i = 0; i < rgb.height(); i++)
        for (int j = 0; j < rgb.width(); j++)
        {
            auto p = rgb.pixel(j, i);
            p[0] = p[1] = p[2] = 100;
        }

    ac::core::Image y{}, uv{};
    ac::core::rgb2yuv(rgb, y, uv);

    ac::core::Image back{};
    ac::core::yuv2rgb(y, uv, back);

    for (int i = 0; i < back.height(); i++)
        for (int j = 0; j < back.width(); j++)
        {
            auto p = back.pixel(j, i);
            CHECK(p[0] >= 99);
            CHECK(p[0] <= 101);
            CHECK(p[1] >= 99);
            CHECK(p[1] <= 101);
            CHECK(p[2] >= 99);
            CHECK(p[2] <= 101);
        }
}

TEST_CASE("gray rgb2yuv then yuv2rgb round-trip three plane")
{
    ac::core::Image rgb{ 4, 4, 3, ac::core::Image::UInt8 };
    for (int i = 0; i < rgb.height(); i++)
        for (int j = 0; j < rgb.width(); j++)
        {
            auto p = rgb.pixel(j, i);
            p[0] = p[1] = p[2] = 100;
        }

    ac::core::Image y{}, u{}, v{};
    ac::core::rgb2yuv(rgb, y, u, v);

    ac::core::Image back{};
    ac::core::yuv2rgb(y, u, v, back);

    for (int i = 0; i < back.height(); i++)
        for (int j = 0; j < back.width(); j++)
        {
            auto p = back.pixel(j, i);
            CHECK(p[0] >= 99);
            CHECK(p[0] <= 101);
            CHECK(p[1] >= 99);
            CHECK(p[1] <= 101);
            CHECK(p[2] >= 99);
            CHECK(p[2] <= 101);
        }
}

TEST_CASE("shl and shr")
{
    ac::core::Image img{ 2, 2, 1, ac::core::Image::UInt8 };
    for (int i = 0; i < img.height(); i++)
        for (int j = 0; j < img.width(); j++)
            *img.pixel(j, i) = 0x0F;

    ac::core::shl(img, 4);
    CHECK(*img.pixel(0, 0) == 0xf0);
    CHECK(*img.pixel(1, 0) == 0xf0);

    ac::core::shr(img, 4);
    CHECK(*img.pixel(0, 0) == 0x0f);
    CHECK(*img.pixel(1, 0) == 0x0f);
}

TEST_CASE("shl src dst")
{
    ac::core::Image src{ 2, 2, 1, ac::core::Image::UInt8 };
    for (int i = 0; i < src.height(); i++)
        for (int j = 0; j < src.width(); j++)
            *src.pixel(j, i) = 0x03;

    ac::core::Image dst{};
    ac::core::shl(src, dst, 6);
    CHECK(*dst.pixel(0, 0) == 0xc0);
}

TEST_CASE("shr src dst")
{
    ac::core::Image src{ 2, 2, 1, ac::core::Image::UInt8 };
    for (int i = 0; i < src.height(); i++)
        for (int j = 0; j < src.width(); j++)
            *src.pixel(j, i) = 0xc0;

    ac::core::Image dst{};
    ac::core::shr(src, dst, 6);
    CHECK(*dst.pixel(0, 0) == 0x03);
}

TEST_CASE("astype uint8 to float32")
{
    ac::core::Image src{ 2, 2, 1, ac::core::Image::UInt8 };
    *src.pixel(0, 0) = 0;
    *src.pixel(1, 0) = 255;
    *src.pixel(0, 1) = 128;
    *src.pixel(1, 1) = 64;

    ac::core::Image dst = ac::core::astype(src, ac::core::Image::Float32);
    CHECK(dst.type() == ac::core::Image::Float32);
    CHECK(dst.width() == 2);
    CHECK(dst.height() == 2);
    CHECK(dst.channels() == 1);

    CHECK(*static_cast<const float*>(dst.ptr(0, 0)) == doctest::Approx(0.0f));
    CHECK(*static_cast<const float*>(dst.ptr(1, 0)) == doctest::Approx(1.0f));
    CHECK(*static_cast<const float*>(dst.ptr(0, 1)) == doctest::Approx(128.0f / 255.0f));
    CHECK(*static_cast<const float*>(dst.ptr(1, 1)) == doctest::Approx(64.0f / 255.0f));
}

TEST_CASE("astype same type returns src")
{
    ac::core::Image src{ 2, 2, 1, ac::core::Image::UInt8 };
    ac::core::Image dst = ac::core::astype(src, ac::core::Image::UInt8);
    CHECK(dst == src);
}

TEST_CASE("copy same type")
{
    ac::core::Image src{ 2, 2, 1, ac::core::Image::UInt8 };
    for (int i = 0; i < src.height(); i++)
        for (int j = 0; j < src.width(); j++)
            *src.pixel(j, i) = static_cast<std::uint8_t>(i * 2 + j);

    ac::core::Image dst{ 2, 2, 1, ac::core::Image::UInt8 };
    ac::core::copy(src, dst);

    for (int i = 0; i < dst.height(); i++)
        for (int j = 0; j < dst.width(); j++)
            CHECK(*dst.pixel(j, i) == *src.pixel(j, i));
}

TEST_CASE("copy different type conversion")
{
    ac::core::Image src{ 2, 2, 1, ac::core::Image::UInt8 };
    for (int i = 0; i < src.height(); i++)
        for (int j = 0; j < src.width(); j++)
            *src.pixel(j, i) = 128;

    ac::core::Image dst{ 2, 2, 1, ac::core::Image::Float32 };
    ac::core::copy(src, dst);

    CHECK(dst.type() == ac::core::Image::Float32);
    CHECK(*static_cast<const float*>(dst.ptr(0, 0)) == doctest::Approx(128.0f / 255.0f));
    CHECK(*static_cast<const float*>(dst.ptr(1, 0)) == doctest::Approx(128.0f / 255.0f));
}

TEST_CASE("crop")
{
    ac::core::Image src{ 4, 4, 1, ac::core::Image::UInt8 };
    auto c = ac::core::crop(src, 1, 1, 2, 2);

    CHECK(c.width() == 2);
    CHECK(c.height() == 2);
    CHECK(c.channels() == 1);
    CHECK(c == src);
}

TEST_CASE("crop negative width height")
{
    ac::core::Image src{ 4, 4, 1, ac::core::Image::UInt8 };
    auto c = ac::core::crop(src, 3, 3, -2, -2);

    CHECK(c.width() == 2);
    CHECK(c.height() == 2);
}

TEST_CASE("crop empty source")
{
    ac::core::Image src{};
    auto c = ac::core::crop(src, 0, 0, 2, 2);
    CHECK(c.empty());
}

TEST_CASE("extract channels")
{
    ac::core::Image src{ 4, 4, 4, ac::core::Image::UInt8 };
    auto e = ac::core::extract(src, 0, 2);

    CHECK(e.width() == 4);
    CHECK(e.height() == 4);
    CHECK(e.channels() == 2);
    CHECK(e.type() == ac::core::Image::UInt8);
}

TEST_CASE("extract invalid parameters")
{
    ac::core::Image src{ 4, 4, 3, ac::core::Image::UInt8 };
    auto e1 = ac::core::extract(src, 5, 1);
    CHECK(e1.empty());

    auto e2 = ac::core::extract(src, 0, 0);
    CHECK(e2.empty());
}

TEST_CASE("insert channel")
{
    ac::core::Image src{ 2, 2, 2, ac::core::Image::UInt8 };
    ac::core::Image inserted{ 2, 2, 1, ac::core::Image::UInt8 };
    auto result = ac::core::insert(src, inserted, 1);

    CHECK(result.width() == 2);
    CHECK(result.height() == 2);
    CHECK(result.channels() == 3);
    CHECK(result.type() == ac::core::Image::UInt8);
}

TEST_CASE("pixelShuffle dimensions")
{
    ac::core::Image src{ 4, 4, 16, ac::core::Image::UInt8 };
    ac::core::Image dst{};
    ac::core::pixelShuffle(src, dst, 2);

    CHECK(dst.width() == 8);
    CHECK(dst.height() == 8);
    CHECK(dst.channels() == 4);
}

TEST_CASE("pixelShuffle invalid")
{
    ac::core::Image src{ 4, 4, 3, ac::core::Image::UInt8 };
    ac::core::Image dst{};
    ac::core::pixelShuffle(src, dst, 2);

    CHECK(dst.empty());
}

TEST_CASE("unpadding")
{
    ac::core::Image src{ 4, 4, 1, ac::core::Image::UInt8 };
    int lineSize = src.width() * src.pixelSize();

    CHECK(ac::core::unpadding(src) == src);
    CHECK(src.stride() >= lineSize);
}

TEST_CASE("unpadding with custom stride")
{
    std::uint8_t buffer[4 * 16]{};
    ac::core::Image src{};
    src.map(4, 4, 1, ac::core::Image::UInt8, buffer, 16);

    auto u = ac::core::unpadding(src);
    CHECK(u.width() == 4);
    CHECK(u.height() == 4);
    CHECK(u.channels() == 1);
    CHECK(u.stride() == 4);
}
