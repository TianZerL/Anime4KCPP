#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "AC/Core.hpp"

TEST_CASE("Image default construction")
{
    ac::core::Image img{};

    CHECK(img.empty());
    CHECK(img.width() == 0);
    CHECK(img.height() == 0);
    CHECK(img.channels() == 0);
    CHECK(img.size() == 0);
    CHECK(img.data() == nullptr);
    CHECK(img.ptr() == nullptr);
    CHECK(!img.ownership());
}

TEST_CASE("Image sized construction")
{
    ac::core::Image img{ 4, 3, 3, ac::core::Image::UInt8 };

    CHECK(!img.empty());
    CHECK(img.width() == 4);
    CHECK(img.height() == 3);
    CHECK(img.channels() == 3);
    CHECK(img.type() == ac::core::Image::UInt8);
    CHECK(img.elementSize() == 1);
    CHECK(img.pixelSize() == 3);
    CHECK(img.isUint());
    CHECK(!img.isFloat());
    CHECK(img.ownership());
    CHECK(img.ptr() != nullptr);

    auto lineSize = img.width() * img.pixelSize();
    CHECK(img.stride() >= lineSize);
    CHECK(img.size() == img.height() * img.stride());
}

TEST_CASE("Image sized construction stride")
{
    int lineSize = 4 * 3 * 1;
    ac::core::Image img{ 4, 3, 3, ac::core::Image::UInt8, 16 };

    CHECK(img.stride() == 16);
    CHECK(img.size() == 3 * 16);
}

TEST_CASE("Image external data construction")
{
    std::uint8_t buffer[4 * 3 * 4]{};
    ac::core::Image img{ 4, 3, 4, ac::core::Image::UInt8, buffer };

    CHECK(!img.empty());
    CHECK(img.width() == 4);
    CHECK(img.height() == 3);
    CHECK(img.channels() == 4);
    CHECK(img.ptr() == static_cast<void*>(buffer));
    CHECK(!img.ownership());
    CHECK(img.data() == buffer);
}

TEST_CASE("Image create")
{
    ac::core::Image img{};
    img.create(5, 2, 1, ac::core::Image::UInt8);

    CHECK(!img.empty());
    CHECK(img.width() == 5);
    CHECK(img.height() == 2);
    CHECK(img.channels() == 1);
    CHECK(img.ownership());
    CHECK(img.ptr() != nullptr);
}

TEST_CASE("Image create with invalid dimensions")
{
    ac::core::Image img{};
    img.create(0, 0, 0, ac::core::Image::UInt8);
    CHECK(img.empty());

    img.create(4, 0, 3, ac::core::Image::UInt8);
    CHECK(img.empty());

    img.create(0, 3, 3, ac::core::Image::UInt8);
    CHECK(img.empty());
}

TEST_CASE("Image map")
{
    std::uint8_t buffer[16]{};
    ac::core::Image img{};
    img.map(4, 2, 2, ac::core::Image::UInt8, buffer);

    CHECK(!img.empty());
    CHECK(img.width() == 4);
    CHECK(img.height() == 2);
    CHECK(img.channels() == 2);
    CHECK(img.ptr() == static_cast<void*>(buffer));
    CHECK(!img.ownership());
}

TEST_CASE("Image map with null data")
{
    ac::core::Image img{};
    img.map(4, 2, 1, ac::core::Image::UInt8, nullptr);
    CHECK(img.empty());
}

TEST_CASE("Image from")
{
    std::uint8_t src[4]{ 1, 2, 3, 4 };
    ac::core::Image img{};
    img.from(2, 2, 1, ac::core::Image::UInt8, src);

    CHECK(!img.empty());
    CHECK(img.width() == 2);
    CHECK(img.height() == 2);
    CHECK(img.channels() == 1);
    CHECK(img.ownership());
    CHECK(img.ptr() != static_cast<void*>(src));

    for (int i = 0; i < img.height(); i++)
        for (int j = 0; j < img.width(); j++)
        {
            auto v = *img.pixel(j, i);
            CHECK(v == src[i * img.width() + j]);
        }
}

TEST_CASE("Image to")
{
    ac::core::Image img{ 2, 2, 1, ac::core::Image::UInt8 };
    for (int i = 0; i < img.height(); i++)
        for (int j = 0; j < img.width(); j++)
            *img.pixel(j, i) = static_cast<std::uint8_t>(i * img.width() + j);

    std::uint8_t dst[4]{};
    img.to(dst);

    CHECK(dst[0] == 0);
    CHECK(dst[1] == 1);
    CHECK(dst[2] == 2);
    CHECK(dst[3] == 3);
}

TEST_CASE("Image view")
{
    ac::core::Image img{ 4, 4, 1, ac::core::Image::UInt8 };
    auto v = img.view(1, 1, 2, 2);

    CHECK(v.width() == 2);
    CHECK(v.height() == 2);
    CHECK(v.channels() == 1);
    CHECK(v.type() == img.type());
    CHECK(v.stride() == img.stride());
    CHECK(v == img);
}

TEST_CASE("Image view out of bounds")
{
    ac::core::Image img{ 4, 4, 1, ac::core::Image::UInt8 };

    auto v1 = img.view(-1, -1, 2, 2);
    CHECK(v1.width() == 1);
    CHECK(v1.height() == 1);

    auto v2 = img.view(3, 3, 4, 4);
    CHECK(v2.width() == 1);
    CHECK(v2.height() == 1);

    auto v3 = img.view(5, 5, 2, 2);
    CHECK(v3.empty());
}

TEST_CASE("Image clone")
{
    ac::core::Image img{ 2, 2, 1, ac::core::Image::UInt8 };
    for (int i = 0; i < img.height(); i++)
        for (int j = 0; j < img.width(); j++)
            *img.pixel(j, i) = static_cast<std::uint8_t>(42);

    auto c = img.clone();
    CHECK(c.width() == img.width());
    CHECK(c.height() == img.height());
    CHECK(c.channels() == img.channels());
    CHECK(c.type() == img.type());
    CHECK(c.ownership());
    CHECK(c != img);
    CHECK(c.ptr() != img.ptr());

    for (int i = 0; i < img.height(); i++)
        for (int j = 0; j < img.width(); j++)
            CHECK(*c.pixel(j, i) == *img.pixel(j, i));
}

TEST_CASE("Image copy constructor")
{
    ac::core::Image img{ 2, 2, 1, ac::core::Image::UInt8 };
    ac::core::Image copy{ img };
    CHECK(copy.width() == img.width());
    CHECK(copy.height() == img.height());
    CHECK(copy == img);

    auto v = img.view(0, 0, 1, 1);
    ac::core::Image copyView{ v };
    CHECK(copyView.width() == 1);
    CHECK(copyView == img);
}

TEST_CASE("Image move constructor")
{
    ac::core::Image img{ 2, 2, 1, ac::core::Image::UInt8 };
    void* origPtr = img.ptr();

    ac::core::Image moved{ std::move(img) };
    CHECK(moved.ptr() == origPtr);
    CHECK(!moved.empty());
}

TEST_CASE("Image copy assignment")
{
    ac::core::Image img{ 2, 2, 1, ac::core::Image::UInt8 };
    ac::core::Image assigned{};
    assigned = img;

    CHECK(assigned.width() == img.width());
    CHECK(assigned == img);
}

TEST_CASE("Image move assignment")
{
    ac::core::Image img{ 2, 2, 1, ac::core::Image::UInt8 };
    auto p = img.ptr();

    ac::core::Image moved{};
    moved = std::move(img);
    CHECK(moved.ptr() == p);
}

TEST_CASE("Image element type constants")
{
    CHECK(ac::core::Image::UInt8 == 1);
    CHECK(ac::core::Image::UInt16 == 2);
    CHECK(ac::core::Image::Float16 == 514);
    CHECK(ac::core::Image::Float32 == 516);
}

TEST_CASE("Image float type")
{
    ac::core::Image img{ 2, 2, 1, ac::core::Image::Float32 };
    CHECK(img.type() == ac::core::Image::Float32);
    CHECK(img.elementSize() == 4);
    CHECK(img.pixelSize() == 4);
    CHECK(img.isFloat());
    CHECK(!img.isUint());
    CHECK(!img.isInt());
}

TEST_CASE("Image line and pixel access")
{
    ac::core::Image img{ 4, 3, 3, ac::core::Image::UInt8 };
    auto d = img.data();

    std::uint8_t* line0 = img.line(0);
    std::uint8_t* line1 = img.line(1);
    CHECK(line0 == d);
    CHECK(line1 == d + img.stride());

    std::uint8_t* p21 = img.pixel(2, 1);
    CHECK(p21 == line1 + 2 * 3);

    CHECK(img.ptr(0) == static_cast<void*>(line0));
    CHECK(img.ptr(2, 1) == static_cast<void*>(p21));
}
