#include <cmath>
#include <cstdint>
#include <cstdlib>

#include <doctest/doctest.h>

#include "AC/Core/Internal/Util.hpp"

TEST_CASE("toFloat uint8_t")
{
    CHECK(ac::core::toFloat(std::uint8_t{0}) == doctest::Approx(0.0f));
    CHECK(ac::core::toFloat(std::uint8_t{255}) == doctest::Approx(1.0f));
    CHECK(ac::core::toFloat(std::uint8_t{128}) == doctest::Approx(128.0f / 255.0f));
}

TEST_CASE("fromFloat uint8_t")
{
    CHECK(ac::core::fromFloat<std::uint8_t>(0.0f) == 0);
    CHECK(ac::core::fromFloat<std::uint8_t>(1.0f) == 255);
    CHECK(ac::core::fromFloat<std::uint8_t>(-0.1f) == 0);
    CHECK(ac::core::fromFloat<std::uint8_t>(1.1f) == 255);
}

TEST_CASE("toFloat / fromFloat round-trip uint8_t")
{
    for (int v = 0; v <= 255; v++)
    {
        auto f = ac::core::toFloat(static_cast<std::uint8_t>(v));
        auto r = ac::core::fromFloat<std::uint8_t>(f);
        CHECK(r == v);
    }
}

TEST_CASE("toFloat / fromFloat round-trip uint16_t")
{
    for (int v = 0; v <= 255; v++)
    {
        auto f = ac::core::toFloat(static_cast<std::uint16_t>(v));
        auto r = ac::core::fromFloat<std::uint16_t>(f);
        CHECK(r == v);
    }
}

TEST_CASE("toFloat int8_t")
{
    CHECK(ac::core::toFloat(std::int8_t{-128}) == doctest::Approx(0.0f));
    CHECK(ac::core::toFloat(std::int8_t{0}) == doctest::Approx(0.5f).epsilon(0.01));
    CHECK(ac::core::toFloat(std::int8_t{127}) == doctest::Approx(1.0f));
}

TEST_CASE("fromFloat int8_t")
{
    CHECK(ac::core::fromFloat<std::int8_t>(0.0f) == -128);
    CHECK(ac::core::fromFloat<std::int8_t>(1.0f) == 127);

    auto mid = ac::core::fromFloat<std::int8_t>(0.5f);
    CHECK((mid == 0 || mid == -1));
}

TEST_CASE("toFloat float passthrough")
{
    CHECK(ac::core::toFloat(0.5f) == doctest::Approx(0.5f));
    CHECK(ac::core::toFloat(1.0f) == doctest::Approx(1.0f));
    CHECK(ac::core::toFloat(-1.0f) == doctest::Approx(-1.0f));
}

TEST_CASE("toFloat uint16_t")
{
    CHECK(ac::core::toFloat(std::uint16_t{0}) == doctest::Approx(0.0f));
    CHECK(ac::core::toFloat(std::uint16_t{65535}) == doctest::Approx(1.0f));
}

TEST_CASE("ceilLog2")
{
    CHECK(ac::core::ceilLog2(0.5) == -1);
    CHECK(ac::core::ceilLog2(1.0) == 0);
    CHECK(ac::core::ceilLog2(1.5) == 1);
    CHECK(ac::core::ceilLog2(2.0) == 1);
    CHECK(ac::core::ceilLog2(3.0) == 2);
    CHECK(ac::core::ceilLog2(4.0) == 2);
    CHECK(ac::core::ceilLog2(7.0) == 3);
    CHECK(ac::core::ceilLog2(8.0) == 3);
    CHECK(ac::core::ceilLog2(255.0) == 8);
    CHECK(ac::core::ceilLog2(256.0) == 8);
}

TEST_CASE("fastMalloc / fastFree")
{
    void* ptr = ac::core::fastMalloc(1024);
    REQUIRE(ptr != nullptr);

    auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    CHECK((addr % 16) == 0);

    ac::core::fastFree(ptr);
}

TEST_CASE("fastMalloc zero size")
{
    void* ptr = ac::core::fastMalloc(0);
    if (ptr) ac::core::fastFree(ptr);
}
