#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

#include "AC/Core/Model/Param/ACNet.p"
#include "AC/Core/Model/Param/ARNet.p"

#define PROCESS(NET, VARIANT) { process(NET, NET##_##VARIANT##_NHWC_kernels_float, #NET"_"#VARIANT"_NHWC_kernels_half"); process(NET, NET##_##VARIANT##_NHWC_biases_float, #NET"_"#VARIANT"_NHWC_biases_half"); }

template<std::float_round_style R>
static inline constexpr std::uint16_t overflow(std::uint32_t sign) noexcept
{
    switch (R)
    {
    case std::round_toward_infinity:
        return static_cast<std::uint16_t>(sign + 0x7c00 - (sign >> 15));
    case std::round_toward_neg_infinity:
        return static_cast<std::uint16_t>(sign + 0x7bff + (sign >> 15));
    case std::round_toward_zero:
        return static_cast<std::uint16_t>(sign | 0x7bff);
    default:
        return static_cast<std::uint16_t>(sign | 0x7c00);
    }
}
template<std::float_round_style R>
static inline constexpr std::uint16_t rounded(std::uint32_t value, int g, int s) noexcept
{
    switch (R)
    {
    case std::round_to_nearest:
        return static_cast<std::uint16_t>(value + (g & (s | value)));
    case std::round_toward_infinity:
        return static_cast<std::uint16_t>(value + (~(value >> 15) & (g | s)));
    case std::round_toward_neg_infinity:
        return static_cast<std::uint16_t>(value + ((value >> 15) & (g | s)));
    default:
        return static_cast<std::uint16_t>(value);
    }
}
template<std::float_round_style R>
static inline constexpr std::uint16_t underflow(std::uint32_t sign) noexcept
{
    switch (R)
    {
    case std::round_toward_infinity:
        return static_cast<std::uint16_t>(sign + 1 - (sign >> 15));
    case std::round_toward_neg_infinity:
        return static_cast<std::uint16_t>(sign + (sign >> 15));
    default:
        return static_cast<std::uint16_t>(sign);
    }
}
static std::uint16_t toHalf(float fp32) noexcept
{
    std::uint32_t fbits{};
    std::memcpy(&fbits, &fp32, sizeof(float));
    std::uint32_t sign = (fbits >> 16) & 0x8000;
    fbits &= 0x7fffffff;
    if (fbits >= 0x7f800000)
        return sign | 0x7c00 | ((fbits > 0x7f800000) ? (0x200 | ((fbits >> 13) & 0x3ff)) : 0);
    if (fbits >= 0x47800000)
        return overflow<std::numeric_limits<float>::round_style>(sign);
    if (fbits >= 0x38800000)
        return rounded<std::numeric_limits<float>::round_style>(sign | (((fbits >> 23) - 112) << 10) | ((fbits >> 13) & 0x3ff), (fbits >> 12) & 1, (fbits & 0xfff) != 0);
    if (fbits >= 0x33000000)
    {
        int i = 125 - (fbits >> 23);
        fbits = (fbits & 0x7fffff) | 0x800000;
        return rounded<std::numeric_limits<float>::round_style>(sign | (fbits >> (i + 1)), (fbits >> i) & 1, (fbits & ((1 << i) - 1)) != 0);
    }
    if (fbits != 0)
        return underflow<std::numeric_limits<float>::round_style>(sign);
    return sign;
}

template<std::size_t N>
void process(std::ofstream& file, const float(&array)[N], const char* name)
{
    std::vector<std::uint16_t> halfs{};
    halfs.resize(N);
    for (int i = 0; i < N; i++) halfs[i] = toHalf(array[i]);

    std::string head{}, tail{};
    head.append("alignas(32) constexpr std::uint16_t ").append(name).append("[] = {\n");
    tail.append("};\n");
    file << head << std::hex;
    for (std::size_t i = 0; i < N; i++)
    {
        file << "0x" << std::setw(4) << std::setfill('0') << std::right << halfs[i] << ',' << (((i + 1) % 8 == 0) ? '\n' : ' ');
    }
    file << tail << std::endl;
}

int main(int argc, char* argv[])
{
    std::string path{ "." };
    if (argc > 1) path = argv[1];

    std::ofstream ACNet(path + "/ACNet.half.p");
    PROCESS(ACNet, GAN);
    PROCESS(ACNet, HDN0);
    PROCESS(ACNet, HDN1);
    PROCESS(ACNet, HDN2);
    PROCESS(ACNet, HDN3);
    ACNet.close();

    std::ofstream ARNet(path + "/ARNet.half.p");
    PROCESS(ARNet, HDN);
    ARNet.close();

    return 0;
}
