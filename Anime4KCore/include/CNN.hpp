#pragma once

namespace Anime4KCPP
{
    class CNNType;
}

class Anime4KCPP::CNNType
{
public:
    enum Value { Default, ACNetHDNL0, ACNetHDNL1, ACNetHDNL2, ACNetHDNL3 };

    CNNType() = default;
    constexpr CNNType(Value v) : value(v) { }

    explicit operator bool() = delete;
    constexpr operator Value() const { return value; }

    constexpr const char* toString() const
    {
        switch (value)
        {
        case Anime4KCPP::CNNType::Default:
            return "ACNetHDNL0";
            break;
        case Anime4KCPP::CNNType::ACNetHDNL0:
            return "ACNetHDNL0";
            break;
        case Anime4KCPP::CNNType::ACNetHDNL1:
            return "ACNetHDNL1";
            break;
        case Anime4KCPP::CNNType::ACNetHDNL2:
            return "ACNetHDNL2";
            break;
        case Anime4KCPP::CNNType::ACNetHDNL3:
            return "ACNetHDNL3";
            break;
        default:
            return "ACNetHDNL0";
            break;
        }
    }

private:
    Value value;
};
