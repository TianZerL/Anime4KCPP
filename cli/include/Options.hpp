#ifndef AC_CLI_OPTIONS_HPP
#define AC_CLI_OPTIONS_HPP

#include <string>
#include <vector>

#include "AC/Specs.hpp"

struct Options
{
    std::vector<std::string> inputs{};
    std::vector<std::string> outputs{};
    std::string model{ "acnet-f8b8-hdn" };
    std::string processor{ "auto" };
    double factor = 2.0;
    int device = 0;
    unsigned int threads = 0;

    struct {
        // decoder hints
        std::string decoder{};
        std::string format{};
        // encoder hints
        std::string encoder{};
        int bitrate = 0;

        bool enable = false;
        operator bool() const noexcept { return enable; }
    } video;

    struct
    {
        bool devices = false;
        bool models = false;
        bool processors = false;
        bool version = false;

        operator bool() const noexcept { return devices || models || processors || version; }
    } list;
};

Options parse(int argc, char** const argv) noexcept;

#endif
