#ifndef AC_CLI_OPTIONS_CPP
#define AC_CLI_OPTIONS_CPP

#include <string>

struct Options
{
    std::string input{};
    std::string output{};
    std::string model{"acnet"};
    std::string processor{"cpu"};
    double factor = 2.0;
    int device = 0;
    bool list = false;
    bool version = false;
    struct {
        // decoder hints
        std::string decoder{};
        // encoder hints
        std::string encoder{};
        int bitrate = 0;

        bool enable = false;
        operator bool() const noexcept { return enable; }
    } video;
};

Options parse(int argc, const char* argv[]) noexcept;

#endif
