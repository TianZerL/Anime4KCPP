#ifndef AC_CLI_OPTIONS_CPP
#define AC_CLI_OPTIONS_CPP

#include <string>
#include <vector>

struct Options
{
    std::vector<std::string> inputs{};
    std::vector<std::string> outputs{};
    std::string model{"acnet"};
    std::string processor{"cpu"};
    double factor = 2.0;
    int device = 0;
    bool list = false;
    bool version = false;
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
};

Options parse(int argc, const char* const* argv) noexcept;

#endif
