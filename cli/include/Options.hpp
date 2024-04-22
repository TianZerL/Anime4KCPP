#ifndef AC_CLI_OPTIONS_CPP
#define AC_CLI_OPTIONS_CPP

#include <string>

struct Options
{
    std::string input{};
    std::string output{"output.jpg"};
    std::string model{"acnet"};
    std::string processor{"cpu"};
    int device = 0;
    double factor = 2.0;
    bool list = false;
    bool version = false;
};

Options parse(int argc, const char* argv[]) noexcept;

#endif
