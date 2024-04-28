#include <iostream>

#include "AC/Core.hpp"
#include "AC/Util/Stopwatch.hpp"

#include "Options.hpp"

#define CHECK_ERROR(P) if (!(P)->ok()) { std::cout << (P)->error() << '\n'; return 0; }

static void version()
{
    std::cout <<
            "Anime4KCPP CLI:\n"
            "  core version: " AC_CORE_VERSION_STR "\n"
            "  build date: " AC_CORE_BUILD_DATE "\n"
            "  built by: " AC_CORE_COMPILER_ID " (v" AC_CORE_COMPILER_VERSION ")\n\n"
            "Copyright (c) by TianZerL the Anime4KCPP project 2020-" AC_CORE_BUILD_YEAR "\n"
            "https://github.com/TianZerL/Anime4KCPP\n";
}

static void list()
{
    std::cout << ac::core::Processor::info<ac::core::Processor::CPU>();
#   ifdef AC_CORE_WITH_OPENCL
        std::cout << ac::core::Processor::info<ac::core::Processor::OpenCL>();
#   endif
#   ifdef AC_CORE_WITH_CUDA
        std::cout << ac::core::Processor::info<ac::core::Processor::CUDA>();
#   endif
}

int main(int argc, const char* argv[])
{
    auto options = parse(argc, argv);

    if (options.version)
    {
        version();
        return 0;
    }

    if (options.list)
    {
        list();
        return 0;
    }

    if (options.input.empty()) return 0;

    auto processor = [&]() {
        ac::core::model::ACNet model { [&]() {
            if(options.model.find('1') != std::string::npos)
            {
                options.model = "ACNet HDN1";
                return ac::core::model::ACNet::Variant::HDN1 ;
            }
            if(options.model.find('2') != std::string::npos)
            {
                options.model = "ACNet HDN2";
                return ac::core::model::ACNet::Variant::HDN2 ;
            }
            if(options.model.find('3') != std::string::npos)
            {
                options.model = "ACNet HDN3";
                return ac::core::model::ACNet::Variant::HDN3 ;
            }
            options.model = "ACNet HDN0";
            return ac::core::model::ACNet::Variant::HDN0 ;
        }() };

#       ifdef AC_CORE_WITH_OPENCL
            if (options.processor == "opencl") return ac::core::Processor::create<ac::core::Processor::OpenCL>(options.device, model);
#       endif
#       ifdef AC_CORE_WITH_CUDA
            if (options.processor == "cuda") return ac::core::Processor::create<ac::core::Processor::CUDA>(options.device, model);
#       endif
        options.processor = "cpu";
        return ac::core::Processor::create<ac::core::Processor::CPU>(options.device, model);
    }();
    CHECK_ERROR(processor);

    std::cout << "Model: " << options.model << '\n';
    std::cout << "Processor: " << options.processor << ' ' << processor->name() << '\n';

    auto src = ac::core::imread(options.input.c_str(), ac::core::IMREAD_UNCHANGED);

    ac::util::Stopwatch stopwatch{};
    auto dst = processor->process(src, options.factor);
    stopwatch.stop();
    CHECK_ERROR(processor);
    std::cout << "Finished in: " << stopwatch.elapsed() << "s\n";

    if (ac::core::imwrite(options.output.c_str(), dst)) std::cout << "Save to "<< options.output << '\n';
    else std::cout << "Failed to save file\n";

    return 0;
}
