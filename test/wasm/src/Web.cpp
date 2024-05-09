#include <string>

#include <emscripten.h>
#include <emscripten/bind.h>

#include "AC/Core.hpp"
#include "AC/Util/Stopwatch.hpp"

static void upscale(const std::string filename, double factor, int device)
{
    auto processor = ac::core::Processor::create<ac::core::Processor::CPU>(device, ac::core::model::ACNet{ ac::core::model::ACNet::Variant::HDN0 });
    std::printf("Processor: %s\n", processor->name());

    ac::core::Image src = ac::core::imread(filename.c_str(), ac::core::IMREAD_UNCHANGED);

    ac::util::Stopwatch stopwatch{};
    ac::core::Image dst = processor->process(src, 2.0);
    stopwatch.stop();

    std::printf("Finished in: %lfs\n", stopwatch.elapsed());
    ac::core::imwrite("output.jpg", dst);
}

EMSCRIPTEN_BINDINGS(ac) {
    emscripten::function("upscale", &upscale);
}
