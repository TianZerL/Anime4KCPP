#include <cstdio>

#include "AC/Core.hpp"
#include "AC/Util/Stopwatch.hpp"
#ifdef AC_CLI_ENABLE_VIDEO
#   include "AC/Video.hpp"
#endif

#include "Options.hpp"

#define CHECK_FLAG(O, F, ...) if (O.F) { F(__VA_ARGS__); return 0; }
#define CHECK_PROCESSOR(P) if (!(P)->ok()) { std::printf("%s\n", (P)->error()); std::exit(0); }

static void version()
{
    std::printf(
            "Anime4KCPP CLI:\n"
            "  core version: " AC_CORE_VERSION_STR "\n"
            "  build date: " AC_CORE_BUILD_DATE "\n"
            "  built by: " AC_CORE_COMPILER_ID " (v" AC_CORE_COMPILER_VERSION ")\n\n"
            "Copyright (c) by TianZerL the Anime4KCPP project 2020-" AC_CORE_BUILD_YEAR "\n"
            "https://github.com/TianZerL/Anime4KCPP\n"
    );
}

static void list()
{
    std::printf(ac::core::Processor::info<ac::core::Processor::CPU>());
#   ifdef AC_CORE_WITH_OPENCL
        std::printf(ac::core::Processor::info<ac::core::Processor::OpenCL>());
#   endif
#   ifdef AC_CORE_WITH_CUDA
        std::printf(ac::core::Processor::info<ac::core::Processor::CUDA>());
#   endif
}

static void image(std::shared_ptr<ac::core::Processor> processor, Options& options)
{
    if (options.output.empty()) options.output = "output.jpg";

    auto src = ac::core::imread(options.input.c_str(), ac::core::IMREAD_UNCHANGED);

    ac::util::Stopwatch stopwatch{};
    auto dst = processor->process(src, options.factor);
    stopwatch.stop();
    CHECK_PROCESSOR(processor);
    std::printf("Finished in: %lfs\n", stopwatch.elapsed());

    if (ac::core::imwrite(options.output.c_str(), dst))
        std::printf("Save to %s\n", options.output.c_str());
    else
        std::printf("Failed to save file\n");
}

static void video(std::shared_ptr<ac::core::Processor> processor, Options& options)
{
#ifdef AC_CLI_ENABLE_VIDEO
    if (options.output.empty()) options.output = "output.mp4";

    ac::video::Pipeline pipeline{};
    ac::video::DecoderHints dhints{};
    ac::video::EncoderHints ehints{};
    dhints.decoder = options.video.decoder.c_str();
    ehints.encoder = options.video.encoder.c_str();
    ehints.bitrate = options.video.bitrate;

    if(!pipeline.openDecoder(options.input.c_str(), dhints))
    {
        std::printf("Failed to open decoder");
        return;
    }
    if(!pipeline.openEncoder(options.output.c_str(), options.factor, ehints))
    {
        std::printf("Failed to open encoder");
        return;
    }

    auto info = pipeline.getInfo();

    struct {
        double factor;
        double frames;
        std::shared_ptr<ac::core::Processor> processor;
    } data;
    data.factor = options.factor;
    data.frames = info.fps * info.length;
    data.processor = processor;

    ac::util::Stopwatch stopwatch{};
    ac::video::filter(pipeline, [](ac::video::Frame& src, ac::video::Frame& dst, void* userdata) {
        auto ctx = static_cast<decltype(data)*>(userdata);
        // y
        ac::core::Image srcy{src.plane[0].width, src.plane[0].height, 1, src.elementType, src.plane[0].data, src.plane[0].stride};
        ac::core::Image dsty{dst.plane[0].width, dst.plane[0].height, 1, dst.elementType, dst.plane[0].data, dst.plane[0].stride};
        ctx->processor->process(srcy, dsty, ctx->factor);
        // uv
        for (int i = 1; i < src.planes; i++)
        {
            ac::core::Image srcp{src.plane[i].width, src.plane[i].height, src.plane[i].channel, src.elementType, src.plane[i].data, src.plane[i].stride};
            ac::core::Image dstp{dst.plane[i].width, dst.plane[i].height, dst.plane[i].channel, dst.elementType, dst.plane[i].data, dst.plane[i].stride};
            ac::core::resize(srcp, dstp, 0.0, 0.0);
        }
        // progress
        if (src.number % 32 == 0) std::printf("%.2lf%%\r", 100 * src.number / ctx->frames);
    }, &data, ac::video::FILTER_AUTO);
    stopwatch.stop();
    pipeline.close();
    CHECK_PROCESSOR(processor);
    std::printf("Finished in: %lfs\n", stopwatch.elapsed());
    std::printf("Save to %s\n", options.output.c_str());
#else
    std::printf("This build does not support video processing");
#endif
}

int main(int argc, const char* argv[])
{
    auto options = parse(argc, argv);

    CHECK_FLAG(options, version);
    CHECK_FLAG(options, list);

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
    CHECK_PROCESSOR(processor);

    std::printf("Model: %s\n"
                "Processor: %s %s\n",
                options.model.c_str(), options.processor.c_str(), processor->name());

    CHECK_FLAG(options, video, processor, options);

    image(processor, options);

    return 0;
}
