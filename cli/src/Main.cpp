#include <cstdio>
#include <memory>

#include "AC/Core.hpp"
#include "AC/Util/Stopwatch.hpp"
#include "AC/Util/ThreadPool.hpp"
#ifdef AC_CLI_ENABLE_VIDEO
#   include "AC/Video.hpp"
#endif

#include "Options.hpp"

#define PROGRESS_BAR_TOKEN "============================================================"

static void version()
{
    std::printf(
        "Anime4KCPP CLI:\n"
        "  core version: " AC_CORE_VERSION_STR " (" AC_CORE_FEATURES ")\n"
        "  video module: "
#       ifdef AC_CLI_ENABLE_VIDEO
            AC_VIDEO_VERSION_STR "\n"
#       else
            "disabled\n"
#       endif
        "  build date: " AC_BUILD_DATE "\n"
        "  toolchain: " AC_COMPILER_ID " (v" AC_COMPILER_VERSION ")\n"
        "  license: "
#       ifdef AC_CLI_ENABLE_VIDEO
            "GPLv3\n\n"
#       else
            "MIT\n\n"
#       endif
        "Copyright (c) 2020-" AC_BUILD_YEAR " the Anime4KCPP project\n\n"
        "https://github.com/TianZerL/Anime4KCPP\n"
    );
}

static void list()
{
    std::printf("%s", ac::core::Processor::info<ac::core::Processor::CPU>());
#   ifdef AC_CORE_WITH_OPENCL
        std::printf("%s", ac::core::Processor::info<ac::core::Processor::OpenCL>());
#   endif
#   ifdef AC_CORE_WITH_CUDA
        std::printf("%s", ac::core::Processor::info<ac::core::Processor::CUDA>());
#   endif
}

static void image(const std::shared_ptr<ac::core::Processor>& processor, Options& options)
{
    auto batch = options.inputs.size();
    auto threads = ac::util::ThreadPool::hardwareThreads();
    auto targetThreads = options.processor == "cpu" ? threads / 4 + 1 : threads / 2 + 1;
    auto poolSize = batch > targetThreads ? targetThreads : batch;
    auto task = [&](const int i) {
        auto& input = options.inputs[i];
        auto& output = options.outputs[i];

        if (output.empty()) output = input + ".out.jpg";

        auto src = ac::core::imread(input.c_str(), ac::core::IMREAD_UNCHANGED);
        if (!src.empty())
            std::printf("Load image from %s\n", input.c_str());
        else
        {
            std::printf("Failed to load image from %s\n", input.c_str());
            return;
        }

        ac::util::Stopwatch stopwatch{};
        auto dst = processor->process(src, options.factor);
        stopwatch.stop();
        if (!processor->ok())
        {
            std::printf("%s: Failed due to %s\n", input.c_str(), processor->error());
            return;
        }
        std::printf("%s: Finished in %lfs\n",input.c_str() ,stopwatch.elapsed());

        if (ac::core::imwrite(output.c_str(), dst))
            std::printf("Save image to %s\n", output.c_str());
        else
        {
            std::printf("Failed to save image to %s\n", output.c_str());
            return;
        }
    };
    if (poolSize > 1)
    {
        ac::util::ThreadPool pool{ poolSize };
        for (decltype(batch) i = 0; i < batch; i++) pool.exec([i, &task]() { task(i); });
    }
    else for (decltype(batch) i = 0; i < batch; i++) task(i);
}

static void video([[maybe_unused]] const std::shared_ptr<ac::core::Processor>& processor, [[maybe_unused]] Options& options)
{
#ifdef AC_CLI_ENABLE_VIDEO
    ac::video::DecoderHints dhints{};
    ac::video::EncoderHints ehints{};
    dhints.decoder = options.video.decoder.c_str();
    dhints.format = options.video.format.c_str();
    ehints.encoder = options.video.encoder.c_str();
    ehints.bitrate = options.video.bitrate * 1000;

    for (decltype(options.inputs.size()) i = 0; i < options.inputs.size(); i++)
    {
        auto& input = options.inputs[i];
        auto& output = options.outputs[i];

        if (output.empty()) output = input + ".out.mp4";

        ac::video::Pipeline pipeline{};

        std::printf("Load video from %s\n", input.c_str());
        if(!pipeline.openDecoder(input.c_str(), dhints))
        {
            std::printf("%s: Failed to open decoder\n", input.c_str());
            return;
        }
        if(!pipeline.openEncoder(output.c_str(), options.factor, ehints))
        {
            std::printf("%s: Failed to open encoder\n", input.c_str());
            return;
        }

        auto info = pipeline.getInfo();

        struct {
            int shift;
            double factor;
            double frames;
            std::shared_ptr<ac::core::Processor> processor;
        } data{};
        data.shift = info.bitDepth.lsb ? ((info.bitDepth.bits - 1) / 8 + 1) * 8 - info.bitDepth.bits : 0; // bytes * 8 - bits
        data.factor = options.factor;
        data.frames = info.fps * info.duration;
        data.processor = processor;

        ac::util::Stopwatch stopwatch{};
        ac::video::filter(pipeline, [](ac::video::Frame& src, ac::video::Frame& dst, void* userdata) -> bool {
            auto ctx = static_cast<decltype(data)*>(userdata);
            // y
            ac::core::Image srcy{src.plane[0].width, src.plane[0].height, 1, src.elementType, src.plane[0].data, src.plane[0].stride};
            ac::core::Image dsty{dst.plane[0].width, dst.plane[0].height, 1, dst.elementType, dst.plane[0].data, dst.plane[0].stride};
            if (ctx->shift) ac::core::shl(srcy, srcy, ctx->shift); // the src frame is decoded from ffmpeg and cannot be directly modified
            ctx->processor->process(srcy, dsty, ctx->factor);
            if (!ctx->processor->ok()) return false;
            if (ctx->shift) ac::core::shr(dsty, ctx->shift);
            // uv
            for (int i = 1; i < src.planes; i++)
            {
                ac::core::Image srcp{src.plane[i].width, src.plane[i].height, src.plane[i].channel, src.elementType, src.plane[i].data, src.plane[i].stride};
                ac::core::Image dstp{dst.plane[i].width, dst.plane[i].height, dst.plane[i].channel, dst.elementType, dst.plane[i].data, dst.plane[i].stride};
                ac::core::resize(srcp, dstp, 0.0, 0.0);
            }
            // a beautiful progress bar
            if (src.number % 32 == 0)
            {
                constexpr int width = sizeof(PROGRESS_BAR_TOKEN) - 1;
                double p = src.number / ctx->frames;
                int done = static_cast<int>(p * width);
                int left = width - done;
                std::printf("\r%6.2lf%% [%.*s%-*s]", p * 100.0, done, PROGRESS_BAR_TOKEN, left, ">");
                std::fflush(stdout);
            }
            return true;
        }, &data, ac::video::FILTER_AUTO);
        stopwatch.stop();
        pipeline.close();
        if (!processor->ok()) std::printf("\r100.00%%\n%s: Failed due to %s\n", input.c_str(), processor->error());
        else std::printf("\r100.00%%\n%s: Finished in %lfs\n",input.c_str(), stopwatch.elapsed());
        std::printf("Save video to %s\n", output.c_str());
    }
#else
    std::printf("This build does not support video processing\n");
#endif
}

int main(int argc, char* argv[])
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

    if (options.inputs.empty()) return 0;
    options.outputs.resize(options.inputs.size());

    auto processor = ac::core::Processor::create(ac::core::Processor::type(options.processor.c_str()), options.device, options.model.c_str());
    if (!processor->ok())
    { 
        std::printf("%s\n", processor->error());
        return 0;
    }

    std::printf("Model: %s\n"
                "Processor: %s %s\n\n",
                options.model.c_str(), options.processor.c_str(), processor->name());

    ac::util::Stopwatch stopwatch{};
    if (options.video)
        video(processor, options);
    else
        image(processor, options);
    stopwatch.stop();

    std::printf("\nInputs %d files, takes %lfs\n", static_cast<int>(options.inputs.size()), stopwatch.elapsed());

    return 0;
}
