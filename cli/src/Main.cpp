#include <cstdio>
#include <iterator>
#include <memory>

#include "AC/Core.hpp"
#include "AC/Specs.hpp"
#include "AC/Util/Stopwatch.hpp"
#include "AC/Util/ThreadPool.hpp"

#include "Options.hpp"

#ifdef AC_CLI_ENABLE_VIDEO
#   include <atomic>
#   include "AC/Video.hpp"
#   include "ProgressBar.hpp"
#endif

static bool list(const Options& options)
{
    if (options.list.version)
    {
        std::printf(
            "Anime4KCPP CLI:\n"
            "  core version: " AC_CORE_VERSION_STR " (" AC_CORE_FEATURES ")\n"
            "  video module: "
#           ifdef AC_CLI_ENABLE_VIDEO
                AC_VIDEO_VERSION_STR "\n"
#           else
                "disabled\n"
#           endif
            "  build date: " AC_BUILD_DATE "\n"
            "  toolchain: " AC_COMPILER_ID " (v" AC_COMPILER_VERSION ")\n"
            "  license: "
#           ifdef AC_CLI_ENABLE_VIDEO
                "GPLv3\n\n"
#           else
                "MIT\n\n"
#           endif
            "Copyright (c) 2020-" AC_BUILD_YEAR " the Anime4KCPP project\n\n"
            "https://github.com/TianZerL/Anime4KCPP\n"
        );
        return true;
    }
    if (options.list.devices)
    {
        std::printf("%s", ac::core::Processor::listInfo());
        return true;
    }
    if (options.list.processors)
    {
        printf("Processors:\n");
        for (std::size_t i = 0; i < std::size(ac::specs::ProcessorList); i++) printf("  %-16s  %s\n", ac::specs::ProcessorList[i], ac::specs::ProcessorDescriptionList[i]);
        return true;
    }
    if (options.list.models)
    {
        printf("Models:\n");
        for (std::size_t i = 0; i < std::size(ac::specs::ModelList); i++) printf("  %-16s  %s\n", ac::specs::ModelList[i], ac::specs::ModelDescriptionList[i]);
        return true;
    }
    return false;
}

static void image(const std::shared_ptr<ac::core::Processor>& processor, Options& options)
{
    auto batch = options.inputs.size();
    auto threads = ac::util::ThreadPool::hardwareThreads();
    auto targetThreads = processor->type() == ac::core::Processor::CPU ? threads / 4 + 1 : threads / 2 + 1;
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
        auto elapsed = stopwatch.elapsed();
        ac::util::Stopwatch::FormatBuffer elapsedBuffer{};
        std::printf("%s: Finished in %lfs (%s)\n", input.c_str(), elapsed, ac::util::Stopwatch::formatDuration(elapsedBuffer, elapsed));

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
    ehints.encoder = options.video.encoder.c_str();
    ehints.format = options.video.format.c_str();
    ehints.bitrate = options.video.bitrate * 1000;

    for (decltype(options.inputs.size()) i = 0; i < options.inputs.size(); i++)
    {
        auto& input = options.inputs[i];
        auto& output = options.outputs[i];

        if (output.empty()) output = input + ".out.mkv";

        ac::video::Pipeline pipeline{};

        std::printf("Load video from %s\n", input.c_str());
        if (!pipeline.openDecoder(input.c_str(), dhints))
        {
            std::printf("%s: Failed to open decoder\n", input.c_str());
            return;
        }
        if (!pipeline.openEncoder(output.c_str(), options.factor, ehints))
        {
            std::printf("%s: Failed to open encoder\n", input.c_str());
            return;
        }

        auto info = pipeline.getInfo();

        ac::util::Stopwatch stopwatch{};
        ProgressBar progressBar{};

        struct {
            int shift;
            double factor;
            double frames;
            std::shared_ptr<ac::core::Processor> processor;
            ProgressBar* progressBar;
            std::atomic<const char*> error;
        } data{};
        data.shift = info.bitDepth.lsb ? ((info.bitDepth.bits + 7) / 8 * 8 - info.bitDepth.bits) : 0; // bytes * 8 - bits
        data.factor = options.factor;
        data.frames = info.fps * info.duration;
        data.processor = processor;
        data.progressBar = &progressBar;
        data.error = nullptr;

        progressBar.reset();
        stopwatch.reset();
        ac::video::filter(pipeline, [](ac::video::Frame& src, ac::video::Frame& dst, void* userdata) -> bool {
            auto ctx = static_cast<decltype(data)*>(userdata);
            // y
            ac::core::Image srcy{ src.plane[0].width, src.plane[0].height, 1, src.elementType, src.plane[0].data, src.plane[0].stride };
            ac::core::Image dsty{ dst.plane[0].width, dst.plane[0].height, 1, dst.elementType, dst.plane[0].data, dst.plane[0].stride };
            if (ctx->shift) ac::core::shl(srcy, srcy, ctx->shift); // the src frame is decoded from ffmpeg and cannot be directly modified
            ctx->processor->process(srcy, dsty, ctx->factor);
            if (!ctx->processor->ok())
            {
                ctx->error.store(ctx->processor->error(), std::memory_order_relaxed);
                return false;
            }
            if (ctx->shift) ac::core::shr(dsty, ctx->shift);
            // uv
            for (int i = 1; i < src.planes; i++)
            {
                ac::core::Image srcp{ src.plane[i].width, src.plane[i].height, src.plane[i].channel, src.elementType, src.plane[i].data, src.plane[i].stride };
                ac::core::Image dstp{ dst.plane[i].width, dst.plane[i].height, dst.plane[i].channel, dst.elementType, dst.plane[i].data, dst.plane[i].stride };
                ac::core::resize(srcp, dstp, 0.0, 0.0);
            }
            // a beautiful progress bar
            if (src.number % 32 == 0) ctx->progressBar->print(src.number / ctx->frames);
            return true;
        }, &data, ac::video::FILTER_AUTO);
        stopwatch.stop();
        progressBar.finish();
        pipeline.close();
        if (data.error.load(std::memory_order_relaxed)) std::printf("%s: Failed due to %s\n", input.c_str(), data.error.load(std::memory_order_relaxed));
        else
        {
            auto elapsed = stopwatch.elapsed();
            ac::util::Stopwatch::FormatBuffer elapsedBuffer{};
            std::printf("%s: Finished in %lfs (%s)\n", input.c_str(), elapsed, ac::util::Stopwatch::formatDuration(elapsedBuffer, elapsed));
        }
        std::printf("Save video to %s\n", output.c_str());
    }
#else
    std::printf("This build does not support video processing\n");
#endif
}

int main(int argc, char* argv[])
{
    auto options = parse(argc, argv);

    if (list(options)) return 0;

    if (options.inputs.empty()) return 0;
    options.outputs.resize(options.inputs.size());

    auto processor = ac::core::Processor::create(options.processor.c_str(), options.device, options.model.c_str());
    if (!processor->ok())
    {
        std::printf("%s\n", processor->error());
        return 0;
    }

    std::printf("Model: %s\n"
                "Processor: %s %s\n\n", options.model.c_str(), processor->typeName(), processor->name());

    ac::util::Stopwatch stopwatch{};
    if (options.video)
        video(processor, options);
    else
        image(processor, options);
    stopwatch.stop();

    auto elapsed = stopwatch.elapsed();
    ac::util::Stopwatch::FormatBuffer elapsedBuffer{};
    std::printf("\nInputs %d files, takes %lfs (%s)\n", static_cast<int>(options.inputs.size()), elapsed, ac::util::Stopwatch::formatDuration(elapsedBuffer, elapsed));

    return 0;
}
