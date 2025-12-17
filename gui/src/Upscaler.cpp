#include <atomic>

#include "AC/Core.hpp"
#include "AC/Util/Defer.hpp"
#include "AC/Util/Stopwatch.hpp"
#include "AC/Util/ThreadPool.hpp"

#include "Config.hpp"
#include "Logger.hpp"

#include "Upscaler.hpp"

#ifdef AC_CLI_ENABLE_VIDEO
#   include "AC/Video.hpp"
#endif

struct Upscaler::UpscalerData
{
    int device = 0;
    double factor = 2.0;
    QString processorType{};
    QString model{};
    std::shared_ptr<ac::core::Processor> processor{};
    std::atomic_bool stopFlag = false;
    std::atomic_size_t total = 0;
};

Upscaler& Upscaler::instance() noexcept
{
    static Upscaler upscaler{};
    return upscaler;
}
QString& Upscaler::listProcessorInfo()
{
    static QString buffer{ ac::core::Processor::listInfo() };
    return buffer;
}

Upscaler::Upscaler() : dptr(std::make_unique<UpscalerData>()) {}
Upscaler::~Upscaler() noexcept = default;

void Upscaler::start(const QList<QSharedPointer<TaskData>>& taskList)
{
    if (dptr->total.load(std::memory_order_relaxed)) return;
    dptr->factor = gConfig.upscaler.factor;

    if (!dptr->processor || dptr->device != gConfig.upscaler.device || dptr->processorType != gConfig.upscaler.processor || dptr->model != gConfig.upscaler.model)
    {
        dptr->processor = ac::core::Processor::create(gConfig.upscaler.processor.toLocal8Bit(), gConfig.upscaler.device, gConfig.upscaler.model.toLocal8Bit());
        dptr->device = gConfig.upscaler.device;
        dptr->processorType = gConfig.upscaler.processor;
        dptr->model = gConfig.upscaler.model;
    }
    if (!dptr->processor->ok())
    {
        gLogger.error() << dptr->processor->error();
        return;
    }

    dptr->stopFlag.store(false, std::memory_order_relaxed);
    dptr->total.store(taskList.size(), std::memory_order_relaxed);
    if (!dptr->total.load(std::memory_order_relaxed)) return;

    emit started();

    QList<QSharedPointer<TaskData>> imageTaskList;
    QList<QSharedPointer<TaskData>> videoTaskList;
    for (auto&& task : taskList)
    {
        if (task->type == TaskData::TYPE_IMAGE)
            imageTaskList << task;
        else
            videoTaskList << task;
    }

    static auto threads = ac::util::ThreadPool::hardwareThreads();
    static ac::util::ThreadPool pool{ dptr->processor->type() == ac::core::Processor::CPU ? threads / 4 + 1 : threads / 2 + 1 };

#ifdef AC_CLI_ENABLE_VIDEO
    pool.exec([=](){
        auto decoder = gConfig.video.decoder.toLocal8Bit();
        auto format = gConfig.video.format.toLocal8Bit();
        auto encoder = gConfig.video.encoder.toLocal8Bit();
        auto bitrate = gConfig.video.bitrate * 1000;

        ac::video::DecoderHints dhints{};
        ac::video::EncoderHints ehints{};
        dhints.decoder = decoder;
        ehints.encoder = encoder;
        ehints.format = format;
        ehints.bitrate = bitrate;

        for (auto&& task : videoTaskList)
        {
            ac::util::Defer defer([this]() { if (dptr->total.fetch_sub(1, std::memory_order_relaxed) == 1) emit stopped(); });
            emit progress(0);
            if (!dptr->stopFlag.load(std::memory_order_relaxed))
            {
                ac::video::Pipeline pipeline{};

                gLogger.info() << "Load video from " << task->path.input;
                if (!pipeline.openDecoder(task->path.input.toUtf8(), dhints)) // ffmpeg api uses utf8 for io
                {
                    gLogger.error() << task->path.input << ": Failed to open decoder";
                    emit task->finished(false);
                    continue;
                }
                if (!pipeline.openEncoder(task->path.output.toUtf8(), dptr->factor, ehints))
                {
                    gLogger.error() << task->path.input << ": Failed to open encoder";
                    emit task->finished(false);
                    continue;
                }

                auto info = pipeline.getInfo();
                struct {
                    const std::atomic_bool& stopFlag;
                    int shift;
                    double factor;
                    double frames;
                    Upscaler* upscaler;
                    std::shared_ptr<ac::core::Processor> processor;
                    std::atomic<const char*> error;
                } data {
                    dptr->stopFlag,
                    info.bitDepth.lsb ? ((info.bitDepth.bits + 7) / 8 * 8 - info.bitDepth.bits) : 0, // bytes * 8 - bits
                    dptr->factor,
                    info.fps * info.duration,
                    this,
                    dptr->processor,
                    nullptr
                };
                ac::util::Stopwatch stopwatch{};
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
                    if (src.number % 32 == 0) emit ctx->upscaler->progress(static_cast<int>(100 * src.number / ctx->frames));
                    if (ctx->stopFlag.load(std::memory_order_relaxed))
                    {
                        ctx->error.store("cancelled", std::memory_order_relaxed);
                        return false;
                    }
                    return true;
                }, &data, ac::video::FILTER_AUTO);
                stopwatch.stop();
                pipeline.close();
                if (data.error.load(std::memory_order_relaxed)) gLogger.error() << task->path.input << ": Failed due to " << data.error.load(std::memory_order_relaxed);
                else gLogger.info() << task->path.input << ": Finished in " << stopwatch.elapsed() << "s [" << dptr->processor->typeName() << ' ' << dptr->processor->name() << ']';
                gLogger.info() << "Save video to " << task->path.output;
            }
            emit progress(100);
            emit task->finished(!dptr->stopFlag.load(std::memory_order_relaxed) && dptr->processor->ok());
        }
    });
#else
    gLogger.warning() << "This build does not support video processing";
    for (auto&& task : videoTaskList)
    {
        emit task->finished(false);
        if (dptr->total.fetch_sub(1, std::memory_order_relaxed) == 1) emit stopped();
    }
#endif

    for (auto&& task : imageTaskList)
    {
        pool.exec([=]() {
            ac::util::Defer defer([this]() { if (dptr->total.fetch_sub(1, std::memory_order_relaxed) == 1) emit stopped(); });
            if (!dptr->stopFlag.load(std::memory_order_relaxed))
            {
                auto src = ac::core::imread(task->path.input.toLocal8Bit(), ac::core::IMREAD_UNCHANGED);
                if (!src.empty())
                    gLogger.info() << "Load image from " << task->path.input;
                else
                {
                    gLogger.error() << "Failed to load image from " << task->path.input;
                    emit task->finished(false);
                    return;
                }

                ac::util::Stopwatch stopwatch{};
                auto dst = dptr->processor->process(src, dptr->factor);
                stopwatch.stop();
                if (!dptr->processor->ok())
                {
                    gLogger.error() << task->path.input << ": Failed due to " << dptr->processor->error();
                    emit task->finished(false);
                    return;
                }
                gLogger.info() << task->path.input << ": Finished in " << stopwatch.elapsed() << "s [" << dptr->processor->typeName() << ' ' << dptr->processor->name() << ']';

                if (ac::core::imwrite(task->path.output.toLocal8Bit(), dst)) gLogger.info() << "Save image to " << task->path.output;
                else
                {
                    gLogger.error() << "Failed to save image to " << task->path.output;
                    emit task->finished(false);
                    return;
                }
            }
            emit task->finished(!dptr->stopFlag.load(std::memory_order_relaxed));
        });
    }
}
void Upscaler::stop()
{
    dptr->stopFlag.store(true, std::memory_order_relaxed);
}
