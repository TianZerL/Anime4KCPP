#include <atomic>

#include "AC/Core.hpp"
#include "AC/Util/ThreadPool.hpp"
#include "AC/Util/Stopwatch.hpp"
#include "AC/Util/Defer.hpp"
#ifdef AC_CLI_ENABLE_VIDEO
#   include "AC/Video.hpp"
#endif

#include "Config.hpp"
#include "Logger.hpp"

#include "Upscaler.hpp"

struct Upscaler::UpscalerData
{
    int processorType = ac::core::Processor::CPU;
    int device = 0;
    double factor = 2.0;
    std::atomic_bool stopFlag = false;
    std::atomic_size_t total = 0;
    std::shared_ptr<ac::core::Processor> processor{};
    QString model{};
};

Upscaler& Upscaler::instance() noexcept
{
    static Upscaler upscaler{};
    return upscaler;
}
QString Upscaler::info()
{
    QString buffer{};
    buffer.append(ac::core::Processor::info<ac::core::Processor::CPU>());
#   ifdef AC_CORE_WITH_OPENCL
        buffer.append(ac::core::Processor::info<ac::core::Processor::OpenCL>());
#   endif
#   ifdef AC_CORE_WITH_CUDA
        buffer.append(ac::core::Processor::info<ac::core::Processor::CUDA>());
#   endif
    return buffer;
}

Upscaler::Upscaler() : dptr(std::make_unique<UpscalerData>()) {}
Upscaler::~Upscaler() noexcept = default;

void Upscaler::start(const QList<QSharedPointer<TaskData>>& taskList)
{
    if (dptr->total) return;
    dptr->device = gConfig.upscaler.device;
    dptr->factor = gConfig.upscaler.factor;
    dptr->stopFlag = false;
    dptr->total = taskList.size();
    dptr->model = gConfig.upscaler.model;
    if (!dptr->total) return;

    int processorType = ac::core::Processor::type(gConfig.upscaler.processor.toLocal8Bit());
    if (!dptr->processor || dptr->processorType != processorType) dptr->processor = ac::core::Processor::create(dptr->processorType = processorType, dptr->device, dptr->model.toLocal8Bit());
    if (!dptr->processor->ok())
    {
        gLogger.error() << dptr->processor->error();
        return;
    }
    emit started();

    QList<QSharedPointer<TaskData>> imageTaskList;
    QList<QSharedPointer<TaskData>> videoTaskList;
    for(auto&& task : taskList)
    {
        if (task->type == TaskData::TYPE_IMAGE)
            imageTaskList << task;
        else
            videoTaskList << task;
    }

    static auto threads = ac::util::ThreadPool::hardwareThreads();
    static ac::util::ThreadPool pool{ dptr->processorType == ac::core::Processor::CPU ? threads / 4 + 1 : threads / 2 + 1 };

#   ifdef AC_CLI_ENABLE_VIDEO
        pool.exec([=](){
            auto decoder =  gConfig.video.decoder.toLocal8Bit();
            auto format =  gConfig.video.format.toLocal8Bit();
            auto encoder =  gConfig.video.encoder.toLocal8Bit();
            auto bitrate = gConfig.video.bitrate * 1000;

            ac::video::DecoderHints dhints{};
            ac::video::EncoderHints ehints{};
            dhints.decoder = decoder;
            dhints.format = format;
            ehints.encoder = encoder;
            ehints.bitrate = bitrate;

            for (auto&& task : videoTaskList)
            {
                ac::util::Defer defer([&]() { if (--dptr->total == 0) emit stopped(); });
                emit progress(0);
                if (!dptr->stopFlag)
                {
                    ac::video::Pipeline pipeline{};

                    gLogger.info() << "Load video from " << task->path.input;
                    if (!pipeline.openDecoder(task->path.input.toUtf8(), dhints)) // ffmpeg api uses utf8 for io
                    {
                        gLogger.error() << task->path.input <<": Failed to open decoder";
                        emit task->finished(false);
                        continue;
                    }
                    if (!pipeline.openEncoder(task->path.output.toUtf8(), dptr->factor, ehints))
                    {
                        gLogger.error() << task->path.input <<": Failed to open encoder";
                        emit task->finished(false);
                        continue;
                    }

                    auto info = pipeline.getInfo();
                    struct {
                        std::atomic_bool& stopFlag;
                        int shift;
                        double factor;
                        double frames;
                        Upscaler* upscaler;
                        std::shared_ptr<ac::core::Processor> processor;
                    } data {
                        dptr->stopFlag,
                        info.bitDepth.lsb ? ((info.bitDepth.bits - 1) / 8 + 1) * 8 - info.bitDepth.bits : 0, // bytes * 8 - bits
                        dptr->factor,
                        info.fps * info.duration,
                        this,
                        dptr->processor
                    };
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
                        emit ctx->upscaler->progress(static_cast<int>(100 * src.number / ctx->frames));
                        return !ctx->stopFlag;
                    }, &data, ac::video::FILTER_AUTO);
                    stopwatch.stop();
                    pipeline.close();
                    if (!dptr->processor->ok()) gLogger.error() << task->path.input << ": Failed due to " << dptr->processor->error();
                    else gLogger.info() << task->path.input <<": Finished in " << stopwatch.elapsed() << "s [" << gConfig.upscaler.processor << ' ' << dptr->processor->name() << ']';
                    gLogger.info() << "Save video to " << task->path.output;
                }
                emit progress(100);
                emit task->finished(!dptr->stopFlag && dptr->processor->ok());
            }
        });
#   else
        for (auto&& task : videoTaskList)
        {
            gLogger.warning() << "This build does not support video processing";
            emit task->finished(false);
            if (--dptr->total == 0) emit stopped();
        }
#   endif

    for (auto&& task : imageTaskList)
    {
        pool.exec([=](){
            ac::util::Defer defer([&]() { if (--dptr->total == 0) emit stopped(); });
            if (!dptr->stopFlag)
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
                gLogger.info() << task->path.input <<": Finished in " << stopwatch.elapsed() << "s [" << gConfig.upscaler.processor << ' ' << dptr->processor->name() << ']';

                if (ac::core::imwrite(task->path.output.toLocal8Bit(), dst)) gLogger.info() << "Save image to " << task->path.output;
                else
                {
                    gLogger.error() << "Failed to save image to " << task->path.output;
                    emit task->finished(false);
                    return;
                }
            }
            emit task->finished(!dptr->stopFlag);
        });
    }
}
void Upscaler::stop()
{
    dptr->stopFlag = true;
}
