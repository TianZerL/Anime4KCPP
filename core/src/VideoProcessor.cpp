#ifdef ENABLE_VIDEO

#include"ACCreator.hpp"
#include"VideoProcessor.hpp"

Anime4KCPP::VideoProcessor::VideoProcessor(const Parameters& parameters, const Processor::Type type)
    :param(parameters), type(type)
{
    totalFrameCount = fps = 0.0;
    width = height = 0;
}

Anime4KCPP::VideoProcessor::VideoProcessor(AC& config)
    :VideoProcessor(config.getParameters(), config.getProcessorType()) {}

void Anime4KCPP::VideoProcessor::loadVideo(const std::string& srcFile)
{
    if (!videoIO.openReader(srcFile))
        throw ACException<ExceptionType::IO>("Failed to load file: file doesn't not exist or decoder isn't installed.");

    fps = videoIO.get(cv::CAP_PROP_FPS);
    totalFrameCount = videoIO.get(cv::CAP_PROP_FRAME_COUNT);
    height = std::round(param.zoomFactor * videoIO.get(cv::CAP_PROP_FRAME_HEIGHT));
    width = std::round(param.zoomFactor * videoIO.get(cv::CAP_PROP_FRAME_WIDTH));
}

void Anime4KCPP::VideoProcessor::setVideoSaveInfo(const std::string& dstFile, const CODEC codec, const double fps)
{
    if (!videoIO.openWriter(dstFile, codec, cv::Size(width, height), fps))
        throw ACException<ExceptionType::IO>("Failed to initialize video writer.");
}

void Anime4KCPP::VideoProcessor::saveVideo()
{
    videoIO.release();
}

void Anime4KCPP::VideoProcessor::process()
{
    videoIO.init(
        [this]()
        {
            Utils::Frame frame = videoIO.read();
            { // Reduce memory usage
                auto ac = ACCreator::createUP(param, type);
                ac->loadImage(frame.first);
                ac->process();
                ac->saveImage(frame.first);
            }
            videoIO.write(frame);
        }, param.maxThreads
    ).process();
}

void Anime4KCPP::VideoProcessor::processWithPrintProgress()
{
    auto s = std::chrono::steady_clock::now();
    processWithProgress([&s](double progress)
        {
            auto e = std::chrono::steady_clock::now();
            double currTime = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0;

            std::cout
                << std::fixed << std::setprecision(2)
                << std::setw(7) << progress * 100 << '%'
                << "    elpsed: " << std::setw(10) << currTime << 's'
                << "    remaining: " << std::setw(10) << currTime / progress - currTime << 's'
                << '\r';

            if (progress == 1.0)
                std::cout << std::endl;
        });
}

void Anime4KCPP::VideoProcessor::processWithProgress(const std::function<void(double)>&& callBack)
{
    std::future<void> p = std::async(std::launch::async, &VideoProcessor::process, this);
    std::chrono::milliseconds timeout(1000);
    for (;;)
    {
        std::future_status status = p.wait_for(timeout);
        if (status == std::future_status::ready)
        {
            callBack(1.0);
            p.get();
            break;
        }
        double progress = videoIO.getProgress();
        callBack(progress);
    }
}

void Anime4KCPP::VideoProcessor::stopVideoProcess() noexcept
{
    videoIO.stopProcess();
}

void Anime4KCPP::VideoProcessor::pauseVideoProcess()
{
    videoIO.pauseProcess();
}

void Anime4KCPP::VideoProcessor::continueVideoProcess()
{
    videoIO.continueProcess();
}

std::string Anime4KCPP::VideoProcessor::getInfo()
{
    std::ostringstream oss;
    oss << "----------------------------------------------" << std::endl
        << "Video information" << std::endl
        << "----------------------------------------------" << std::endl
        << "FPS: " << fps << std::endl
        << "Threads: " << param.maxThreads << std::endl
        << "Total frames: " << totalFrameCount << std::endl
        << "----------------------------------------------" << std::endl;
    return oss.str();
}

#endif // ENABLE_VIDEO
