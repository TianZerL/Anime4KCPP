#ifdef ENABLE_VIDEO

#define DLL

#include"ACCreator.hpp"
#include"VideoProcessor.hpp"

Anime4KCPP::VideoProcessor::VideoProcessor(const Parameters& parameters, const Processor::Type type)
    :param(parameters), type(type)
{
    H = W = 0;
    totalFrameCount = fps = 0.0;
};

Anime4KCPP::VideoProcessor::VideoProcessor(AC& config)
    :param(config.getParameters()), type(config.getProcessorType())
{
    H = W = 0;
    totalFrameCount = fps = 0.0;
};

void Anime4KCPP::VideoProcessor::loadVideo(const std::string& srcFile)
{
    if (!videoIO.openReader(srcFile))
        throw ACException<ExceptionType::IO>("Failed to load file: file doesn't not exist or decoder isn't installed.");
    double orgH = videoIO.get(cv::CAP_PROP_FRAME_HEIGHT);
    double orgW = videoIO.get(cv::CAP_PROP_FRAME_WIDTH);
    fps = videoIO.get(cv::CAP_PROP_FPS);
    totalFrameCount = videoIO.get(cv::CAP_PROP_FRAME_COUNT);
    H = std::round(param.zoomFactor * orgH);
    W = std::round(param.zoomFactor * orgW);
}

void Anime4KCPP::VideoProcessor::setVideoSaveInfo(const std::string& dstFile, const CODEC codec, const double fps)
{
    if (!videoIO.openWriter(dstFile, codec, cv::Size(W, H), fps))
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
            cv::Mat orgFrame = frame.first;
            cv::Mat dstFrame;

            auto ac = ACCreator::createUP(param, type);
            ac->loadImage(orgFrame);
            ac->process();
            ac->saveImage(dstFrame);

            frame.first = dstFrame;
            videoIO.write(frame);
        }, param.maxThreads
    ).process();
}

void Anime4KCPP::VideoProcessor::processWithPrintProgress()
{
    std::future<void> p = std::async(&VideoProcessor::process, this);
    std::chrono::milliseconds timeout(1000);
    std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();
    for (;;)
    {
        std::future_status status = p.wait_for(timeout);
        if (status == std::future_status::ready)
        {
            std::cout
                << std::fixed << std::setprecision(2)
                << std::setw(7) << 100.0 << '%'
                << "    elpsed: " << std::setw(10) << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - s).count() / 1000.0 << 's'
                << "    remaining: " << std::setw(10) << 0.0 << 's'
                << std::endl;
            // get any possible exception
            p.get();
            break;
        }
        std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
        double currTime = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0;
        double progress = videoIO.getProgress();
        std::cout
            << std::fixed << std::setprecision(2)
            << std::setw(7) << progress * 100 << '%'
            << "    elpsed: " << std::setw(10) << currTime << 's'
            << "    remaining: " << std::setw(10) << currTime / progress - currTime << 's'
            << '\r';
    }
}

void Anime4KCPP::VideoProcessor::processWithProgress(const std::function<void(double)>&& callBack)
{
    std::future<void> p = std::async(&VideoProcessor::process, this);
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
    if (!videoIO.isPaused())
    {
        std::thread t(&Utils::VideoIO::pauseProcess, &videoIO);
        t.detach();
    }
}

void Anime4KCPP::VideoProcessor::continueVideoProcess() noexcept
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
