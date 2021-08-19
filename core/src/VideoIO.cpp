#ifdef ENABLE_VIDEO

#include"VideoIO.hpp"
#include"ThreadPool.hpp"

Anime4KCPP::Utils::VideoIO::~VideoIO()
{
    stopProcess();
    release();
}

Anime4KCPP::Utils::VideoIO& Anime4KCPP::Utils::VideoIO::init(std::function<void()>&& p, std::size_t t) noexcept
{
    processor = std::move(p);
    threads = t;
    return *this;
}

void Anime4KCPP::Utils::VideoIO::process()
{
    std::promise<void> barrier;
    std::future<void> barrierFuture = barrier.get_future();

    ThreadPool pool(threads + 1);

    stop = false;

    finished = 0;

    pool.exec([this, &barrier]()
        {
            double totalFrame = reader.get(cv::CAP_PROP_FRAME_COUNT);

            for (std::size_t frameCount = 0; finished == 0 || frameCount < finished; frameCount++)
            {
                std::unique_lock<std::mutex> lock(mtxWrite);
                std::unordered_map<std::size_t, cv::Mat>::iterator it;

                while (!stop && ((it = frameMap.find(frameCount)) == frameMap.end()))
                    cndWrite.wait(lock);

                if (stop)
                    return barrier.set_value();

                writer.write(it->second);
                frameMap.erase(it);
                setProgress(static_cast<double>(frameCount) / totalFrame);
            }

            barrier.set_value();
        });


    for (std::size_t frameCount = 0;; frameCount++)
    {
        cv::Mat frame;
        if (!reader.read(frame))
        {
            finished = frameCount;
            break;
        }
        {
            std::unique_lock<std::mutex> lock(mtxRead);

            while (!stop && rawFrames.size() >= threads)
                cndRead.wait(lock);

            if (stop)
                break;

            rawFrames.emplace(frame, frameCount);
        }
        pool.exec(processor);
    }

    barrierFuture.wait();
}

bool Anime4KCPP::Utils::VideoIO::openReader(const std::string& srcFile)
{
    return reader.open(srcFile);
}

bool Anime4KCPP::Utils::VideoIO::openWriter(const std::string& dstFile, const Codec codec, const cv::Size& size, const double forceFps)
{
    double fps;

    if (forceFps < 1.0)
        fps = reader.get(cv::CAP_PROP_FPS);
    else
        fps = forceFps;

    switch (codec)
    {
    case Codec::MP4V:
#ifndef OLD_OPENCV_API
        writer.open(dstFile, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, size);
        if (!writer.isOpened())
#endif // !OLD_OPENCV_API
        {
            writer.open(dstFile, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, size);
            if (!writer.isOpened())
                return false;
        }
        break;

#if defined(_WIN32) && !defined(OLD_OPENCV_API) //DXVA encoding for windows
    case Codec::DXVA:
        writer.open(dstFile, cv::CAP_MSMF, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), std::ceil(fps), size);
        if (!writer.isOpened())
        {
            writer.open(dstFile, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, size);
            if (!writer.isOpened())
                return false;
        }
        break;
#endif

    case Codec::AVC1:
#ifndef OLD_OPENCV_API
        writer.open(dstFile, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, size);
        if (!writer.isOpened())
#endif // !OLD_OPENCV_API
        {
            writer.open(dstFile, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, size);
            if (!writer.isOpened())
                return false;
        }
        break;

    case Codec::VP09:
#ifndef OLD_OPENCV_API
        writer.open(dstFile, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('v', 'p', '0', '9'), fps, size);
        if (!writer.isOpened())
#endif // !OLD_OPENCV_API
        {
            writer.open(dstFile, cv::VideoWriter::fourcc('v', 'p', '0', '9'), fps, size);
            if (!writer.isOpened())
                return false;
        }
        break;

    case Codec::HEVC:
#ifndef OLD_OPENCV_API
        writer.open(dstFile, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('h', 'e', 'v', '1'), fps, size);
        if (!writer.isOpened())
#endif // !OLD_OPENCV_API
        {
            writer.open(dstFile, cv::VideoWriter::fourcc('h', 'e', 'v', '1'), fps, size);
            if (!writer.isOpened())
                return false;
        }
        break;

    case Codec::AV01:
#ifndef OLD_OPENCV_API
        writer.open(dstFile, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('a', 'v', '0', '1'), fps, size);
        if (!writer.isOpened())
#endif // !OLD_OPENCV_API
        {
            writer.open(dstFile, cv::VideoWriter::fourcc('a', 'v', '0', '1'), fps, size);
            if (!writer.isOpened())
                return false;
        }
        break;

    case Codec::OTHER:
        writer.open(dstFile, -1, fps, size);
        if (!writer.isOpened())
            return false;
        break;
    default:
        writer.open(dstFile, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, size);
        if (!writer.isOpened())
            return false;
    }
    return true;
}

double Anime4KCPP::Utils::VideoIO::get(int p)
{
    return reader.get(p);
}

void Anime4KCPP::Utils::VideoIO::release()
{
    reader.release();
    writer.release();

    if (!rawFrames.empty())
        std::queue<Frame>().swap(rawFrames);
    if (!frameMap.empty())
        frameMap.clear();
}

Anime4KCPP::Utils::Frame Anime4KCPP::Utils::VideoIO::read()
{
    Frame ret;
    {
        const std::lock_guard<std::mutex> lock(mtxRead);
        ret = std::move(rawFrames.front());
        rawFrames.pop();
    }
    cndRead.notify_one();
    return ret;
}

void Anime4KCPP::Utils::VideoIO::write(const Frame& frame)
{
    {
        const std::lock_guard<std::mutex> lock(mtxWrite);
        frameMap.emplace(frame.second, frame.first);
    }
    cndWrite.notify_one();
}

double Anime4KCPP::Utils::VideoIO::getProgress() noexcept
{
    return progress;
}

void Anime4KCPP::Utils::VideoIO::stopProcess() noexcept
{
    {
        std::scoped_lock lock(mtxRead, mtxWrite);
        stop = true;
    }
    cndRead.notify_one();
    cndWrite.notify_one();
}

void Anime4KCPP::Utils::VideoIO::pauseProcess()
{
    if (!pause)
    {
        pausePromise = std::make_unique<std::promise<void>>();
        std::thread t([this, f = pausePromise->get_future()]()
        {
            const std::lock_guard<std::mutex> lock(mtxRead);
            f.wait();
        });
        t.detach();
        pause = true;
    }
}

void Anime4KCPP::Utils::VideoIO::continueProcess()
{
    if (pause)
    {
        pausePromise->set_value();
        pause = false;
    }
}

bool Anime4KCPP::Utils::VideoIO::isPaused() noexcept
{
    return pause;
}

void Anime4KCPP::Utils::VideoIO::setProgress(double p) noexcept
{
    progress = p;
}

#endif
