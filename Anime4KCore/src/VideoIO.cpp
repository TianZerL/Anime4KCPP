#ifdef ENABLE_VIDEO

#include"VideoIO.hpp"

Anime4KCPP::Utils::VideoIO::~VideoIO()
{
    writer.release();
    reader.release();
}

Anime4KCPP::Utils::VideoIO& Anime4KCPP::Utils::VideoIO::init(std::function<void()>&& p, size_t t) noexcept
{
    processor = std::move(p);
    threads = t;
    return *this;
}

void Anime4KCPP::Utils::VideoIO::process()
{
    ThreadPool pool(threads + 1);
    stop = static_cast<size_t>(reader.get(cv::CAP_PROP_FRAME_COUNT));

    pool.exec([this]()
        {
            for (size_t i = 0; i < stop; i++)
            {
                std::unique_lock<std::mutex> lock(mtxWrite);
                std::unordered_map<size_t, cv::Mat>::iterator it;
                for (;;)
                {
                    it = frameMap.find(i);
                    if (it == frameMap.end())
                        cndWrite.wait(lock);
                    else
                        break;
                }
                writer.write(it->second);
                frameMap.erase(it);
                setProgress(static_cast<double>(i + 1) / static_cast<double>(stop));
            }
        });

    for (size_t i = 0; i < stop; i++)
    {
        cv::Mat frame;
        if (!reader.read(frame))
        {
            stop = i;
            break;
        }
        {
            std::unique_lock<std::mutex> lock(mtxRead);
            while (rawFrames.size() >= threads)
                cndRead.wait(lock);
            rawFrames.emplace(std::pair<cv::Mat, size_t>(frame, i));
        }
        pool.exec(processor);
    }
}

bool Anime4KCPP::Utils::VideoIO::openReader(const std::string& srcFile)
{
    if (!reader.open(srcFile, cv::CAP_FFMPEG))
        return reader.open(srcFile);
    return reader.isOpened();
}

bool Anime4KCPP::Utils::VideoIO::openWriter(const std::string& dstFile, const CODEC codec, const cv::Size& size, const double forceFps)
{
    double fps;

    if (forceFps < 1.0)
        fps = reader.get(cv::CAP_PROP_FPS);
    else
        fps = forceFps;

    switch (codec)
    {
    case CODEC::MP4V:
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
    case CODEC::DXVA:
        writer.open(dstFile, cv::CAP_MSMF, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), std::ceil(fps), size);
        if (!writer.isOpened())
        {
            writer.open(dstFile, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, size);
            if (!writer.isOpened())
                return false;
        }
        break;
#endif

    case CODEC::AVC1:
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

    case CODEC::VP09:
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

    case CODEC::HEVC:
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

    case CODEC::AV01:
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

    case CODEC::OTHER:
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
    writer.release();
    reader.release();
}

Anime4KCPP::Utils::Frame Anime4KCPP::Utils::VideoIO::read()
{
    Frame ret;
    {
        std::lock_guard<std::mutex> lock(mtxRead);
        ret = std::move(rawFrames.front());
        rawFrames.pop();
    }
    cndRead.notify_all();
    return ret;
}

void Anime4KCPP::Utils::VideoIO::write(const Frame& frame)
{
    {
        std::lock_guard<std::mutex> lock(mtxWrite);
        frameMap.emplace(frame.second, frame.first);
    }
    cndWrite.notify_all();
}

double Anime4KCPP::Utils::VideoIO::getProgress() noexcept
{
    return progress;
}

void Anime4KCPP::Utils::VideoIO::stopProcess() noexcept
{
    stop = 1;
}

void Anime4KCPP::Utils::VideoIO::pauseProcess()
{
    pause = true;
    {
        std::lock_guard<std::mutex> lock(mtxRead);
        while (pause)
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

void Anime4KCPP::Utils::VideoIO::continueProcess() noexcept
{
    pause = false;
}

bool Anime4KCPP::Utils::VideoIO::isPaused() noexcept
{
    return pause;
}

inline void Anime4KCPP::Utils::VideoIO::setProgress(double p) noexcept
{
    progress = p;
}

#endif
