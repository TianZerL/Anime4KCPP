#include "VideoIO.h"

Anime4KCPP::VideoIO::~VideoIO()
{
    writer.release();
    reader.release();
}

Anime4KCPP::VideoIO& Anime4KCPP::VideoIO::instance()
{
    static VideoIO videoIOInstance;
    return videoIOInstance;
}

Anime4KCPP::VideoIO& Anime4KCPP::VideoIO::init(std::function<void()>&& p, size_t t)
{
    processor = std::move(p);
    threads = t;
    return *this;
}

void Anime4KCPP::VideoIO::process()
{
    ThreadPool pool(threads + 1);
    stop = reader.get(cv::CAP_PROP_FRAME_COUNT);

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
        if(!reader.read(frame))
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

bool Anime4KCPP::VideoIO::openReader(const std::string& srcFile)
{
    if (!reader.open(srcFile, cv::CAP_FFMPEG))
        return reader.open(srcFile);
    return reader.isOpened();
}

bool Anime4KCPP::VideoIO::openWriter(const std::string& dstFile,const CODEC codec, const cv::Size& size,const double forceFps)
{
    double fps;
    if (!forceFps)
        fps = reader.get(cv::CAP_PROP_FPS);
    else
        fps = forceFps;
    switch (codec)
    {
    case CODEC::MP4V:
        writer.open(dstFile, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, size);
        if (!writer.isOpened())
        {
            writer.open(dstFile, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, size);
            if (!writer.isOpened())
                return false;
        }
        break;
#ifdef _WIN32 //DXVA encoding for windows
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
        writer.open(dstFile, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, size);
        if (!writer.isOpened())
        {
            writer.open(dstFile, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, size);
            if (!writer.isOpened())
                return false;
        }
        break;
    case CODEC::VP09:
        writer.open(dstFile, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('v', 'p', '0', '9'), fps, size);
        if (!writer.isOpened())
        {
            writer.open(dstFile, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, size);
            if (!writer.isOpened())
                return false;
        }
        break;
    case CODEC::HEVC:
        writer.open(dstFile, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('h', 'e', 'v', '1'), fps, size);
        if (!writer.isOpened())
        {
            writer.open(dstFile, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, size);
            if (!writer.isOpened())
                return false;
        }
        break;
    case CODEC::AV01:
        writer.open(dstFile, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('a', 'v', '0', '1'), fps, size);
        if (!writer.isOpened())
        {
            writer.open(dstFile, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, size);
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

double Anime4KCPP::VideoIO::get(int p)
{
    return reader.get(p);
}

void Anime4KCPP::VideoIO::release()
{
    writer.release();
    reader.release();
}

Anime4KCPP::Frame Anime4KCPP::VideoIO::read()
{
    Frame ret;
    {
        std::lock_guard<std::mutex> lock(mtxRead);
        ret = std::move(rawFrames.front());
        rawFrames.pop();
    }
    cndRead.notify_all();
    return std::move(ret);
}

void Anime4KCPP::VideoIO::write(const Frame& frame)
{
    {
        std::lock_guard<std::mutex> lock(mtxWrite);
        frameMap[frame.second] = frame.first;
    }
    cndWrite.notify_all();
}

double Anime4KCPP::VideoIO::getProgress()
{
    return progress;
}

void Anime4KCPP::VideoIO::stopProcess()
{
    stop = 1;
}

void Anime4KCPP::VideoIO::pauseProcess()
{
    pause = true;
    {
        std::lock_guard<std::mutex> lock(mtxRead);
        while (pause)
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

void Anime4KCPP::VideoIO::continueProcess()
{
    pause = false;
}

inline void Anime4KCPP::VideoIO::setProgress(double p)
{
    progress = p;
}
