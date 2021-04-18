#pragma once

#ifdef ENABLE_VIDEO

#include<opencv2/opencv.hpp>
#include<atomic>
#include<queue>
#include<unordered_map>

#include"ThreadPool.hpp"
#include"VideoCodec.hpp"

#if (CV_VERSION_MAJOR < 3)
#error OpenCV3 is needed at least!
#endif
#if (CV_VERSION_MAJOR == 3) && (CV_VERSION_MINOR <= 2)
#define OLD_OPENCV_API
#endif

namespace Anime4KCPP
{
    namespace Utils
    {
        class VideoIO;
        typedef std::pair<cv::Mat, size_t> Frame;
    }
}

class Anime4KCPP::Utils::VideoIO
{
public:
    VideoIO() = default;
    ~VideoIO();
    VideoIO(const VideoIO&) = delete;
    VideoIO& operator=(const VideoIO&) = delete;
    //initialize frame process callback function `p` and thread count `t`, it's ready to call process after this
    VideoIO& init(std::function<void()> &&p, size_t t) noexcept;
    void process();
    //initialize VideoCapture
    bool openReader(const std::string& srcFile);
    //initialize VideoWriter
    bool openWriter(const std::string& dstFile, const CODEC codec, const cv::Size& size,const double forceFps = 0.0);
    //get the specifying video property from VideoCapture
    double get(int p);
    void release();
    Frame read();
    void write(const Frame& frame);
    double getProgress() noexcept;
    void stopProcess() noexcept;
    void pauseProcess();
    void continueProcess() noexcept;
    bool isPaused() noexcept;
private:
    void setProgress(double p) noexcept;
private:
    size_t threads = 0;
    std::function<void()> processor;
    cv::VideoCapture reader;
    cv::VideoWriter writer;
    std::queue <Frame> rawFrames;
    std::unordered_map<size_t, cv::Mat> frameMap;
    //lock
    std::mutex mtxRead;
    std::condition_variable cndRead;
    std::mutex mtxWrite;
    std::condition_variable cndWrite;
    //callback data
    std::atomic<double> progress;
    std::atomic<size_t> stop;
    std::atomic<bool> pause{ false };
};

#endif
