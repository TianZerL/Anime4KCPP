#pragma once

#ifdef ENABLE_VIDEO

#include<atomic>
#include<future>
#include<mutex>
#include<condition_variable>
#include<functional>
#include<memory>
#include<queue>
#include<unordered_map>
#include<cstddef>

#include<opencv2/opencv.hpp>

#include"VideoCodec.hpp"

#if (CV_VERSION_MAJOR < 3)
#error OpenCV3 is needed at least!
#endif
#if (CV_VERSION_MAJOR == 3) && (CV_VERSION_MINOR <= 2)
#define OLD_OPENCV_API
#endif

namespace Anime4KCPP::Utils
{
    class VideoIO;

    using Frame = std::pair<cv::Mat, std::size_t>;
}

class Anime4KCPP::Utils::VideoIO
{
public:
    VideoIO() = default;
    ~VideoIO();
    VideoIO(const VideoIO&) = delete;
    VideoIO& operator=(const VideoIO&) = delete;
    //initialize frame process callback function `p` and thread count `t`, it's ready to call process after this
    VideoIO& init(std::function<void()>&& p, std::size_t t) noexcept;
    void process();
    //initialize VideoCapture
    bool openReader(const std::string& srcFile);
    //initialize VideoWriter
    bool openWriter(const std::string& dstFile, Codec codec, const cv::Size& size, double forceFps = 0.0);
    //get the specifying video property from VideoCapture
    double get(int p);
    void release();
    Frame read();
    void write(const Frame& frame);
    double getProgress() noexcept;
    void stopProcess() noexcept;
    void pauseProcess();
    void continueProcess();
    bool isPaused() noexcept;
private:
    void setProgress(double p) noexcept;
private:
    std::size_t threads = 0;
    std::function<void()> processor;
    cv::VideoCapture reader;
    cv::VideoWriter writer;
    std::queue <Frame> rawFrames;
    std::unordered_map<std::size_t, cv::Mat> frameMap;
    //lock
    std::mutex mtxRead;
    std::condition_variable cndRead;
    std::mutex mtxWrite;
    std::condition_variable cndWrite;
    //callback data
    std::atomic<double> progress;
    std::atomic<std::size_t> stop;
    std::unique_ptr<std::promise<void>> pausePromise;
    bool pause{ false };
};

#endif
