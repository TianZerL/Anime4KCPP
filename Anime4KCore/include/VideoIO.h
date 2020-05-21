#pragma once

#include<opencv2/opencv.hpp>
#include<queue>

#include"threadpool.h"

namespace Anime4KCPP
{
    class VideoIO;
    enum class CODEC;
    typedef std::pair<cv::Mat, size_t> Frame;
}

enum class Anime4KCPP::CODEC 
{
    OTHER = -1, MP4V = 0, DXVA = 1, AVC1 = 2, VP09 = 3, HEVC = 4, AV01 = 5
};

class Anime4KCPP::VideoIO
{
public:
    ~VideoIO();
    VideoIO(const VideoIO&) = delete;
    VideoIO& operator=(const VideoIO&) = delete;
    static VideoIO& instance();
    VideoIO& init(std::function<void()> &&p, size_t t);
    void process();
    bool openReader(const std::string& srcFile);
    bool openWriter(const std::string& dstFile, CODEC codec, const cv::Size& size);
    double get(int p);
    void release();
    Frame read();
    void write(const Frame& frame);
private:
    VideoIO() = default;
private:
    size_t threads=0;
    std::function<void()> processor;
    cv::VideoCapture reader;
    cv::VideoWriter writer;
    std::queue <Frame> rawFrames;
    std::unordered_map<size_t, cv::Mat> frameMap;

    std::mutex mtxRead;
    std::condition_variable cndRead;
    std::mutex mtxWrite;
    std::condition_variable cndWrite;
};

