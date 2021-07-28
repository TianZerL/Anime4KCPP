#pragma once

#ifdef ENABLE_VIDEO

#include"AC.hpp"
#include"VideoIO.hpp"

namespace Anime4KCPP
{
    class DLL VideoProcessor;
}

class Anime4KCPP::VideoProcessor
{
public:
    VideoProcessor(const Parameters& parameters, Processor::Type type);
    explicit VideoProcessor(AC& config);
    void setVideoSaveInfo(const std::string& dstFile, CODEC codec = CODEC::MP4V, double fps = 0.0);
    void loadVideo(const std::string& srcFile);
    void saveVideo();

    void process();
    void processWithPrintProgress();
    void processWithProgress(const std::function<void(double)>&& callBack);
    void stopVideoProcess() noexcept;
    void pauseVideoProcess();
    void continueVideoProcess() noexcept;

    std::string getInfo();
private:
    double fps;
    double totalFrameCount;
    int H, W;

    Utils::VideoIO videoIO;
    Parameters param;
    Processor::Type type;
};

#endif // ENABLE_VIDEO
