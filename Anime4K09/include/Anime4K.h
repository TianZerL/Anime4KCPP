#pragma once
#include<iostream>
#include<functional>

#include<opencv2/opencv.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/core/hal/interface.h>

#include"threadpool.h"
#include"postprocessing.h"

#define MAX3(a, b, c) (a > b && a > c ? a : (b > c ? b : c))
#define MIN3(a, b, c) (a < b && a < c ? a : (b < c ? b : c))
#define UNFLOAT(n) (n >= 255 ? 255 : (n <= 0 ? 0 : uint8_t(n + 0.5)))

typedef unsigned char* RGBA;
typedef unsigned char* Line;


class Anime4K
{
public:
    Anime4K(
        int passes = 2, 
        double strengthColor = 0.3, 
        double strengthGradient = 1.0, 
        double zoomFactor = 2.0, 
        bool fastMode = false, 
        bool videoMode = false, 
        bool postProcessing = false,
        uint8_t filters = 12,
        unsigned int maxThreads = std::thread::hardware_concurrency());
    void loadVideo(const std::string &srcFile);
    void loadImage(const std::string &srcFile);
    void setVideoSaveInfo(const std::string &dstFile);
    void saveImage(const std::string &dstFile);
    void saveVideo();
    void showInfo();
    void showFiltersInfo();
    void showImage();
    void process();
private:
    void getGray(cv::InputArray img);
    void pushColor(cv::InputArray img);
    void getGradient(cv::InputArray img);
    void pushGradient(cv::InputArray img);
    void changEachPixel(cv::InputArray _src, const std::function<void(int, int, RGBA,Line)>&& callBack);
    void getLightest(RGBA mc, RGBA a, RGBA b, RGBA c);
    void getAverage(RGBA mc, RGBA a, RGBA b, RGBA c);
private:
    const static int B = 0, G = 1, R = 2, A = 3;
    int orgH, orgW, H, W;
    size_t totalFrameCount, frameCount;
    cv::Mat orgImg, dstImg;
    cv::VideoCapture video;
    cv::VideoWriter videoWriter;
    std::mutex videoMtx;
    std::condition_variable cnd;
private://arguments
    unsigned int mt;
    int ps;
    double sc, sg, zf, fps;
    bool fm, vm, pp;
    uint8_t fl;
};
