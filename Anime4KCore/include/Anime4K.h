#pragma once
#include<iostream>
#include<sstream>
#include<functional>

#include<opencv2/opencv.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/core/hal/interface.h>

#include"threadpool.h"
#include"filterprocessor.h"

#ifdef _MSC_VER
#include<ppl.h>
#else
#include<omp.h>
#endif

#ifdef _MSC_VER
#ifndef DLL
#define DLL __declspec(dllimport)
#else
#undef DLL
#define DLL __declspec(dllexport)
#endif
#else
#ifndef DLL
#define DLL
#endif
#endif

#define ANIME4KCPP_CORE_VERSION "1.7.0"

#define MAX3(a, b, c) std::max({a, b, c})
#define MIN3(a, b, c) std::min({a, b, c})
#define UNFLOAT(n) ((n) >= 255 ? 255 : ((n) <= 0 ? 0 : uint8_t((n) + 0.5)))

typedef unsigned char* RGBA;
typedef unsigned char* Line;

enum BGRA
{
    B = 0, G = 1, R = 2, A = 3
};

enum CODEC {
    OTHER = -1, MP4V = 0, DXVA = 1, AVC1 = 2, VP09 = 3, HEVC = 4, AV01 = 5
};

class DLL Anime4K
{
public:
    Anime4K(
        int passes = 2,
        int pushColorCount = 2,
        double strengthColor = 0.3,
        double strengthGradient = 1.0,
        double zoomFactor = 2.0,
        bool fastMode = false,
        bool videoMode = false,
        bool PreProcessing = false,
        bool postProcessing = false,
        uint8_t preFilters = 4,
        uint8_t postFilters = 40,
        unsigned int maxThreads = std::thread::hardware_concurrency()
    );
    virtual ~Anime4K();
    void setArguments(
        int passes = 2,
        int pushColorCount = 2,
        double strengthColor = 0.3,
        double strengthGradient = 1.0,
        double zoomFactor = 2.0,
        bool fastMode = false,
        bool videoMode = false,
        bool PreProcessing = false,
        bool postProcessing = false,
        uint8_t preFilters = 40,
        uint8_t postFilters = 40,
        unsigned int maxThreads = std::thread::hardware_concurrency()
    );
    void setVideoMode(const bool flag);
    void loadVideo(const std::string& srcFile);
    void loadImage(const std::string& srcFile);
    void setVideoSaveInfo(const std::string& dstFile,const CODEC codec = MP4V);
    void saveImage(const std::string& dstFile);
    void saveVideo();
    void showInfo();
    void showFiltersInfo();
    std::string getInfo();
    std::string getFiltersInfo();
    void showImage();
    virtual void process();
protected:
    void getGray(cv::InputArray img);
    void pushColor(cv::InputArray img);
    void getGradient(cv::InputArray img);
    void pushGradient(cv::InputArray img);
private:
    void changEachPixelBGRA(cv::InputArray _src, const std::function<void(int, int, RGBA, Line)>&& callBack);
    void getLightest(RGBA mc, RGBA a, RGBA b, RGBA c);
    void getAverage(RGBA mc, RGBA a, RGBA b, RGBA c);
protected:
    int orgH, orgW, H, W;
    double fps;
    uint64_t totalFrameCount, frameCount;
    cv::Mat orgImg, dstImg;
    cv::VideoCapture video;
    cv::VideoWriter videoWriter;
    std::mutex videoMtx;
    std::condition_variable cnd;
protected://arguments
    int ps, pcc;
    double sc, sg, zf;
    bool fm, vm, pre, post;
    uint8_t pref, postf;
    unsigned int mt;
};
