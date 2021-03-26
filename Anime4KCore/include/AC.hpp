#pragma once

#include<iostream>
#include<functional>
#include<atomic>
#include<future>
#include<cmath>

#include<opencv2/opencv.hpp>

#include"ACException.hpp"
#include"VideoIO.hpp"
#include"ACProcessor.hpp"

#if defined(_MSC_VER) && !defined(USE_TBB)
#include<ppl.h>
namespace Parallel = Concurrency;
#define PARALLEL_LIBRARY "PPL"
#elif defined(USE_TBB)
#include<tbb/parallel_for.h>
namespace Parallel = tbb;
#define PARALLEL_LIBRARY "TBB"
#else
#include<omp.h>
#define PARALLEL_LIBRARY "OpenMP"
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

namespace Anime4KCPP
{
    struct DLL Parameters;
    class DLL AC;

    enum BGRA
    {
        B = 0, G = 1, R = 2, A = 3
    };

    enum YUV
    {
        Y = 0, U = 1, V = 2
    };

    enum FilterType : uint8_t
    {
        MEDIAN_BLUR = 1, MEAN_BLUR = 2, CAS_SHARPENING = 4,
        GAUSSIAN_BLUR_WEAK = 8, GAUSSIAN_BLUR = 16,
        BILATERAL_FILTER = 32, BILATERAL_FILTER_FAST = 64
    };

    typedef float* ChanF;
    typedef double* ChanD;
    typedef unsigned char* ChanB;
    typedef unsigned short int* ChanW;

    typedef float* PixelF;
    typedef double* PixelD;
    typedef unsigned char* PixelB;
    typedef unsigned short int* PixelW;

    typedef float* LineF;
    typedef double* LineD;
    typedef unsigned char* LineB;
    typedef unsigned short int* LineW;

    namespace Utils
    {
        int ceilLog2(double v);
    }
}

// compute log2(v) then do ceil(v)
inline int Anime4KCPP::Utils::ceilLog2(double v)
{
    long long int data = *reinterpret_cast<long long int*>(&v);
    return static_cast<int>((((data >> 52) & 0x7ff) - 1023) + ((data << 12) || 0));
}

struct Anime4KCPP::Parameters
{
    int passes;
    int pushColorCount;
    double strengthColor;
    double strengthGradient;
    double zoomFactor;
    bool fastMode;
    bool videoMode;
    bool preprocessing;
    bool postprocessing;
    uint8_t preFilters;
    uint8_t postFilters;
    unsigned int maxThreads;
    bool HDN;
    bool alpha;
    int HDNLevel;

    // return true if zoomFactor is not a power of 2
    inline bool isNonIntegerScale() noexcept
    {
        return ((*reinterpret_cast<long long int*>(&zoomFactor)) << 12) || (zoomFactor < 2.0);
    }

    void reset() noexcept;

    Parameters(
        int passes = 2,
        int pushColorCount = 2,
        double strengthColor = 0.3,
        double strengthGradient = 1.0,
        double zoomFactor = 2.0,
        bool fastMode = false,
        bool videoMode = false,
        bool preprocessing = false,
        bool postprocessing = false,
        uint8_t preFilters = 4,
        uint8_t postFilters = 40,
        unsigned int maxThreads = std::thread::hardware_concurrency(),
        bool HDN = false,
        int HDNLevel = 1,
        bool alpha = false
    ) noexcept;
};

//Base class for IO operation
class Anime4KCPP::AC
{
public:
    AC(const Parameters& parameters);
    virtual ~AC();

    virtual void setArguments(const Parameters& parameters);
    void setVideoMode(const bool value);

    void loadVideo(const std::string& srcFile);
    void loadImage(const std::string& srcFile);
    void loadImage(const cv::Mat& srcImage);
    void loadImage(int rows, int cols, size_t stride, unsigned char* data, bool inputAsYUV444 = false, bool inputAsRGB32 = false, bool inputAsGrayscale = false);
    void loadImage(int rows, int cols, size_t stride, unsigned short int* data, bool inputAsYUV444 = false, bool inputAsRGB32 = false, bool inputAsGrayscale = false);
    void loadImage(int rows, int cols, size_t stride, float* data, bool inputAsYUV444 = false, bool inputAsRGB32 = false, bool inputAsGrayscale = false);
    void loadImage(int rows, int cols, size_t stride, unsigned char* r, unsigned char* g, unsigned char* b, bool inputAsYUV444 = false);
    void loadImage(int rows, int cols, size_t stride, unsigned short int* r, unsigned short int* g, unsigned short int* b, bool inputAsYUV444 = false);
    void loadImage(int rows, int cols, size_t stride, float* r, float* g, float* b, bool inputAsYUV444 = false);
    void loadImage(
        int rowsY, int colsY, size_t strideY, unsigned char* y,
        int rowsU, int colsU, size_t strideU, unsigned char* u,
        int rowsV, int colsV, size_t strideV, unsigned char* v);
    void loadImage(
        int rowsY, int colsY, size_t strideY, unsigned short int* y,
        int rowsU, int colsU, size_t strideU, unsigned short int* u,
        int rowsV, int colsV, size_t strideV, unsigned short int* v);
    void loadImage(
        int rowsY, int colsY, size_t strideY, float* y,
        int rowsU, int colsU, size_t strideU, float* u,
        int rowsV, int colsV, size_t strideV, float* v);
    void loadImage(const cv::Mat& y, const cv::Mat& u, const cv::Mat& v);
    void setVideoSaveInfo(const std::string& dstFile, const CODEC codec = CODEC::MP4V, const double fps = 0.0);
    void saveImage(const std::string& dstFile);
    void saveImage(cv::Mat& dstImage);
    void saveImage(cv::Mat& r, cv::Mat& g, cv::Mat& b);
    void saveImage(unsigned char* data, size_t dstStride = 0);
    void saveImage(
        unsigned char* r, size_t dstStrideR, 
        unsigned char* g, size_t dstStrideG, 
        unsigned char* b, size_t dstStrideB);
    void saveVideo();

    virtual std::string getInfo();
    virtual std::string getFiltersInfo();

    //R2B = true will exchange R channel and B channel
    void showImage(bool R2B = false);

    void process();
    // for video processing
    void processWithPrintProgress();
    void processWithProgress(const std::function<void(double)>&& callBack);
    void stopVideoProcess() noexcept;
    void pauseVideoProcess();
    void continueVideoProcess() noexcept;
    virtual Processor::Type getProcessorType() noexcept = 0;
    virtual std::string getProcessorInfo() = 0;
private:
    void initVideoIO();
    void releaseVideoIO() noexcept;
protected:
    virtual void processYUVImageB() = 0;
    virtual void processRGBImageB() = 0;
    virtual void processGrayscaleB() = 0;
    virtual void processRGBVideoB() = 0;

    virtual void processYUVImageW() = 0;
    virtual void processRGBImageW() = 0;
    virtual void processGrayscaleW() = 0;

    virtual void processYUVImageF() = 0;
    virtual void processRGBImageF() = 0;
    virtual void processGrayscaleF() = 0;
private:
    double fps;
    double totalFrameCount;
    cv::Mat alphaChannel;
    bool inputRGB32 = false;
    bool checkAlphaChannel = false;
    bool inputYUV = false;
    bool inputGrayscale = false;
protected:
    int bitDepth = 8;
    int orgH, orgW, H, W;
    cv::Mat orgImg, dstImg;
    cv::Mat orgY, orgU, orgV;
    cv::Mat dstY, dstU, dstV;
    Utils::VideoIO* videoIO = nullptr;

    Parameters param;
};
