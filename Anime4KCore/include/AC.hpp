#pragma once

#include<iostream>
#include<atomic>
#include<future>
#include<cmath>

#include<opencv2/opencv.hpp>

#include"ACException.hpp"
#include"ACProcessor.hpp"

#if defined(_MSC_VER) && !defined(USE_TBB)
#define PARALLEL_LIBRARY "PPL"
#elif defined(USE_TBB)
#define PARALLEL_LIBRARY "TBB"
#else
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
    typedef unsigned char* ChanB;
    typedef unsigned short* ChanW;

    typedef float* PixelF;
    typedef unsigned char* PixelB;
    typedef unsigned short* PixelW;

    typedef float* LineF;
    typedef unsigned char* LineB;
    typedef unsigned short* LineW;

    namespace Utils
    {
        int fastCeilLog2(double v) noexcept;
    }
}

// compute log2(v) then do ceil(v)
inline int Anime4KCPP::Utils::fastCeilLog2(double v) noexcept
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
    bool preprocessing;
    bool postprocessing;
    uint8_t preFilters;
    uint8_t postFilters;
    unsigned int maxThreads;
    bool HDN;
    bool alpha;
    int HDNLevel;

    // return true if zoomFactor is not a power of 2 or zoomFactor < 2.0
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

    virtual void setParameters(const Parameters& parameters);
    Parameters getParameters();

    void loadImage(const std::string& srcFile);
    void loadImage(const cv::Mat& srcImage);
    void loadImage(int rows, int cols, size_t stride, unsigned char* data, bool inputAsYUV444 = false, bool inputAsRGB32 = false, bool inputAsGrayscale = false);
    void loadImage(int rows, int cols, size_t stride, unsigned short* data, bool inputAsYUV444 = false, bool inputAsRGB32 = false, bool inputAsGrayscale = false);
    void loadImage(int rows, int cols, size_t stride, float* data, bool inputAsYUV444 = false, bool inputAsRGB32 = false, bool inputAsGrayscale = false);
    void loadImage(int rows, int cols, size_t stride, unsigned char* r, unsigned char* g, unsigned char* b, bool inputAsYUV444 = false);
    void loadImage(int rows, int cols, size_t stride, unsigned short* r, unsigned short* g, unsigned short* b, bool inputAsYUV444 = false);
    void loadImage(int rows, int cols, size_t stride, float* r, float* g, float* b, bool inputAsYUV444 = false);
    void loadImage(
        int rowsY, int colsY, size_t strideY, unsigned char* y,
        int rowsU, int colsU, size_t strideU, unsigned char* u,
        int rowsV, int colsV, size_t strideV, unsigned char* v);
    void loadImage(
        int rowsY, int colsY, size_t strideY, unsigned short* y,
        int rowsU, int colsU, size_t strideU, unsigned short* u,
        int rowsV, int colsV, size_t strideV, unsigned short* v);
    void loadImage(
        int rowsY, int colsY, size_t strideY, float* y,
        int rowsU, int colsU, size_t strideU, float* u,
        int rowsV, int colsV, size_t strideV, float* v);
    void loadImage(const cv::Mat& y, const cv::Mat& u, const cv::Mat& v);
    void saveImage(const std::string& dstFile);
    void saveImage(cv::Mat& dstImage);
    void saveImage(cv::Mat& r, cv::Mat& g, cv::Mat& b);
    void saveImage(unsigned char* data, size_t dstStride = 0);
    void saveImage(
        unsigned char* r, size_t dstStrideR, 
        unsigned char* g, size_t dstStrideG, 
        unsigned char* b, size_t dstStrideB);
    void saveImageBufferSize(size_t& dataSize, size_t dstStride = 0);
    void saveImageBufferSize(
        size_t& rSize, size_t dstStrideR, 
        size_t& gSize, size_t dstStrideG, 
        size_t& bSize, size_t dstStrideB);
    void saveImageShape(int& cols, int& rows, int& channels);
    //R2B = true will exchange R channel and B channel
    void showImage(bool R2B = false);

    void process();

    virtual std::string getInfo();
    virtual std::string getFiltersInfo();
    virtual Processor::Type getProcessorType() noexcept = 0;
    virtual std::string getProcessorInfo() = 0;
protected:
    virtual void processYUVImageB() = 0;
    virtual void processRGBImageB() = 0;
    virtual void processGrayscaleB() = 0;

    virtual void processYUVImageW() = 0;
    virtual void processRGBImageW() = 0;
    virtual void processGrayscaleW() = 0;

    virtual void processYUVImageF() = 0;
    virtual void processRGBImageF() = 0;
    virtual void processGrayscaleF() = 0;
private:
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

    Parameters param;
};
