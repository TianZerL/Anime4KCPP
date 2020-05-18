#define DLL

#include "Anime4K.h"

Anime4KCPP::Anime4K::Anime4K(const Parameters& parameters)
{
    ps = parameters.passes;
    pcc = parameters.pushColorCount;
    sc = parameters.strengthColor;
    sg = parameters.strengthGradient;
    zf = parameters.zoomFactor;
    fm = parameters.fastMode;
    vm = parameters.videoMode;
    pre = parameters.preprocessing;
    post = parameters.postprocessing;
    pref = parameters.preFilters;
    postf = parameters.postFilters;
    mt = parameters.maxThreads;

    orgH = orgW = H = W = 0;
    fps = 0.0;
    frameCount = totalFrameCount = 0;
}

Anime4KCPP::Anime4K::~Anime4K()
{
    orgImg.release();
    dstImg.release();
    videoWriter.release();
    video.release();
}

void Anime4KCPP::Anime4K::setArguments(const Parameters& parameters)
{
    ps = parameters.passes;
    pcc = parameters.pushColorCount;
    sc = parameters.strengthColor;
    sg = parameters.strengthGradient;
    zf = parameters.zoomFactor;
    fm = parameters.fastMode;
    vm = parameters.videoMode;
    pre = parameters.preprocessing;
    post = parameters.postprocessing;
    pref = parameters.preFilters;
    postf = parameters.postFilters;
    mt = parameters.maxThreads;

    orgH = orgW = H = W = 0;
    fps = 0.0;
    frameCount = totalFrameCount = 0;
}

void Anime4KCPP::Anime4K::setVideoMode(const bool flag)
{
    vm = flag;
}

void Anime4KCPP::Anime4K::loadVideo(const std::string& srcFile)
{
    video.open(srcFile);
    if (!video.isOpened())
        throw "Failed to load file: file doesn't not exist or decoder isn't installed.";
    orgH = video.get(cv::CAP_PROP_FRAME_HEIGHT);
    orgW = video.get(cv::CAP_PROP_FRAME_WIDTH);
    fps = video.get(cv::CAP_PROP_FPS);
    totalFrameCount = video.get(cv::CAP_PROP_FRAME_COUNT);
    H = zf * orgH;
    W = zf * orgW;
}

void Anime4KCPP::Anime4K::loadImage(const std::string& srcFile)
{
    dstImg = orgImg = cv::imread(srcFile, cv::IMREAD_COLOR);
    if (orgImg.empty())
        throw "Failed to load file: file doesn't not exist or incorrect file format.";
    orgH = orgImg.rows;
    orgW = orgImg.cols;
    H = zf * orgH;
    W = zf * orgW;
}

void Anime4KCPP::Anime4K::loadImage(cv::InputArray srcImage)
{
    dstImg = orgImg = srcImage.getMat();
    if (orgImg.empty() || orgImg.type() != CV_8UC3)
        throw "Empty image or it is not a BGR image data.";
    orgH = orgImg.rows;
    orgW = orgImg.cols;
    H = zf * orgH;
    W = zf * orgW;
}

void Anime4KCPP::Anime4K::loadImage(int rows, int cols, unsigned char* data, size_t bytesPerLine)
{
    dstImg = orgImg = cv::Mat(rows, cols, CV_8UC3, data, bytesPerLine);
    orgH = rows;
    orgW = cols;
    H = zf * orgH;
    W = zf * orgW;
}

void Anime4KCPP::Anime4K::loadImage(int rows, int cols, unsigned char* r, unsigned char* g, unsigned char* b)
{
    cv::merge(std::vector{
    cv::Mat(rows, cols, CV_8UC1, b),
    cv::Mat(rows, cols, CV_8UC1, g),
    cv::Mat(rows, cols, CV_8UC1, r) },
    orgImg);
    dstImg = orgImg;
    orgH = rows;
    orgW = cols;
    H = zf * orgH;
    W = zf * orgW;
}

void Anime4KCPP::Anime4K::setVideoSaveInfo(const std::string& dstFile, const CODEC codec)
{
    switch (codec)
    {
    case CODEC::MP4V:
        videoWriter.open(dstFile, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(W, H));
        if (!videoWriter.isOpened())
        {
            videoWriter.open(dstFile, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(W, H));
            if (!videoWriter.isOpened())
                throw "Failed to initialize video writer.";
        }
        break;
#ifdef _WIN32 //DXVA encoding for windows
    case CODEC::DXVA:
        videoWriter.open(dstFile, cv::CAP_MSMF, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), std::ceil(fps), cv::Size(W, H));
        if (!videoWriter.isOpened())
        {
            videoWriter.open(dstFile, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(W, H));
            if (!videoWriter.isOpened())
                throw "Failed to initialize video writer.";
        }
        break;
#endif
    case CODEC::AVC1:
        videoWriter.open(dstFile, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, cv::Size(W, H));
        if (!videoWriter.isOpened())
        {
            videoWriter.open(dstFile, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(W, H));
            if (!videoWriter.isOpened())
                throw "Failed to initialize video writer.";
        }
        break;
    case CODEC::VP09:
        videoWriter.open(dstFile, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('v', 'p', '0', '9'), fps, cv::Size(W, H));
        if (!videoWriter.isOpened())
        {
            videoWriter.open(dstFile, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(W, H));
            if (!videoWriter.isOpened())
                throw "Failed to initialize video writer.";
        }
        break;
    case CODEC::HEVC:
        videoWriter.open(dstFile, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('h', 'e', 'v', '1'), fps, cv::Size(W, H));
        if (!videoWriter.isOpened())
        {
            videoWriter.open(dstFile, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(W, H));
            if (!videoWriter.isOpened())
                throw "Failed to initialize video writer.";
        }
        break;
    case CODEC::AV01:
        videoWriter.open(dstFile, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('a', 'v', '0', '1'), fps, cv::Size(W, H));
        if (!videoWriter.isOpened())
        {
            videoWriter.open(dstFile, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(W, H));
            if (!videoWriter.isOpened())
                throw "Failed to initialize video writer.";
        }
        break;
    case CODEC::OTHER:
        videoWriter.open(dstFile, -1, fps, cv::Size(W, H));
        if (!videoWriter.isOpened())
            throw "Failed to initialize video writer.";
        break;
    default:
        videoWriter.open(dstFile, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(W, H));
        if (!videoWriter.isOpened())
            throw "Failed to initialize video writer.";
    }
}

void Anime4KCPP::Anime4K::saveImage(const std::string& dstFile)
{
    cv::imwrite(dstFile, dstImg);
}

void Anime4KCPP::Anime4K::saveImage(cv::Mat& dstImage)
{
    dstImage = dstImg;
}

void Anime4KCPP::Anime4K::saveImage(unsigned char*& data)
{
    if (data == nullptr)
        throw "Pointer can not be nullptr";
    size_t size = dstImg.step * H;
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGR2RGB);
    memcpy(data, dstImg.data, size);
}

void Anime4KCPP::Anime4K::saveImage(unsigned char*& r, unsigned char*& g, unsigned char*& b)
{
    if (r == nullptr || g == nullptr || b == nullptr)
        throw "Pointers can not be nullptr";
    size_t size = (size_t)W * (size_t)H;
    std::vector<cv::Mat> bgr(3);
    cv::split(dstImg, bgr);
    memcpy(r, bgr[R].data, size);
    memcpy(g, bgr[G].data, size);
    memcpy(b, bgr[B].data, size);
}

void Anime4KCPP::Anime4K::saveVideo()
{
    videoWriter.release();
    video.release();
}

void Anime4KCPP::Anime4K::showInfo()
{
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "Welcome to Anime4KCPP" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    if (vm)
    {
        std::cout << "FPS: " << fps << std::endl;
        std::cout << "Threads: " << mt << std::endl;
        std::cout << "Total frames: " << totalFrameCount << std::endl;
    }
    std::cout << orgW << "x" << orgH << " to " << W << "x" << H << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "Passes: " << ps << std::endl
        << "pushColorCount: " << pcc << std::endl
        << "Zoom Factor: " << zf << std::endl
        << "Video Mode: " << std::boolalpha << vm << std::endl
        << "Fast Mode: " << std::boolalpha << fm << std::endl
        << "Strength Color: " << sc << std::endl
        << "Strength Gradient: " << sg << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
}

void Anime4KCPP::Anime4K::showFiltersInfo()
{
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "Preprocessing filters list:" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    if (!pre)
    {
        std::cout << "Preprocessing disabled" << std::endl;
    }
    else
    {
        if (pref & MEDIAN_BLUR)
            std::cout << "Median blur" << std::endl;
        if (pref & MEAN_BLUR)
            std::cout << "Mean blur" << std::endl;
        if (pref & CAS_SHARPENING)
            std::cout << "CAS Sharpening" << std::endl;
        if (pref & GAUSSIAN_BLUR_WEAK)
            std::cout << "Gaussian blur weak" << std::endl;
        else if (pref & GAUSSIAN_BLUR)
            std::cout << "Gaussian blur" << std::endl;
        if (pref & BILATERAL_FILTER)
            std::cout << "Bilateral filter" << std::endl;
        else if (pref & BILATERAL_FILTER_FAST)
            std::cout << "Bilateral filter faster" << std::endl;
    }
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "Postprocessing filters list:" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    if (!post)
    {
        std::cout << "Postprocessing disabled" << std::endl;
    }
    else
    {
        if (postf & MEDIAN_BLUR)
            std::cout << "Median blur" << std::endl;
        if (postf & MEAN_BLUR)
            std::cout << "Mean blur" << std::endl;
        if (postf & CAS_SHARPENING)
            std::cout << "CAS Sharpening" << std::endl;
        if (postf & GAUSSIAN_BLUR_WEAK)
            std::cout << "Gaussian blur weak" << std::endl;
        else if (postf & GAUSSIAN_BLUR)
            std::cout << "Gaussian blur" << std::endl;
        if (postf & BILATERAL_FILTER)
            std::cout << "Bilateral filter" << std::endl;
        else if (postf & BILATERAL_FILTER_FAST)
            std::cout << "Bilateral filter faster" << std::endl;
    }
    std::cout << "----------------------------------------------" << std::endl;
}

std::string Anime4KCPP::Anime4K::getInfo()
{
    std::ostringstream oss;
    oss << "----------------------------------------------" << std::endl;
    oss << "Welcome to Anime4KCPP" << std::endl;
    oss << "----------------------------------------------" << std::endl;
    if (vm)
    {
        oss << "FPS: " << fps << std::endl;
        oss << "Threads: " << mt << std::endl;
        oss << "Total frames: " << totalFrameCount << std::endl;
    }
    oss << orgW << "x" << orgH << " to " << W << "x" << H << std::endl;
    oss << "----------------------------------------------" << std::endl;
    oss << "Passes: " << ps << std::endl
        << "pushColorCount: " << pcc << std::endl
        << "Zoom Factor: " << zf << std::endl
        << "Video Mode: " << std::boolalpha << vm << std::endl
        << "Fast Mode: " << std::boolalpha << fm << std::endl
        << "Strength Color: " << sc << std::endl
        << "Strength Gradient: " << sg << std::endl;
    oss << "----------------------------------------------" << std::endl;
    return std::string(oss.str());
}

std::string Anime4KCPP::Anime4K::getFiltersInfo()
{
    std::ostringstream oss;
    oss << "----------------------------------------------" << std::endl;
    oss << "Preprocessing filters list:" << std::endl;
    oss << "----------------------------------------------" << std::endl;
    if (!pre)
    {
        oss << "Preprocessing disabled" << std::endl;
    }
    else
    {
        if (pref & MEDIAN_BLUR)
            oss << "Median blur" << std::endl;
        if (pref & MEAN_BLUR)
            oss << "Mean blur" << std::endl;
        if (pref & CAS_SHARPENING)
            oss << "CAS Sharpening" << std::endl;
        if (pref & GAUSSIAN_BLUR_WEAK)
            oss << "Gaussian blur weak" << std::endl;
        else if (pref & GAUSSIAN_BLUR)
            oss << "Gaussian blur" << std::endl;
        if (pref & BILATERAL_FILTER)
            oss << "Bilateral filter" << std::endl;
        else if (pref & BILATERAL_FILTER_FAST)
            oss << "Bilateral filter faster" << std::endl;
    }
    oss << "----------------------------------------------" << std::endl;
    oss << "Postprocessing filters list:" << std::endl;
    oss << "----------------------------------------------" << std::endl;
    if (!post)
    {
        oss << "Postprocessing disabled" << std::endl;
    }
    else
    {
        if (postf & MEDIAN_BLUR)
            oss << "Median blur" << std::endl;
        if (postf & MEAN_BLUR)
            oss << "Mean blur" << std::endl;
        if (postf & CAS_SHARPENING)
            oss << "CAS Sharpening" << std::endl;
        if (postf & GAUSSIAN_BLUR_WEAK)
            oss << "Gaussian blur weak" << std::endl;
        else if (postf & GAUSSIAN_BLUR)
            oss << "Gaussian blur" << std::endl;
        if (postf & BILATERAL_FILTER)
            oss << "Bilateral filter" << std::endl;
        else if (postf & BILATERAL_FILTER_FAST)
            oss << "Bilateral filter faster" << std::endl;
    }
    oss << "----------------------------------------------" << std::endl;
    return std::string(oss.str());
}

size_t Anime4KCPP::Anime4K::getResultDataLength()
{
    return dstImg.step * static_cast<size_t>(H);
}

size_t Anime4KCPP::Anime4K::getResultDataPerChannelLength()
{
    return static_cast<size_t>(W) * static_cast<size_t>(H);
}

void Anime4KCPP::Anime4K::showImage()
{
    cv::imshow("dstImg", dstImg);
    cv::waitKey();
}

void Anime4KCPP::Parameters::reset()
{
    passes = 2;
    pushColorCount = 2;
    strengthColor = 0.3;
    strengthGradient = 1.0;
    zoomFactor = 2.0;
    fastMode = false;
    videoMode = false;
    preprocessing = false;
    postprocessing = false;
    preFilters = 4;
    postFilters = 40;
    maxThreads = std::thread::hardware_concurrency();
}

Anime4KCPP::Parameters::Parameters(
    int passes,
    int pushColorCount,
    double strengthColor,
    double strengthGradient,
    double zoomFactor,
    bool fastMode,
    bool videoMode,
    bool preprocessing,
    bool postprocessing,
    uint8_t preFilters,
    uint8_t postFilters,
    unsigned int maxThreads
) :
    passes(passes), pushColorCount(pushColorCount),
    strengthColor(strengthColor), strengthGradient(strengthGradient),
    zoomFactor(zoomFactor), fastMode(fastMode), videoMode(videoMode),
    preprocessing(preprocessing), postprocessing(postprocessing),
    preFilters(preFilters), postFilters(postFilters), maxThreads(maxThreads) {}
