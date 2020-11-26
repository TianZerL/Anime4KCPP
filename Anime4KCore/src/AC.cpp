#define DLL

#include "AC.hpp"

Anime4KCPP::AC::AC(const Parameters& parameters) :param(parameters)
{
    orgH = orgW = H = W = 0;
    totalFrameCount = fps = 0.0;

    if (param.videoMode)
        initVideoIO();
}

Anime4KCPP::AC::~AC()
{
    orgImg.release();
    dstImg.release();
    orgY.release();
    orgU.release();
    orgV.release();
    dstY.release();
    dstU.release();
    dstV.release();
    alphaChannel.release();
    releaseVideoIO();
}

void Anime4KCPP::AC::setArguments(const Parameters& parameters)
{
    param = parameters;

    orgH = orgW = H = W = 0;
    totalFrameCount = fps = 0.0;

    if (param.videoMode)
        initVideoIO();
}

void Anime4KCPP::AC::setVideoMode(const bool value)
{
    param.videoMode = value;
    if (param.videoMode)
        initVideoIO();
}

void Anime4KCPP::AC::loadVideo(const std::string& srcFile)
{
    if (!videoIO->openReader(srcFile))
        throw ACException<ExceptionType::IO>("Failed to load file: file doesn't not exist or decoder isn't installed.");
    orgH = videoIO->get(cv::CAP_PROP_FRAME_HEIGHT);
    orgW = videoIO->get(cv::CAP_PROP_FRAME_WIDTH);
    fps = videoIO->get(cv::CAP_PROP_FPS);
    totalFrameCount = videoIO->get(cv::CAP_PROP_FRAME_COUNT);
    H = param.zoomFactor * orgH;
    W = param.zoomFactor * orgW;
}

void Anime4KCPP::AC::loadImage(const std::string& srcFile)
{
    if (!param.alpha)
        dstImg = orgImg = cv::imread(srcFile, cv::IMREAD_COLOR);
    else
    {
        orgImg = cv::imread(srcFile, cv::IMREAD_UNCHANGED);
        switch (orgImg.channels())
        {
        case 4:
            cv::extractChannel(orgImg, alphaChannel, A);
            cv::resize(alphaChannel, alphaChannel, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_LANCZOS4);
            cv::cvtColor(orgImg, orgImg, cv::COLOR_BGRA2BGR);
            dstImg = orgImg;
            checkAlphaChannel = true;
            break;
        case 3:
            dstImg = orgImg;
            checkAlphaChannel = false;
            break;
        case 1:
            cv::cvtColor(orgImg, orgImg, cv::COLOR_GRAY2BGR);
            dstImg = orgImg;
            checkAlphaChannel = false;
            break;
        default:
            throw ACException<ExceptionType::IO>("Failed to load file: incorrect file format.");
        }
    }
    if (orgImg.empty())
        throw ACException<ExceptionType::IO>("Failed to load file: file doesn't exist or incorrect file format.");
    
    orgH = orgImg.rows;
    orgW = orgImg.cols;
    H = param.zoomFactor * orgH;
    W = param.zoomFactor * orgW;

    highPrecisionMode = false;
}

void Anime4KCPP::AC::loadImage(cv::InputArray srcImage)
{
    orgImg = srcImage.getMat();
    if (orgImg.empty())
        throw ACException<ExceptionType::RunTimeError>("Empty image.");
    switch (orgImg.type())
    {
    case CV_8UC3:
        dstImg = orgImg;
        inputRGB32 = false;
        checkAlphaChannel = false;
        highPrecisionMode = false;
        break;
    case CV_8UC4:
        if (!param.alpha)
        {
            inputRGB32 = true;
            checkAlphaChannel = false;
            cv::cvtColor(orgImg, orgImg, cv::COLOR_RGBA2RGB);
        }
        else
        {
            inputRGB32 = false;
            checkAlphaChannel = true;
            cv::extractChannel(orgImg, alphaChannel, A);
            cv::resize(alphaChannel, alphaChannel, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_LANCZOS4);
            cv::cvtColor(orgImg, orgImg, cv::COLOR_BGRA2BGR);
            dstImg = orgImg;
        }
        highPrecisionMode = false;
        break;
    case CV_32FC3:
        dstImg = orgImg;
        inputRGB32 = false;
        checkAlphaChannel = false;
        highPrecisionMode = true;
        break;
    case CV_32FC4:
        if (!param.alpha)
        {
            inputRGB32 = true;
            checkAlphaChannel = false;
            cv::cvtColor(orgImg, orgImg, cv::COLOR_RGBA2RGB);
        }
        else
        {
            inputRGB32 = false;
            checkAlphaChannel = true;
            cv::extractChannel(orgImg, alphaChannel, A);
            cv::resize(alphaChannel, alphaChannel, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_LANCZOS4);
            cv::cvtColor(orgImg, orgImg, cv::COLOR_BGRA2BGR);
            dstImg = orgImg;
        }
        highPrecisionMode = true;
        break;
    default:
        throw ACException<ExceptionType::RunTimeError>("Error data type.");
    }
    orgH = orgImg.rows;
    orgW = orgImg.cols;
    H = param.zoomFactor * orgH;
    W = param.zoomFactor * orgW;

    inputYUV = false;
}

void Anime4KCPP::AC::loadImage(int rows, int cols, unsigned char* data, size_t bytesPerLine, bool inputAsYUV444, bool inputAsRGB32)
{
    switch (inputAsRGB32 + inputAsYUV444)
    {
    case 0:
        inputRGB32 = false;
        inputYUV = false;
        orgImg = cv::Mat(rows, cols, CV_8UC3, data, bytesPerLine);
        break;
    case 1:
        if (inputAsRGB32)
        {
            inputRGB32 = true;
            inputYUV = false;
            cv::cvtColor(cv::Mat(rows, cols, CV_8UC4, data, bytesPerLine), orgImg, cv::COLOR_RGBA2RGB);
        }
        else //YUV444
        {
            inputRGB32 = false;
            inputYUV = true;
            orgImg = cv::Mat(rows, cols, CV_8UC3, data, bytesPerLine);

            std::vector<cv::Mat> yuv(3);
            cv::split(orgImg, yuv);
            dstY = orgY = yuv[Y];
            dstU = orgU = yuv[U];
            dstV = orgV = yuv[V];
        }
        break;
    case 2:
        throw ACException<ExceptionType::IO>("Failed to load data: inputAsRGB32 and inputAsYUV444 can't be ture at same time.");
    }

    dstImg = orgImg;
    orgH = rows;
    orgW = cols;
    H = param.zoomFactor * orgH;
    W = param.zoomFactor * orgW;

    checkAlphaChannel = false;
    highPrecisionMode = false;
}

void Anime4KCPP::AC::loadImage(int rows, int cols, float* data, size_t bytesPerLine, bool inputAsYUV444, bool inputAsRGB32)
{
    switch (inputAsRGB32 + inputAsYUV444)
    {
    case 0:
        inputRGB32 = false;
        inputYUV = false;
        orgImg = cv::Mat(rows, cols, CV_32FC3, data, bytesPerLine);
        break;
    case 1:
        if (inputAsRGB32)
        {
            inputRGB32 = true;
            inputYUV = false;
            cv::cvtColor(cv::Mat(rows, cols, CV_32FC4, data, bytesPerLine), orgImg, cv::COLOR_RGBA2RGB);
        }
        else //YUV444
        {
            inputRGB32 = false;
            inputYUV = true;
            orgImg = cv::Mat(rows, cols, CV_32FC3, data, bytesPerLine);

            std::vector<cv::Mat> yuv(3);
            cv::split(orgImg, yuv);
            dstY = orgY = yuv[Y];
            dstU = orgU = yuv[U];
            dstV = orgV = yuv[V];
        }
        break;
    case 2:
        throw ACException<ExceptionType::IO>("Failed to load data: inputAsRGB32 and inputAsYUV444 can't be ture at same time.");
    }

    dstImg = orgImg;
    orgH = rows;
    orgW = cols;
    H = param.zoomFactor * orgH;
    W = param.zoomFactor * orgW;

    checkAlphaChannel = false;
    highPrecisionMode = true;
}

void Anime4KCPP::AC::loadImage(int rows, int cols, unsigned char* r, unsigned char* g, unsigned char* b, bool inputAsYUV444)
{
    if (inputAsYUV444)
    {
        inputYUV = true;
        dstY = orgY = cv::Mat(rows, cols, CV_8UC1, r);
        dstU = orgU = cv::Mat(rows, cols, CV_8UC1, g);
        dstV = orgV = cv::Mat(rows, cols, CV_8UC1, b);
    }
    else
    {
        inputYUV = false;
        cv::merge(std::vector<cv::Mat>{
                cv::Mat(rows, cols, CV_8UC1, b),
                cv::Mat(rows, cols, CV_8UC1, g),
                cv::Mat(rows, cols, CV_8UC1, r) },
            orgImg);
        dstImg = orgImg;
    }
    orgH = rows;
    orgW = cols;
    H = param.zoomFactor * orgH;
    W = param.zoomFactor * orgW;

    inputRGB32 = false;
    checkAlphaChannel = false;
    highPrecisionMode = false;
}

void Anime4KCPP::AC::loadImage(int rows, int cols, float* r, float* g, float* b, bool inputAsYUV444)
{
    if (inputAsYUV444)
    {
        inputYUV = true;
        dstY = orgY = cv::Mat(rows, cols, CV_32FC1, r);
        dstU = orgU = cv::Mat(rows, cols, CV_32FC1, g);
        dstV = orgV = cv::Mat(rows, cols, CV_32FC1, b);
    }
    else
    {
        inputYUV = false;
        cv::merge(std::vector<cv::Mat>{
            cv::Mat(rows, cols, CV_32FC1, b),
                cv::Mat(rows, cols, CV_32FC1, g),
                cv::Mat(rows, cols, CV_32FC1, r) },
            orgImg);
        dstImg = orgImg;
    }
    orgH = rows;
    orgW = cols;
    H = param.zoomFactor * orgH;
    W = param.zoomFactor * orgW;

    inputRGB32 = false;
    checkAlphaChannel = false;
    highPrecisionMode = true;
}

void Anime4KCPP::AC::loadImage(int rowsY, int colsY, unsigned char* y, int rowsU, int colsU, unsigned char* u, int rowsV, int colsV, unsigned char* v)
{
    dstY = orgY = cv::Mat(rowsY, colsY, CV_8UC1, y);
    dstU = orgU = cv::Mat(rowsU, colsU, CV_8UC1, u);
    dstV = orgV = cv::Mat(rowsV, colsV, CV_8UC1, v);
    orgH = rowsY;
    orgW = colsY;
    H = param.zoomFactor * orgH;
    W = param.zoomFactor * orgW;

    inputYUV = true;
    inputRGB32 = false;
    checkAlphaChannel = false;
    highPrecisionMode = false;
}

void Anime4KCPP::AC::loadImage(int rowsY, int colsY, float* y, int rowsU, int colsU, float* u, int rowsV, int colsV, float* v)
{
    dstY = orgY = cv::Mat(rowsY, colsY, CV_32FC1, y);
    dstU = orgU = cv::Mat(rowsU, colsU, CV_32FC1, u);
    dstV = orgV = cv::Mat(rowsV, colsV, CV_32FC1, v);
    orgH = rowsY;
    orgW = colsY;
    H = param.zoomFactor * orgH;
    W = param.zoomFactor * orgW;

    inputYUV = true;
    inputRGB32 = false;
    checkAlphaChannel = false;
    highPrecisionMode = true;
}

void Anime4KCPP::AC::loadImage(const cv::Mat& y, const cv::Mat& u, const cv::Mat& v)
{
    dstY = orgY = y;
    dstU = orgU = u;
    dstV = orgV = v;
    orgH = y.rows;
    orgW = y.cols;
    H = param.zoomFactor * orgH;
    W = param.zoomFactor * orgW;

    inputYUV = true;
    inputRGB32 = false;
    checkAlphaChannel = false;

    if (y.type() == CV_32FC1)
        highPrecisionMode = true;
    else
        highPrecisionMode = false;
}

void Anime4KCPP::AC::setVideoSaveInfo(const std::string& dstFile, const CODEC codec, const double fps)
{
    if (!videoIO->openWriter(dstFile, codec, cv::Size(W, H), fps))
        throw ACException<ExceptionType::IO>("Failed to initialize video writer.");
}

void Anime4KCPP::AC::saveImage(const std::string& dstFile)
{
    if (inputYUV)
    {
        if (dstY.size() != dstU.size())
            cv::resize(dstU, dstU, dstY.size(), 0.0, 0.0, cv::INTER_LANCZOS4);
        if (dstY.size() != dstV.size())
            cv::resize(dstV, dstV, dstY.size(), 0.0, 0.0, cv::INTER_LANCZOS4);
        cv::merge(std::vector<cv::Mat>{ dstY, dstU, dstV }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
    if (highPrecisionMode)
    {
        dstImg.convertTo(dstImg, CV_8UC(dstImg.channels()), 255.0);
    }
    if (checkAlphaChannel)
    {
        std::string fileSuffix = dstFile.substr(dstFile.rfind('.'));
        if (std::string(".jpg.jpeg.bmp").find(fileSuffix) != std::string::npos)
        {
            cv::Mat tmp;
            cv::cvtColor(alphaChannel, tmp, cv::COLOR_GRAY2BGR);
            tmp.convertTo(tmp, CV_32FC3, 1.0 / 255.0);
            cv::multiply(dstImg, tmp, dstImg, 1.0, CV_8UC3);
        }
        else
            cv::merge(std::vector<cv::Mat>{ dstImg, alphaChannel }, dstImg);
    }

    cv::imwrite(dstFile, dstImg);
}

void Anime4KCPP::AC::saveImage(cv::Mat& dstImage)
{
    if (inputYUV)
    {
        if (dstY.size() == dstU.size() && dstU.size() == dstV.size())
            cv::merge(std::vector<cv::Mat>{ dstY, dstU, dstV }, dstImg);
        else
            throw ACException<ExceptionType::IO>("Only YUV444 can be saved to opencv Mat");
    }
    else if (inputRGB32)
        cv::cvtColor(dstImg, dstImg, cv::COLOR_RGB2RGBA);
    else if (checkAlphaChannel)
        cv::merge(std::vector<cv::Mat>{ dstImg, alphaChannel }, dstImg);

    dstImage = dstImg;
}

void Anime4KCPP::AC::saveImage(cv::Mat& r, cv::Mat& g, cv::Mat& b)
{
    if (inputYUV)
    {
        r = dstY;
        g = dstU;
        b = dstV;
    }
    else
    {
        std::vector<cv::Mat> bgr(3);
        cv::split(dstImg, bgr);
        r = bgr[R];
        g = bgr[G];
        b = bgr[B];
    }
}

void Anime4KCPP::AC::saveImage(unsigned char* data)
{
    if (data == nullptr)
        throw ACException<ExceptionType::RunTimeError>("Pointer can not be nullptr");
    if (highPrecisionMode)
        throw ACException<ExceptionType::RunTimeError>("High precision mode expect a float pointer");
    if (inputYUV)
    {
        if (dstY.size() == dstU.size() && dstU.size() == dstV.size())
            cv::merge(std::vector<cv::Mat>{ dstY, dstU, dstV }, dstImg);
        else
            throw ACException<ExceptionType::IO>("Only YUV444 can be saved to data pointer");
    }
    else if (inputRGB32)
        cv::cvtColor(dstImg, dstImg, cv::COLOR_RGB2RGBA);

    memcpy(data, dstImg.data, dstImg.step * H);
}

void Anime4KCPP::AC::saveImage(float* data)
{
    if (data == nullptr)
        throw ACException<ExceptionType::RunTimeError>("Pointer can not be nullptr");
    if(!highPrecisionMode)
        throw ACException<ExceptionType::RunTimeError>("Non high precision mode expect a unsigned char pointer");
    if (inputYUV)
    {
        if (dstY.size() == dstU.size() && dstU.size() == dstV.size())
            cv::merge(std::vector<cv::Mat>{ dstY, dstU, dstV }, dstImg);
        else
            throw ACException<ExceptionType::IO>("Only YUV444 can be saved to data pointer");
    }
    else if (inputRGB32)
        cv::cvtColor(dstImg, dstImg, cv::COLOR_RGB2RGBA);

    std::copy(reinterpret_cast<float*>(dstImg.data),
        reinterpret_cast<float*>(dstImg.data) + dstImg.step * H, data);
}

void Anime4KCPP::AC::saveImage(unsigned char* r, unsigned char* g, unsigned char* b)
{
    if (r == nullptr || g == nullptr || b == nullptr)
        throw ACException<ExceptionType::RunTimeError>("Pointers can not be nullptr");
    if (highPrecisionMode)
        throw ACException<ExceptionType::RunTimeError>("High precision mode expect a float pointer");
    if (inputYUV)
    {
        memcpy(r, dstY.data, (size_t)dstY.cols * (size_t)dstY.rows);
        memcpy(g, dstU.data, (size_t)dstU.cols * (size_t)dstU.rows);
        memcpy(b, dstV.data, (size_t)dstV.cols * (size_t)dstV.rows);
    }
    else
    {
        size_t size = (size_t)W * (size_t)H;
        std::vector<cv::Mat> bgr(3);
        cv::split(dstImg, bgr);
        memcpy(r, bgr[R].data, size);
        memcpy(g, bgr[G].data, size);
        memcpy(b, bgr[B].data, size);
    }
}

void Anime4KCPP::AC::saveImage(float* r, float* g, float* b)
{
    if (r == nullptr || g == nullptr || b == nullptr)
        throw ACException<ExceptionType::RunTimeError>("Pointers can not be nullptr");
    if (!highPrecisionMode)
        throw ACException<ExceptionType::RunTimeError>("Non high precision mode expect a unsigned char pointer");
    if (inputYUV)
    {
        std::copy(reinterpret_cast<float*>(dstY.data),
            reinterpret_cast<float*>(dstY.data) +
            static_cast<size_t>(dstY.cols) * static_cast<size_t>(dstY.rows), r);
        std::copy(reinterpret_cast<float*>(dstU.data),
            reinterpret_cast<float*>(dstU.data) + 
            static_cast<size_t>(dstU.cols) * static_cast<size_t>(dstU.rows), g);
        std::copy(reinterpret_cast<float*>(dstV.data),
            reinterpret_cast<float*>(dstV.data) + 
            static_cast<size_t>(dstV.cols) * static_cast<size_t>(dstV.rows), b);
    }
    else
    {
        size_t size = static_cast<size_t>(W) * static_cast<size_t>(H);
        std::vector<cv::Mat> bgr(3);
        cv::split(dstImg, bgr);
        std::copy(reinterpret_cast<float*>(bgr[R].data),
            reinterpret_cast<float*>(bgr[R].data) + size, r);
        std::copy(reinterpret_cast<float*>(bgr[G].data),
            reinterpret_cast<float*>(bgr[G].data) + size, g);
        std::copy(reinterpret_cast<float*>(bgr[B].data),
            reinterpret_cast<float*>(bgr[B].data) + size, b);
    }
}

void Anime4KCPP::AC::saveVideo()
{
    videoIO->release();
}

std::string Anime4KCPP::AC::getInfo()
{
    std::ostringstream oss;
    Processor::Type type = getProcessorType();
    oss << "----------------------------------------------" << std::endl
        << "Parameter Infomation" << std::endl
        << "----------------------------------------------" << std::endl;
    if (param.videoMode)
    {
        oss << "FPS: " << fps << std::endl
            << "Threads: " << param.maxThreads << std::endl
            << "Total frames: " << totalFrameCount << std::endl;
    }
    if (orgW && orgH)
    {
        oss << orgW << "x" << orgH << " to " << W << "x" << H << std::endl
            << "----------------------------------------------" << std::endl;
    }
    oss << "Processor type: " << type << std::endl;
    
    return oss.str();
}

std::string Anime4KCPP::AC::getFiltersInfo()
{
    std::ostringstream oss;
    oss << "----------------------------------------------" << std::endl
        << "Filter Infomation" << std::endl
        << "----------------------------------------------" << std::endl;

    return oss.str();
}

size_t Anime4KCPP::AC::getResultDataLength() noexcept
{
    if (inputYUV)
        return dstY.size().area() + dstU.size().area() + dstV.size().area();
    else if (checkAlphaChannel || inputRGB32)
        return 4 * static_cast<size_t>(H) * static_cast<size_t>(W);
    else
        return 3 * static_cast<size_t>(H) * static_cast<size_t>(W);
}

size_t Anime4KCPP::AC::getResultDataPerChannelLength() noexcept
{
    return static_cast<size_t>(W) * static_cast<size_t>(H);
}

std::array<int, 3> Anime4KCPP::AC::getResultShape()
{
    std::array<int, 3> shape = { H, W, 3 };
    return shape;
}

void Anime4KCPP::AC::showImage(bool R2B)
{
    cv::Mat tmpImg = dstImg;

    if (R2B)
        cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2RGB);

    if (inputYUV)
    {
        cv::Mat tmpU, tmpV;
        if (dstY.size() != dstU.size())
            cv::resize(dstU, tmpU, dstY.size(), 0.0, 0.0, cv::INTER_LANCZOS4);
        if (dstY.size() != dstV.size())
            cv::resize(dstV, tmpV, dstY.size(), 0.0, 0.0, cv::INTER_LANCZOS4);
        cv::merge(std::vector<cv::Mat>{ dstY, tmpU, tmpV }, tmpImg);
        cv::cvtColor(tmpImg, tmpImg, cv::COLOR_YUV2BGR);
    }

    if (checkAlphaChannel)
    {
        cv::Mat tmp;
        cv::cvtColor(alphaChannel, tmp, cv::COLOR_GRAY2BGR);
        tmp.convertTo(tmp, CV_32FC3, 1.0 / 255.0);
        cv::multiply(tmpImg, tmp, tmpImg, 1.0, CV_8UC3);
    }

    cv::imshow("preview", tmpImg);
    cv::waitKey();
}

void Anime4KCPP::AC::process()
{
    if (highPrecisionMode)
    {
        if (!param.videoMode)
        {
            if (inputYUV)
                processYUVImageF();
            else
                processRGBImageF();
        }
    }
    else
    {
        if (!param.videoMode)
        {
            if (inputYUV)
                processYUVImageB();
            else
                processRGBImageB();
        }
        else
        {
            processRGBVideoB();
        }
    }
}

void Anime4KCPP::AC::processWithPrintProgress()
{
    if (!param.videoMode)
    {
        process();
        return;
    }
    std::future<void> p = std::async(&AC::process, this);
    std::chrono::milliseconds timeout(1000);
    std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();
    for (;;)
    {
        std::future_status status = p.wait_for(timeout);
        if (status == std::future_status::ready)
        {
            std::cout
                << std::fixed << std::setprecision(2)
                << std::setw(5) << 100.0 << '%'
                << "  elpsed: " << std::setw(5) << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - s).count() / 1000.0 << 's'
                << "  remaining: " << std::setw(5) << 0.0 << 's'
                << std::endl;
            // get any possible exception
            p.get();
            break;
        }
        std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
        double currTime = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0;
        double progress = videoIO->getProgress();
        std::cout
            << std::fixed << std::setprecision(2)
            << std::setw(5) << progress * 100 << '%'
            << "  elpsed: " << std::setw(5) << currTime << 's'
            << "  remaining: " << std::setw(5) << currTime / progress - currTime << 's'
            << '\r';
    }
}

void Anime4KCPP::AC::processWithProgress(const std::function<void(double)>&& callBack)
{
    if (!param.videoMode)
    {
        process();
        return;
    }
    std::future<void> p = std::async(&AC::process, this);
    std::chrono::milliseconds timeout(1000);
    for (;;)
    {
        std::future_status status = p.wait_for(timeout);
        if (status == std::future_status::ready)
        {
            callBack(1.0);
            p.get();
            break;
        }
        double progress = videoIO->getProgress();
        callBack(progress);
    }
}

void Anime4KCPP::AC::stopVideoProcess() noexcept
{
    if (param.videoMode)
        videoIO->stopProcess();
}

void Anime4KCPP::AC::pauseVideoProcess()
{
    if (param.videoMode && !videoIO->isPaused())
    {
        std::thread t(&Utils::VideoIO::pauseProcess, videoIO);
        t.detach();
    }
}

void Anime4KCPP::AC::continueVideoProcess() noexcept
{
    if (param.videoMode)
        videoIO->continueProcess();
}

inline void Anime4KCPP::AC::initVideoIO()
{
    if (videoIO == nullptr)
        videoIO = new Utils::VideoIO;
}

inline void Anime4KCPP::AC::releaseVideoIO() noexcept
{
    if (videoIO != nullptr)
    {
        delete videoIO;
        videoIO = nullptr;
    }
}

void Anime4KCPP::Parameters::reset() noexcept
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
    HDN = false;
    HDNLevel = 1;
    alpha = false;
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
    unsigned int maxThreads,
    bool HDN,
    int HDNLevel,
    bool alpha
) noexcept :
    passes(passes), pushColorCount(pushColorCount),
    strengthColor(strengthColor), strengthGradient(strengthGradient),
    zoomFactor(zoomFactor), fastMode(fastMode), videoMode(videoMode),
    preprocessing(preprocessing), postprocessing(postprocessing),
    preFilters(preFilters), postFilters(postFilters), maxThreads(maxThreads),
    HDN(HDN), HDNLevel(HDNLevel), alpha(alpha) {}
