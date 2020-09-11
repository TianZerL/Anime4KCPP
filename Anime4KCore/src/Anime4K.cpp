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
    HDN = parameters.HDN;
    HDNLevel = parameters.HDNLevel;
    alpha = parameters.alpha;

    orgH = orgW = H = W = 0;
    totalFrameCount = fps = 0.0;

    if (vm)
        initVideoIO();
}

Anime4KCPP::Anime4K::~Anime4K()
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
    HDN = parameters.HDN;
    HDNLevel = parameters.HDNLevel;
    alpha = parameters.alpha;

    orgH = orgW = H = W = 0;
    totalFrameCount = fps = 0.0;

    if (vm)
        initVideoIO();
}

void Anime4KCPP::Anime4K::setVideoMode(const bool flag)
{
    vm = flag;
    if (vm)
        initVideoIO();
}

void Anime4KCPP::Anime4K::loadVideo(const std::string& srcFile)
{
    if (!videoIO->openReader(srcFile))
        throw "Failed to load file: file doesn't not exist or decoder isn't installed.";
    orgH = videoIO->get(cv::CAP_PROP_FRAME_HEIGHT);
    orgW = videoIO->get(cv::CAP_PROP_FRAME_WIDTH);
    fps = videoIO->get(cv::CAP_PROP_FPS);
    totalFrameCount = videoIO->get(cv::CAP_PROP_FRAME_COUNT);
    H = zf * orgH;
    W = zf * orgW;
}

void Anime4KCPP::Anime4K::loadImage(const std::string& srcFile)
{
    if (!alpha)
        dstImg = orgImg = cv::imread(srcFile, cv::IMREAD_COLOR);
    else
    {
        orgImg = cv::imread(srcFile, cv::IMREAD_UNCHANGED);
        switch (orgImg.channels())
        {
        case 4:
            cv::extractChannel(orgImg, alphaChannel, A);
            cv::resize(alphaChannel, alphaChannel, cv::Size(0, 0), zf, zf, cv::INTER_LANCZOS4);
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
            throw "Failed to load file: incorrect file format.";
        }
    }
    if (orgImg.empty())
        throw "Failed to load file: file doesn't exist or incorrect file format.";
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

void Anime4KCPP::Anime4K::loadImage(int rows, int cols, unsigned char* data, size_t bytesPerLine, bool inputAsYUV444, bool inputAsRGB32)
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
            cv::cvtColor(cv::Mat(rows, cols, CV_8UC4, data, bytesPerLine), orgImg, cv::COLOR_RGBA2RGB);
        }
        else //YUV444
        {
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
        throw "Failed to load data: inputAsRGB32 and inputAsYUV444 can't be ture at same time.";
    }

    dstImg = orgImg;
    orgH = rows;
    orgW = cols;
    H = zf * orgH;
    W = zf * orgW;
}

void Anime4KCPP::Anime4K::loadImage(int rows, int cols, unsigned char* r, unsigned char* g, unsigned char* b, bool inputAsYUV444)
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
    H = zf * orgH;
    W = zf * orgW;
}

void Anime4KCPP::Anime4K::loadImage(int rowsY, int colsY, unsigned char* y, int rowsU, int colsU, unsigned char* u, int rowsV, int colsV, unsigned char* v)
{
    inputYUV = true;
    dstY = orgY = cv::Mat(rowsY, colsY, CV_8UC1, y);
    dstU = orgU = cv::Mat(rowsU, colsU, CV_8UC1, u);
    dstV = orgV = cv::Mat(rowsV, colsV, CV_8UC1, v);
    orgH = rowsY;
    orgW = colsY;
    H = zf * orgH;
    W = zf * orgW;
}

void Anime4KCPP::Anime4K::setVideoSaveInfo(const std::string& dstFile, const CODEC codec, const double fps)
{
    if (!videoIO->openWriter(dstFile, codec, cv::Size(W, H), fps))
        throw "Failed to initialize video writer.";
}

void Anime4KCPP::Anime4K::saveImage(const std::string& dstFile)
{
    if (inputYUV)
    {
        if (dstY.size() == dstU.size() && dstU.size() == dstV.size())
        {
            cv::merge(std::vector<cv::Mat>{ dstY, dstU, dstV }, dstImg);
            cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
        }
        else
            throw "Only YUV444 can be saved to file";
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

void Anime4KCPP::Anime4K::saveImage(cv::Mat& dstImage)
{
    if (inputYUV)
    {
        if (dstY.size() == dstU.size() && dstU.size() == dstV.size())
            cv::merge(std::vector<cv::Mat>{ dstY, dstU, dstV }, dstImg);
        else
            throw "Only YUV444 can be saved to opencv Mat";
    }
    else if (inputRGB32)
        cv::cvtColor(dstImg, dstImg, cv::COLOR_RGB2RGBA);
    else if (checkAlphaChannel)
        cv::merge(std::vector<cv::Mat>{ dstImg, alphaChannel }, dstImg);

    dstImage = dstImg;
}

void Anime4KCPP::Anime4K::saveImage(cv::Mat& r, cv::Mat& g, cv::Mat& b)
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

void Anime4KCPP::Anime4K::saveImage(unsigned char*& data)
{
    if (data == nullptr)
        throw "Pointer can not be nullptr";
    if (inputYUV)
    {
        if (dstY.size() == dstU.size() && dstU.size() == dstV.size())
            cv::merge(std::vector<cv::Mat>{ dstY, dstU, dstV }, dstImg);
        else
            throw "Only YUV444 can be saved to data pointer";
    }
    else if (inputRGB32)
        cv::cvtColor(dstImg, dstImg, cv::COLOR_RGB2RGBA);

    size_t size = dstImg.step * H;
    memcpy(data, dstImg.data, size);
}

void Anime4KCPP::Anime4K::saveImage(unsigned char*& r, unsigned char*& g, unsigned char*& b)
{
    if (r == nullptr || g == nullptr || b == nullptr)
        throw "Pointers can not be nullptr";
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

void Anime4KCPP::Anime4K::saveVideo()
{
    videoIO->release();
}

void Anime4KCPP::Anime4K::showInfo()
{
    ProcessorType type = getProcessorType();
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
    std::cout << "Processor type: " << type << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    switch (type)
    {
    case ProcessorType::CPU:
    case ProcessorType::GPU:
        std::cout
            << "Passes: " << ps << std::endl
            << "pushColorCount: " << pcc << std::endl
            << "Zoom Factor: " << zf << std::endl
            << "Video Mode: " << std::boolalpha << vm << std::endl
            << "Fast Mode: " << std::boolalpha << fm << std::endl
            << "Strength Color: " << sc << std::endl
            << "Strength Gradient: " << sg << std::endl;
        break;
    case ProcessorType::CPUCNN:
    case ProcessorType::GPUCNN:
        std::cout
            << "Zoom Factor: " << zf << std::endl
            << "HDN Mode: " << std::boolalpha << HDN << std::endl;
        if (HDN)
            std::cout
            << "HDN level: " << HDNLevel << std::endl;
        break;
    }
    std::cout << "----------------------------------------------" << std::endl;
}

void Anime4KCPP::Anime4K::showFiltersInfo()
{
    switch (getProcessorType())
    {
    case ProcessorType::CPU:
    case ProcessorType::GPU:
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
        break;
    case ProcessorType::CPUCNN:
    case ProcessorType::GPUCNN:
        std::cout
            << "----------------------------------------------" << std::endl
            << "Filters does not support CNN mode" << std::endl
            << "----------------------------------------------" << std::endl;
        break;
    }
}

std::string Anime4KCPP::Anime4K::getInfo()
{
    std::ostringstream oss;
    ProcessorType type = getProcessorType();
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
    oss << "Processor type: " << type << std::endl;
    oss << "----------------------------------------------" << std::endl;
    switch (type)
    {
    case ProcessorType::CPU:
    case ProcessorType::GPU:
        oss << "Passes: " << ps << std::endl
            << "pushColorCount: " << pcc << std::endl
            << "Zoom Factor: " << zf << std::endl
            << "Video Mode: " << std::boolalpha << vm << std::endl
            << "Fast Mode: " << std::boolalpha << fm << std::endl
            << "Strength Color: " << sc << std::endl
            << "Strength Gradient: " << sg << std::endl;
        break;
    case ProcessorType::CPUCNN:
    case ProcessorType::GPUCNN:
        oss << "Zoom Factor: " << zf << std::endl
            << "HDN Mode: " << std::boolalpha << HDN << std::endl;
        if (HDN)
            oss
            << "HDN level: " << HDNLevel << std::endl;
        break;
    }
    oss << "----------------------------------------------" << std::endl;
    return std::string(oss.str());
}

std::string Anime4KCPP::Anime4K::getFiltersInfo()
{
    std::ostringstream oss;
    switch (getProcessorType())
    {
    case ProcessorType::CPU:
    case ProcessorType::GPU:
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
        break;
    case ProcessorType::CPUCNN:
    case ProcessorType::GPUCNN:
        oss
            << "----------------------------------------------" << std::endl
            << "Filters does not support CNN mode" << std::endl
            << "----------------------------------------------" << std::endl;
        break;
    }
    return std::string(oss.str());
}

size_t Anime4KCPP::Anime4K::getResultDataLength() noexcept
{
    if (!inputYUV)
        return dstImg.step * static_cast<size_t>(H);
    else
        return dstY.step * 3 * static_cast<size_t>(H);
}

size_t Anime4KCPP::Anime4K::getResultDataPerChannelLength() noexcept
{
    return static_cast<size_t>(W) * static_cast<size_t>(H);
}

std::array<int, 3> Anime4KCPP::Anime4K::getResultShape()
{
    std::array<int, 3> shape = { H, W, 3 };
    return shape;
}

void Anime4KCPP::Anime4K::showImage(bool R2B)
{
    cv::Mat tmpImg = dstImg;

    if (R2B)
        cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2RGB);

    if (inputYUV)
    {
        if (dstY.size() == dstU.size() && dstU.size() == dstV.size())
        {
            cv::merge(std::vector<cv::Mat>{ dstY, dstU, dstV }, tmpImg);
            cv::cvtColor(tmpImg, tmpImg, cv::COLOR_YUV2BGR);
        }
        else
            throw "Only YUV444 can be saved to file";
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

void Anime4KCPP::Anime4K::processWithPrintProgress()
{
    if (!vm)
    {
        process();
        return;
    }
    std::future<void> p = std::async(&Anime4K::process, this);
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

void Anime4KCPP::Anime4K::processWithProgress(const std::function<void(double)>&& callBack)
{
    if (!vm)
    {
        process();
        return;
    }
    std::future<void> p = std::async(&Anime4K::process, this);
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

void Anime4KCPP::Anime4K::stopVideoProcess() noexcept
{
    if (vm)
        videoIO->stopProcess();
}

void Anime4KCPP::Anime4K::pauseVideoProcess()
{
    if (vm && !videoIO->isPaused())
    {
        std::thread t(&VideoIO::pauseProcess, videoIO);
        t.detach();
    }
}

void Anime4KCPP::Anime4K::continueVideoProcess() noexcept
{
    if (vm)
        videoIO->continueProcess();
}

inline void Anime4KCPP::Anime4K::initVideoIO()
{
    if (videoIO == nullptr)
        videoIO = new VideoIO;
}

inline void Anime4KCPP::Anime4K::releaseVideoIO() noexcept
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
