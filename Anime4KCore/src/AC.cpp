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

    bitDepth = 8;
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
            cv::resize(alphaChannel, alphaChannel, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
            cv::cvtColor(orgImg, orgImg, cv::COLOR_BGRA2BGR);
            dstImg = orgImg;
            checkAlphaChannel = true;
            break;
        case 3:
            dstImg = orgImg;
            checkAlphaChannel = false;
            break;
        case 1:
            dstImg = orgImg;
            inputGrayscale = true;
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

    switch (orgImg.depth())
    {
    case CV_8U:
        break;
        bitDepth = 8;
    case CV_16U:
        bitDepth = 16;
        break;
    case CV_32F:
        bitDepth = 32;
        break;
    default:
        throw ACException<ExceptionType::RunTimeError>(
            "Unsupported data type");
        break;
    }

    inputRGB32 = false;
    inputYUV = false;
}

void Anime4KCPP::AC::loadImage(const cv::Mat& srcImage)
{
    orgImg = srcImage;
    if (orgImg.empty())
        throw ACException<ExceptionType::RunTimeError>("Empty image.");
    switch (orgImg.type())
    {
    case CV_8UC1:
        dstImg = orgImg;
        inputRGB32 = false;
        checkAlphaChannel = false;
        inputGrayscale = true;
        bitDepth = 8;
        break;
    case CV_8UC3:
        dstImg = orgImg;
        inputRGB32 = false;
        checkAlphaChannel = false;
        inputGrayscale = false;
        bitDepth = 8;
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
            cv::resize(alphaChannel, alphaChannel, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
            cv::cvtColor(orgImg, orgImg, cv::COLOR_BGRA2BGR);
            dstImg = orgImg;
        }
        inputGrayscale = false;
        bitDepth = 8;
        break;
    case CV_16UC1:
        dstImg = orgImg;
        inputRGB32 = false;
        checkAlphaChannel = false;
        inputGrayscale = true;
        bitDepth = 16;
        break;
    case CV_16UC3:
        dstImg = orgImg;
        inputRGB32 = false;
        checkAlphaChannel = false;
        inputGrayscale = false;
        bitDepth = 16;
        break;
    case CV_16UC4:
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
            cv::resize(alphaChannel, alphaChannel, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
            cv::cvtColor(orgImg, orgImg, cv::COLOR_BGRA2BGR);
            dstImg = orgImg;
        }
        inputGrayscale = false;
        bitDepth = 16;
        break;
    case CV_32FC1:
        dstImg = orgImg;
        inputRGB32 = false;
        checkAlphaChannel = false;
        inputGrayscale = true;
        bitDepth = 32;
        break;
    case CV_32FC3:
        dstImg = orgImg;
        inputRGB32 = false;
        checkAlphaChannel = false;
        inputGrayscale = false;
        bitDepth = 32;
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
            cv::resize(alphaChannel, alphaChannel, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
            cv::cvtColor(orgImg, orgImg, cv::COLOR_BGRA2BGR);
            dstImg = orgImg;
        }
        inputGrayscale = false;
        bitDepth = 32;
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

void Anime4KCPP::AC::loadImage(int rows, int cols, size_t stride, unsigned char* data, bool inputAsYUV444, bool inputAsRGB32, bool inputAsGrayscale)
{
    switch (inputAsRGB32 + inputAsYUV444)
    {
    case 0:
        if (inputAsGrayscale)
        {
            inputGrayscale = true;
            inputRGB32 = false;
            inputYUV = false;
            orgImg = cv::Mat(rows, cols, CV_8UC1, data, stride);
        }
        else
        {
            inputGrayscale = false;
            inputRGB32 = false;
            inputYUV = false;
            orgImg = cv::Mat(rows, cols, CV_8UC3, data, stride);
        }
        break;
    case 1:
        if (inputAsRGB32)
        {
            inputRGB32 = true;
            inputYUV = false;
            cv::cvtColor(cv::Mat(rows, cols, CV_8UC4, data, stride), orgImg, cv::COLOR_RGBA2RGB);
        }
        else //YUV444
        {
            inputRGB32 = false;
            inputYUV = true;
            orgImg = cv::Mat(rows, cols, CV_8UC3, data, stride);

            std::vector<cv::Mat> yuv(3);
            cv::split(orgImg, yuv);
            dstY = orgY = yuv[Y];
            dstU = orgU = yuv[U];
            dstV = orgV = yuv[V];
        }
        inputGrayscale = false;
        break;
    case 2:
        throw ACException<ExceptionType::IO>("Failed to load data: inputAsRGB32 and inputAsYUV444 can't be ture at same time.");
    }

    dstImg = orgImg;
    orgH = rows;
    orgW = cols;
    H = param.zoomFactor * orgH;
    W = param.zoomFactor * orgW;

    bitDepth = 8;
    checkAlphaChannel = false;
}

void Anime4KCPP::AC::loadImage(int rows, int cols, size_t stride, unsigned short int* data, bool inputAsYUV444, bool inputAsRGB32, bool inputAsGrayscale)
{
    switch (inputAsRGB32 + inputAsYUV444)
    {
    case 0:
        if (inputAsGrayscale)
        {
            inputGrayscale = true;
            inputRGB32 = false;
            inputYUV = false;
            orgImg = cv::Mat(rows, cols, CV_16UC1, data, stride);
        }
        else
        {
            inputGrayscale = false;
            inputRGB32 = false;
            inputYUV = false;
            orgImg = cv::Mat(rows, cols, CV_16UC3, data, stride);
        }
        break;
    case 1:
        if (inputAsRGB32)
        {
            inputRGB32 = true;
            inputYUV = false;
            cv::cvtColor(cv::Mat(rows, cols, CV_16UC4, data, stride), orgImg, cv::COLOR_RGBA2RGB);
        }
        else //YUV444
        {
            inputRGB32 = false;
            inputYUV = true;
            orgImg = cv::Mat(rows, cols, CV_16UC3, data, stride);

            std::vector<cv::Mat> yuv(3);
            cv::split(orgImg, yuv);
            dstY = orgY = yuv[Y];
            dstU = orgU = yuv[U];
            dstV = orgV = yuv[V];
        }
        inputGrayscale = false;
        break;
    case 2:
        throw ACException<ExceptionType::IO>("Failed to load data: inputAsRGB32 and inputAsYUV444 can't be ture at same time.");
    }

    dstImg = orgImg;
    orgH = rows;
    orgW = cols;
    H = param.zoomFactor * orgH;
    W = param.zoomFactor * orgW;

    bitDepth = 16;
    checkAlphaChannel = false;
}

void Anime4KCPP::AC::loadImage(int rows, int cols, size_t stride, float* data, bool inputAsYUV444, bool inputAsRGB32, bool inputAsGrayscale)
{
    switch (inputAsRGB32 + inputAsYUV444)
    {
    case 0:
        if (inputAsGrayscale)
        {
            inputGrayscale = true;
            inputRGB32 = false;
            inputYUV = false;
            orgImg = cv::Mat(rows, cols, CV_32FC1, data, stride);
        }
        else
        {
            inputGrayscale = false;
            inputRGB32 = false;
            inputYUV = false;
            orgImg = cv::Mat(rows, cols, CV_32FC3, data, stride);
        }
        break;
    case 1:
        if (inputAsRGB32)
        {
            inputRGB32 = true;
            inputYUV = false;
            cv::cvtColor(cv::Mat(rows, cols, CV_32FC4, data, stride), orgImg, cv::COLOR_RGBA2RGB);
        }
        else //YUV444
        {
            inputRGB32 = false;
            inputYUV = true;
            orgImg = cv::Mat(rows, cols, CV_32FC3, data, stride);

            std::vector<cv::Mat> yuv(3);
            cv::split(orgImg, yuv);
            dstY = orgY = yuv[Y];
            dstU = orgU = yuv[U];
            dstV = orgV = yuv[V];
        }
        inputGrayscale = false;
        break;
    case 2:
        throw ACException<ExceptionType::IO>("Failed to load data: inputAsRGB32 and inputAsYUV444 can't be ture at same time.");
    }

    dstImg = orgImg;
    orgH = rows;
    orgW = cols;
    H = param.zoomFactor * orgH;
    W = param.zoomFactor * orgW;

    bitDepth = 32;
    checkAlphaChannel = false;
}

void Anime4KCPP::AC::loadImage(int rows, int cols, size_t stride, unsigned char* r, unsigned char* g, unsigned char* b, bool inputAsYUV444)
{
    if (inputAsYUV444)
    {
        inputYUV = true;
        dstY = orgY = cv::Mat(rows, cols, CV_8UC1, r, stride);
        dstU = orgU = cv::Mat(rows, cols, CV_8UC1, g, stride);
        dstV = orgV = cv::Mat(rows, cols, CV_8UC1, b, stride);
    }
    else
    {
        inputYUV = false;
        cv::merge(std::vector<cv::Mat>{
                cv::Mat(rows, cols, CV_8UC1, b, stride),
                cv::Mat(rows, cols, CV_8UC1, g, stride),
                cv::Mat(rows, cols, CV_8UC1, r, stride)},
            orgImg);
        dstImg = orgImg;
    }
    orgH = rows;
    orgW = cols;
    H = param.zoomFactor * orgH;
    W = param.zoomFactor * orgW;

    bitDepth = 8;
    inputGrayscale = false;
    inputRGB32 = false;
    checkAlphaChannel = false;
}

void Anime4KCPP::AC::loadImage(int rows, int cols, size_t stride, unsigned short int* r, unsigned short int* g, unsigned short int* b, bool inputAsYUV444)
{
    if (inputAsYUV444)
    {
        inputYUV = true;
        dstY = orgY = cv::Mat(rows, cols, CV_16UC1, r, stride);
        dstU = orgU = cv::Mat(rows, cols, CV_16UC1, g, stride);
        dstV = orgV = cv::Mat(rows, cols, CV_16UC1, b, stride);
    }
    else
    {
        inputYUV = false;
        cv::merge(std::vector<cv::Mat>{
            cv::Mat(rows, cols, CV_16UC1, b, stride),
            cv::Mat(rows, cols, CV_16UC1, g, stride),
            cv::Mat(rows, cols, CV_16UC1, r, stride)},
            orgImg);
        dstImg = orgImg;
    }
    orgH = rows;
    orgW = cols;
    H = param.zoomFactor * orgH;
    W = param.zoomFactor * orgW;

    bitDepth = 16;
    inputGrayscale = false;
    inputRGB32 = false;
    checkAlphaChannel = false;
}

void Anime4KCPP::AC::loadImage(int rows, int cols, size_t stride, float* r, float* g, float* b, bool inputAsYUV444)
{
    if (inputAsYUV444)
    {
        inputYUV = true;
        dstY = orgY = cv::Mat(rows, cols, CV_32FC1, r, stride);
        dstU = orgU = cv::Mat(rows, cols, CV_32FC1, g, stride);
        dstV = orgV = cv::Mat(rows, cols, CV_32FC1, b, stride);
    }
    else
    {
        inputYUV = false;
        cv::merge(std::vector<cv::Mat>{
            cv::Mat(rows, cols, CV_32FC1, b, stride),
            cv::Mat(rows, cols, CV_32FC1, g, stride),
            cv::Mat(rows, cols, CV_32FC1, r, stride)},
            orgImg);
        dstImg = orgImg;
    }
    orgH = rows;
    orgW = cols;
    H = param.zoomFactor * orgH;
    W = param.zoomFactor * orgW;

    bitDepth = 32;
    inputGrayscale = false;
    inputRGB32 = false;
    checkAlphaChannel = false;
}

void Anime4KCPP::AC::loadImage(
    int rowsY, int colsY, size_t strideY, unsigned char* y,
    int rowsU, int colsU, size_t strideU, unsigned char* u,
    int rowsV, int colsV, size_t strideV, unsigned char* v) 
{
    dstY = orgY = cv::Mat(rowsY, colsY, CV_8UC1, y, strideY);
    dstU = orgU = cv::Mat(rowsU, colsU, CV_8UC1, u, strideU);
    dstV = orgV = cv::Mat(rowsV, colsV, CV_8UC1, v, strideV);
    orgH = rowsY;
    orgW = colsY;
    H = param.zoomFactor * orgH;
    W = param.zoomFactor * orgW;

    bitDepth = 8;
    inputGrayscale = false;
    inputYUV = true;
    inputRGB32 = false;
    checkAlphaChannel = false;
}

void Anime4KCPP::AC::loadImage(
    int rowsY, int colsY, size_t strideY, unsigned short int* y,
    int rowsU, int colsU, size_t strideU, unsigned short int* u,
    int rowsV, int colsV, size_t strideV, unsigned short int* v) 
{
    dstY = orgY = cv::Mat(rowsY, colsY, CV_16UC1, y, strideY);
    dstU = orgU = cv::Mat(rowsU, colsU, CV_16UC1, u, strideU);
    dstV = orgV = cv::Mat(rowsV, colsV, CV_16UC1, v, strideV);
    orgH = rowsY;
    orgW = colsY;
    H = param.zoomFactor * orgH;
    W = param.zoomFactor * orgW;

    bitDepth = 16;
    inputGrayscale = false;
    inputYUV = true;
    inputRGB32 = false;
    checkAlphaChannel = false;
}

void Anime4KCPP::AC::loadImage(
    int rowsY, int colsY, size_t strideY, float* y,
    int rowsU, int colsU, size_t strideU, float* u,
    int rowsV, int colsV, size_t strideV, float* v) 
{
    dstY = orgY = cv::Mat(rowsY, colsY, CV_32FC1, y, strideY);
    dstU = orgU = cv::Mat(rowsU, colsU, CV_32FC1, u, strideU);
    dstV = orgV = cv::Mat(rowsV, colsV, CV_32FC1, v, strideV);
    orgH = rowsY;
    orgW = colsY;
    H = param.zoomFactor * orgH;
    W = param.zoomFactor * orgW;

    bitDepth = 32;
    inputGrayscale = false;
    inputYUV = true;
    inputRGB32 = false;
    checkAlphaChannel = false;
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

    inputGrayscale = false;
    inputYUV = true;
    inputRGB32 = false;
    checkAlphaChannel = false;

    switch (y.depth())
    {
    case CV_8U:
        break;
        bitDepth = 8;
    case CV_16U:
        bitDepth = 16;
        break;
    case CV_32F:
        bitDepth = 32;
        break;
    default:
        throw ACException<ExceptionType::RunTimeError>(
            "Unsupported data type");
        break;
    }
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
            cv::resize(dstU, dstU, dstY.size(), 0.0, 0.0, cv::INTER_CUBIC);
        if (dstY.size() != dstV.size())
            cv::resize(dstV, dstV, dstY.size(), 0.0, 0.0, cv::INTER_CUBIC);
        cv::merge(std::vector<cv::Mat>{ dstY, dstU, dstV }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
    if (bitDepth == 32)
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

void Anime4KCPP::AC::saveImage(unsigned char* data, size_t dstStride)
{
    if (data == nullptr)
        throw ACException<ExceptionType::RunTimeError>("Pointer can not be nullptr");
    if (inputYUV)
    {
        if (dstY.size() == dstU.size() && dstU.size() == dstV.size())
            cv::merge(std::vector<cv::Mat>{ dstY, dstU, dstV }, dstImg);
        else
            throw ACException<ExceptionType::IO>("Only YUV444 can be saved to data pointer");
    }
    else if (inputRGB32)
        cv::cvtColor(dstImg, dstImg, cv::COLOR_RGB2RGBA);

    size_t stride = dstImg.step;
    size_t step = dstStride > stride ? dstStride : stride;
    if (stride == step)
    {
        memcpy(data, dstImg.data, stride * H);
    }
    else
    {
        for (size_t i = 0; i < H; i++)
        {
            memcpy(data, dstImg.data + i * stride, stride);
            data += step;
        }
    }
}

void Anime4KCPP::AC::saveImage(
    unsigned char* r, size_t dstStrideR,
    unsigned char* g, size_t dstStrideG,
    unsigned char* b, size_t dstStrideB)
{
    if (r == nullptr || g == nullptr || b == nullptr)
        throw ACException<ExceptionType::RunTimeError>("Pointers can not be nullptr");
    if (inputYUV)
    {
        size_t strideY = dstY.step;
        size_t strideU = dstU.step;
        size_t strideV = dstV.step;

        size_t stepY = dstStrideR > strideY ? dstStrideR : strideY;
        size_t stepU = dstStrideG > strideU ? dstStrideG : strideU;
        size_t stepV = dstStrideB > strideV ? dstStrideB : strideV;

        size_t HY = dstY.rows;
        size_t HUV = dstU.rows;

        if (strideY == stepY && strideU == stepU && strideV == stepV)
        {
            memcpy(r, dstY.data, strideY * HY);
            memcpy(g, dstU.data, strideU * HUV);
            memcpy(b, dstV.data, strideV * HUV);
        }
        else
        {
            for (size_t i = 0; i < HY; i++)
            {
                memcpy(r, dstY.data + i * strideY, strideY);
                r += stepY;

                if (i < HUV)
                {
                    memcpy(g, dstU.data + i * strideU, strideU);
                    memcpy(b, dstV.data + i * strideV, strideV);
                    b += stepV;
                    g += stepU;
                }
            }
        }
    }
    else
    {
        std::vector<cv::Mat> bgr(3);
        cv::split(dstImg, bgr);

        size_t stride = bgr[R].step;
        size_t step = dstStrideR > stride ? dstStrideR : stride;

        if (stride == step)
        {
            memcpy(b, bgr[B].data, stride * H);
            memcpy(g, bgr[G].data, stride * H);
            memcpy(r, bgr[R].data, stride * H);
        }
        else
        {
            for (size_t i = 0; i < H; i++)
            {
                memcpy(b, bgr[B].data + i * stride, stride);
                memcpy(g, bgr[G].data + i * stride, stride);
                memcpy(r, bgr[R].data + i * stride, stride);

                b += step;
                g += step;
                r += step;
            }
        }
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
        << "Parameter information" << std::endl
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
        << "Filter information" << std::endl
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
            cv::resize(dstU, tmpU, dstY.size(), 0.0, 0.0, cv::INTER_CUBIC);
        if (dstY.size() != dstV.size())
            cv::resize(dstV, tmpV, dstY.size(), 0.0, 0.0, cv::INTER_CUBIC);
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
    cv::destroyWindow("preview");
}

void Anime4KCPP::AC::process()
{
    switch (bitDepth)
    {
    case 8:
        if (!param.videoMode)
        {
            if (inputYUV)
                processYUVImageB();
            else if (inputGrayscale)
                processGrayscaleB();
            else
                processRGBImageB();
        }
        else
        {
            processRGBVideoB();
        }
        break;
    case 16:
        if (inputYUV)
            processYUVImageW();
        else if (inputGrayscale)
            processGrayscaleW();
        else
            processRGBImageW();
        break;
    case 32:
        if (inputYUV)
            processYUVImageF();
        else if (inputGrayscale)
            processGrayscaleF();
        else
            processRGBImageF();
        break;
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
