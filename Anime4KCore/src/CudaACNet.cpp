#ifdef ENABLE_CUDA

#define DLL

#include"CudaInterface.hpp"
#include"CudaACNet.hpp"

Anime4KCPP::Cuda::ACNet::ACNet(const Parameters& parameters) :
    AC(parameters) {}

std::string Anime4KCPP::Cuda::ACNet::getInfo()
{
    std::ostringstream oss;
    oss << AC::getInfo()
        << "----------------------------------------------" << std::endl
        << "CUDA Device ID: " << cuGetDeviceID() << std::endl
        << "Zoom Factor: " << param.zoomFactor << std::endl
        << "HDN Mode: " << std::boolalpha << param.HDN << std::endl
        << "HDN Level: " << (param.HDN ? param.HDNLevel : 0) << std::endl
        << "----------------------------------------------" << std::endl;
    return oss.str();
}

std::string Anime4KCPP::Cuda::ACNet::getFiltersInfo()
{
    std::ostringstream oss;
    oss << AC::getFiltersInfo()
        << "----------------------------------------------" << std::endl
        << "Filter not supported" << std::endl
        << "----------------------------------------------" << std::endl;
    return oss.str();
}

inline void Anime4KCPP::Cuda::ACNet::runKernelB(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    ACCudaParamACNet cuParam{ orgImg.cols, orgImg.rows, orgImg.step};
    if (param.HDN)
    {
        switch (param.HDNLevel)
        {
        case 1:
            cuRunKernelACNetHDN1B(orgImg.data, dstImg.data, &cuParam);
            break;
        case 2:
            cuRunKernelACNetHDN2B(orgImg.data, dstImg.data, &cuParam);
            break;
        case 3:
            cuRunKernelACNetHDN3B(orgImg.data, dstImg.data, &cuParam);
            break;
        default:
            cuRunKernelACNetHDN1B(orgImg.data, dstImg.data, &cuParam);
            break;
        }
    }
    else
    {
        cuRunKernelACNetHDN0B(orgImg.data, dstImg.data, &cuParam);
    }
}

inline void Anime4KCPP::Cuda::ACNet::runKernelW(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    ACCudaParamACNet cuParam{ orgImg.cols, orgImg.rows, orgImg.step };
    if (param.HDN)
    {
        switch (param.HDNLevel)
        {
        case 1:
            cuRunKernelACNetHDN1W(
                reinterpret_cast<const unsigned short*>(orgImg.data), 
                reinterpret_cast<unsigned short*>(dstImg.data), &cuParam);
            break;
        case 2:
            cuRunKernelACNetHDN2W(
                reinterpret_cast<const unsigned short*>(orgImg.data), 
                reinterpret_cast<unsigned short*>(dstImg.data), &cuParam);
            break;
        case 3:
            cuRunKernelACNetHDN3W(
                reinterpret_cast<const unsigned short*>(orgImg.data),
                reinterpret_cast<unsigned short*>(dstImg.data), &cuParam);
            break;
        default:
            cuRunKernelACNetHDN1W(
                reinterpret_cast<const unsigned short*>(orgImg.data),
                reinterpret_cast<unsigned short*>(dstImg.data), &cuParam);
            break;
        }
    }
    else
    {
        cuRunKernelACNetHDN0W(
            reinterpret_cast<const unsigned short*>(orgImg.data),
            reinterpret_cast<unsigned short*>(dstImg.data), &cuParam);
    }
}

inline void Anime4KCPP::Cuda::ACNet::runKernelF(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    ACCudaParamACNet cuParam{ orgImg.cols, orgImg.rows, orgImg.step };
    if (param.HDN)
    {
        switch (param.HDNLevel)
        {
        case 1:
            cuRunKernelACNetHDN1F(
                reinterpret_cast<const float*>(orgImg.data),
                reinterpret_cast<float*>(dstImg.data), &cuParam);
            break;
        case 2:
            cuRunKernelACNetHDN2F(
                reinterpret_cast<const float*>(orgImg.data),
                reinterpret_cast<float*>(dstImg.data), &cuParam);
            break;
        case 3:
            cuRunKernelACNetHDN3F(
                reinterpret_cast<const float*>(orgImg.data),
                reinterpret_cast<float*>(dstImg.data), &cuParam);
            break;
        default:
            cuRunKernelACNetHDN1F(
                reinterpret_cast<const float*>(orgImg.data),
                reinterpret_cast<float*>(dstImg.data), &cuParam);
            break;
        }
    }
    else
    {
        cuRunKernelACNetHDN0F(
            reinterpret_cast<const float*>(orgImg.data),
            reinterpret_cast<float*>(dstImg.data), &cuParam);
    }
}

void Anime4KCPP::Cuda::ACNet::processYUVImageB()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        cv::Mat tmpY = orgY;
        for (int i = 0; i < scaleTimes; i++)
        {
            dstY.create(tmpY.rows * 2, tmpY.cols * 2, CV_8UC1);
            runKernelB(tmpY, dstY);
            tmpY = dstY;
        }
        if (param.isNonIntegerScale())
        {
            cv::resize(dstY, dstY, cv::Size(W, H), param.zoomFactor, param.zoomFactor, cv::INTER_AREA);
        }

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgY, orgY, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgY, orgY, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        dstY.create(orgY.rows * 2, orgY.cols * 2, CV_8UC1);
        runKernelB(orgY, dstY);

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
    }
}

void Anime4KCPP::Cuda::ACNet::processRGBImageB()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        cv::Mat tmpImg;
        cv::cvtColor(orgImg, tmpImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(tmpImg, yuv);
        tmpImg = yuv[Y];

        for (int i = 0; i < scaleTimes; i++)
        {
            dstImg.create(tmpImg.rows * 2, tmpImg.cols * 2, CV_8UC1);
            runKernelB(tmpImg, dstImg);
            tmpImg = dstImg;
        }
        if (param.isNonIntegerScale())
        {
            cv::resize(dstImg, dstImg, cv::Size(W, H), param.zoomFactor, param.zoomFactor, cv::INTER_AREA);
        }

        cv::resize(yuv[U], yuv[U], cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(yuv[V], yuv[V], cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        cv::Mat tmpImg;
        cv::cvtColor(orgImg, tmpImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(tmpImg, yuv);

        dstImg.create(yuv[Y].rows * 2, yuv[Y].cols * 2, CV_8UC1);
        runKernelB(yuv[Y], dstImg);

        cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
        cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
}

void Anime4KCPP::Cuda::ACNet::processGrayscaleB()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        cv::Mat tmpImg = orgImg;
        for (int i = 0; i < scaleTimes; i++)
        {
            dstImg.create(tmpImg.rows * 2, tmpImg.cols * 2, CV_8UC1);
            runKernelB(tmpImg, dstImg);
            tmpImg = dstImg;
        }
        if (param.isNonIntegerScale())
        {
            cv::resize(dstImg, dstImg, cv::Size(W, H), param.zoomFactor, param.zoomFactor, cv::INTER_AREA);
        }
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        dstImg.create(orgImg.rows * 2, orgImg.cols * 2, CV_8UC1);
        runKernelB(orgImg, dstImg);
    }
}

void Anime4KCPP::Cuda::ACNet::processYUVImageW()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        cv::Mat tmpY = orgY;
        for (int i = 0; i < scaleTimes; i++)
        {
            dstY.create(tmpY.rows * 2, tmpY.cols * 2, CV_16UC1);
            runKernelW(tmpY, dstY);
            tmpY = dstY;
        }
        if (param.isNonIntegerScale())
        {
            cv::resize(dstY, dstY, cv::Size(W, H), param.zoomFactor, param.zoomFactor, cv::INTER_AREA);
        }

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgY, orgY, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgY, orgY, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        dstY.create(orgY.rows * 2, orgY.cols * 2, CV_16UC1);
        runKernelW(orgY, dstY);

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
    }
}

void Anime4KCPP::Cuda::ACNet::processRGBImageW()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        cv::Mat tmpImg;
        cv::cvtColor(orgImg, tmpImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(tmpImg, yuv);
        tmpImg = yuv[Y];

        for (int i = 0; i < scaleTimes; i++)
        {
            dstImg.create(tmpImg.rows * 2, tmpImg.cols * 2, CV_16UC1);
            runKernelW(tmpImg, dstImg);
            tmpImg = dstImg;
        }
        if (param.isNonIntegerScale())
        {
            cv::resize(dstImg, dstImg, cv::Size(W, H), param.zoomFactor, param.zoomFactor, cv::INTER_AREA);
        }

        cv::resize(yuv[U], yuv[U], cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(yuv[V], yuv[V], cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        cv::Mat tmpImg;
        cv::cvtColor(orgImg, tmpImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(tmpImg, yuv);

        dstImg.create(yuv[Y].rows * 2, yuv[Y].cols * 2, CV_16UC1);
        runKernelW(yuv[Y], dstImg);

        cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
        cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
}

void Anime4KCPP::Cuda::ACNet::processGrayscaleW()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        cv::Mat tmpImg = orgImg;
        for (int i = 0; i < scaleTimes; i++)
        {
            dstImg.create(tmpImg.rows * 2, tmpImg.cols * 2, CV_16UC1);
            runKernelW(tmpImg, dstImg);
            tmpImg = dstImg;
        }
        if (param.isNonIntegerScale())
        {
            cv::resize(dstImg, dstImg, cv::Size(W, H), param.zoomFactor, param.zoomFactor, cv::INTER_AREA);
        }
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        dstImg.create(orgImg.rows * 2, orgImg.cols * 2, CV_16UC1);
        runKernelW(orgImg, dstImg);
    }
}

void Anime4KCPP::Cuda::ACNet::processYUVImageF()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        cv::Mat tmpY = orgY;
        for (int i = 0; i < scaleTimes; i++)
        {
            dstY.create(tmpY.rows * 2, tmpY.cols * 2, CV_32FC1);
            runKernelF(tmpY, dstY);
            tmpY = dstY;
        }
        if (param.isNonIntegerScale())
        {
            cv::resize(dstY, dstY, cv::Size(W, H), param.zoomFactor, param.zoomFactor, cv::INTER_AREA);
        }

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgY, orgY, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgY, orgY, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        dstY.create(orgY.rows * 2, orgY.cols * 2, CV_32FC1);
        runKernelF(orgY, dstY);

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
    }
}

void Anime4KCPP::Cuda::ACNet::processRGBImageF()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        cv::Mat tmpImg;
        cv::cvtColor(orgImg, tmpImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(tmpImg, yuv);
        tmpImg = yuv[Y];

        for (int i = 0; i < scaleTimes; i++)
        {
            dstImg.create(tmpImg.rows * 2, tmpImg.cols * 2, CV_32FC1);
            runKernelF(tmpImg, dstImg);
            tmpImg = dstImg;
        }
        if (param.isNonIntegerScale())
        {
            cv::resize(dstImg, dstImg, cv::Size(W, H), param.zoomFactor, param.zoomFactor, cv::INTER_AREA);
        }

        cv::resize(yuv[U], yuv[U], cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(yuv[V], yuv[V], cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        cv::Mat tmpImg;
        cv::cvtColor(orgImg, tmpImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(tmpImg, yuv);

        dstImg.create(yuv[Y].rows * 2, yuv[Y].cols * 2, CV_32FC1);
        runKernelF(yuv[Y], dstImg);

        cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
        cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
}

void Anime4KCPP::Cuda::ACNet::processGrayscaleF()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        cv::Mat tmpImg = orgImg;
        for (int i = 0; i < scaleTimes; i++)
        {
            dstImg.create(tmpImg.rows * 2, tmpImg.cols * 2, CV_32FC1);
            runKernelF(tmpImg, dstImg);
            tmpImg = dstImg;
        }
        if (param.isNonIntegerScale())
        {
            cv::resize(dstImg, dstImg, cv::Size(W, H), param.zoomFactor, param.zoomFactor, cv::INTER_AREA);
        }
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        dstImg.create(orgImg.rows * 2, orgImg.cols * 2, CV_32FC1);
        runKernelF(orgImg, dstImg);
    }
}

Anime4KCPP::Processor::Type Anime4KCPP::Cuda::ACNet::getProcessorType() noexcept
{
    return Processor::Type::Cuda_ACNet;
}

std::string Anime4KCPP::Cuda::ACNet::getProcessorInfo()
{
    std::ostringstream oss;
    oss << "Processor type: " << getProcessorType() << std::endl
        << "Current CUDA devices:" << std::endl
        << cuGetDeviceInfo(cuGetDeviceID());
    return oss.str();
}

#endif
