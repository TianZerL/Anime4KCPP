#define DLL

#include "OpenCLACNet.hpp"
#include "OpenCLACNetKernel.hpp"

#define CLEAN_KERNEL_AND_THROW_ERROR(err, errCode) \
{\
clReleaseMemObject(imageBufferOrg); \
clReleaseMemObject(imageBufferTmp1); \
clReleaseMemObject(imageBufferTmp2); \
clReleaseMemObject(imageBufferDst); \
clReleaseKernel(kernelConv1To8L1); \
clReleaseKernel(kernelConv8To8L2); \
clReleaseKernel(kernelConv8To8L3); \
clReleaseKernel(kernelConv8To8L4); \
clReleaseKernel(kernelConv8To8L5); \
clReleaseKernel(kernelConv8To8L6); \
clReleaseKernel(kernelConv8To8L7); \
clReleaseKernel(kernelConv8To8L8); \
clReleaseKernel(kernelConv8To8L9); \
clReleaseKernel(kernelConvTranspose8To1L10); \
throw ACException<ExceptionType::GPU, true>(err, errCode); \
}


Anime4KCPP::OpenCL::ACNet::ACNet(const Parameters& parameters) :
    AC(parameters) 
{
    if (param.HDN)
    {
        switch (param.HDNLevel)
        {
        case 1:
            currACNetypeIndex = HDNL1;
            break;
        case 2:
            currACNetypeIndex = HDNL2;
            break;
        case 3:
            currACNetypeIndex = HDNL3;
            break;
        default:
            currACNetypeIndex = HDNL1;
            break;
        }
    }
    else
    {
        currACNetypeIndex = HDNL0;
    }
}

void Anime4KCPP::OpenCL::ACNet::setArguments(const Parameters& parameters)
{
    AC::setArguments(parameters);
    if (param.HDN)
    {
        switch (param.HDNLevel)
        {
        case 1:
            currACNetypeIndex = HDNL1;
            break;
        case 2:
            currACNetypeIndex = HDNL2;
            break;
        case 3:
            currACNetypeIndex = HDNL3;
            break;
        default:
            currACNetypeIndex = HDNL1;
            break;
        }
    }
    else
    {
        currACNetypeIndex = HDNL0;
    }
}

void Anime4KCPP::OpenCL::ACNet::initGPU(const int platformID, const int deviceID, const CNNType type, const int OpenCLQueueNum, const bool OpenCLParallelIO)
{
    if (!isInitialized)
    {
        pID = platformID;
        dID = deviceID;
        commandQueueNum = OpenCLQueueNum >= 1 ? OpenCLQueueNum : 1;
        parallelIO = OpenCLParallelIO;
        initOpenCL(type);
        isInitialized = true;
    }
}

void Anime4KCPP::OpenCL::ACNet::releaseGPU() noexcept
{
    if (isInitialized)
    {
        releaseOpenCL();
        context = nullptr;
        std::fill(commandQueueList.begin(), commandQueueList.end(), nullptr);
        commandQueueIO = nullptr;
        for (int i = HDNL0; i < TotalTypeCount; i++)
            program[i] = nullptr;
        device = nullptr;
        isInitialized = false;
    }
}

bool Anime4KCPP::OpenCL::ACNet::isInitializedGPU()
{
    return isInitialized;
}

std::string Anime4KCPP::OpenCL::ACNet::getInfo()
{
    std::ostringstream oss;
    oss << AC::getInfo()
        << "----------------------------------------------" << std::endl
        << "OpenCL Platform ID:" << pID << std::endl
        << "OpenCL Device ID:" << dID << std::endl
        << "Zoom Factor: " << param.zoomFactor << std::endl
        << "HDN Mode: " << std::boolalpha << param.HDN << std::endl
        << "HDN Level: " << (param.HDN ? param.HDNLevel : 0) << std::endl
        << "Number of OpenCL Command Queues:" << commandQueueNum << std::endl
        << "OpenCL Parallel IO Command Queues:" << std::boolalpha << parallelIO << std::endl
        << "----------------------------------------------" << std::endl;
    return oss.str();
}

std::string Anime4KCPP::OpenCL::ACNet::getFiltersInfo()
{
    std::ostringstream oss;
    oss << AC::getFiltersInfo()
        << "----------------------------------------------" << std::endl
        << "Filter not supported" << std::endl
        << "----------------------------------------------" << std::endl;
    return oss.str();
}

void Anime4KCPP::OpenCL::ACNet::processYUVImageB()
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
            if(parallelIO)
                runKernelPB(tmpY, dstY);
            else
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
        if (parallelIO)
            runKernelPB(orgY, dstY);
        else
            runKernelB(orgY, dstY);

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
    }
}

void Anime4KCPP::OpenCL::ACNet::processRGBImageB()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        cv::Mat tmpImg = orgImg;
        cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(tmpImg, yuv);
        tmpImg = yuv[Y];

        for (int i = 0; i < scaleTimes; i++)
        {
            dstImg.create(tmpImg.rows * 2, tmpImg.cols * 2, CV_8UC1);
            if (parallelIO)
                runKernelPB(tmpImg, dstImg);
            else
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

        cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(orgImg, yuv);
        orgImg = yuv[Y];

        dstImg.create(orgImg.rows * 2, orgImg.cols * 2, CV_8UC1);
        if (parallelIO)
            runKernelPB(orgImg, dstImg);
        else
            runKernelB(orgImg, dstImg);

        cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
        cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
}

void Anime4KCPP::OpenCL::ACNet::processGrayscaleB()
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
            if (parallelIO)
                runKernelPB(tmpImg, dstImg);
            else
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
        if (parallelIO)
            runKernelPB(orgImg, dstImg);
        else
            runKernelB(orgImg, dstImg);
    }
}

void Anime4KCPP::OpenCL::ACNet::processRGBVideoB()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        videoIO->init(
            [this, scaleTimes]()
            {
                Utils::Frame frame = videoIO->read();
                cv::Mat orgFrame = frame.first;
                cv::Mat dstFrame;

                cv::Mat tmpFrame = orgFrame;
                cv::cvtColor(tmpFrame, tmpFrame, cv::COLOR_BGR2YUV);

                std::vector<cv::Mat> yuv(3);
                cv::split(tmpFrame, yuv);
                tmpFrame = yuv[Y];

                for (int i = 0; i < scaleTimes; i++)
                {
                    dstFrame.create(tmpFrame.rows * 2, tmpFrame.cols * 2, CV_8UC1);
                    if (parallelIO)
                        runKernelPB(tmpFrame, dstFrame);
                    else
                        runKernelB(tmpFrame, dstFrame);
                    tmpFrame = dstFrame;
                }
                if (param.isNonIntegerScale())
                {
                    cv::resize(dstFrame, dstFrame, cv::Size(W, H), param.zoomFactor, param.zoomFactor, cv::INTER_AREA);
                }

                cv::resize(yuv[U], yuv[U], cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
                cv::resize(yuv[V], yuv[V], cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);

                cv::merge(std::vector<cv::Mat>{ dstFrame, yuv[U], yuv[V] }, dstFrame);
                cv::cvtColor(dstFrame, dstFrame, cv::COLOR_YUV2BGR);

                frame.first = dstFrame;
                videoIO->write(frame);
            }
            , param.maxThreads
                ).process();
    }
    else
    {
        videoIO->init(
            [this]()
            {
                Utils::Frame frame = videoIO->read();
                cv::Mat orgFrame = frame.first;
                cv::Mat dstFrame;

                if (param.zoomFactor > 2.0)
                    cv::resize(orgFrame, orgFrame, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
                else if (param.zoomFactor < 2.0)
                    cv::resize(orgFrame, orgFrame, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

                cv::cvtColor(orgFrame, orgFrame, cv::COLOR_BGR2YUV);

                std::vector<cv::Mat> yuv(3);
                cv::split(orgFrame, yuv);
                orgFrame = yuv[Y];

                dstFrame.create(orgFrame.rows * 2, orgFrame.cols * 2, CV_8UC1);
                if (parallelIO)
                    runKernelPB(orgFrame, dstFrame);
                else
                    runKernelB(orgFrame, dstFrame);

                cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
                cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);

                cv::merge(std::vector<cv::Mat>{ dstFrame, yuv[U], yuv[V] }, dstFrame);
                cv::cvtColor(dstFrame, dstFrame, cv::COLOR_YUV2BGR);

                frame.first = dstFrame;
                videoIO->write(frame);
            }
            , param.maxThreads
                ).process();
    }
}

void Anime4KCPP::OpenCL::ACNet::processYUVImageW()
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
            if (parallelIO)
                runKernelPW(tmpY, dstY);
            else
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
        if (parallelIO)
            runKernelPW(orgY, dstY);
        else
            runKernelW(orgY, dstY);

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
    }
}

void Anime4KCPP::OpenCL::ACNet::processRGBImageW()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        cv::Mat tmpImg = orgImg;
        cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(tmpImg, yuv);
        tmpImg = yuv[Y];

        for (int i = 0; i < scaleTimes; i++)
        {
            dstImg.create(tmpImg.rows * 2, tmpImg.cols * 2, CV_16UC1);
            if (parallelIO)
                runKernelPW(tmpImg, dstImg);
            else
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

        cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(orgImg, yuv);
        orgImg = yuv[Y];

        dstImg.create(orgImg.rows * 2, orgImg.cols * 2, CV_16UC1);
        if (parallelIO)
            runKernelPW(orgImg, dstImg);
        else
            runKernelW(orgImg, dstImg);

        cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
        cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
}

void Anime4KCPP::OpenCL::ACNet::processGrayscaleW()
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
            if (parallelIO)
                runKernelPW(tmpImg, dstImg);
            else
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
        if (parallelIO)
            runKernelPW(orgImg, dstImg);
        else
            runKernelW(orgImg, dstImg);
    }
}

void Anime4KCPP::OpenCL::ACNet::processYUVImageF()
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
            if (parallelIO)
                runKernelPF(tmpY, dstY);
            else
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
        if (parallelIO)
            runKernelPF(orgY, dstY);
        else
            runKernelF(orgY, dstY);

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
    }
}

void Anime4KCPP::OpenCL::ACNet::processRGBImageF()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        cv::Mat tmpImg = orgImg;
        cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(tmpImg, yuv);
        tmpImg = yuv[Y];

        for (int i = 0; i < scaleTimes; i++)
        {
            dstImg.create(tmpImg.rows * 2, tmpImg.cols * 2, CV_32FC1);
            if (parallelIO)
                runKernelPF(tmpImg, dstImg);
            else
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

        cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(orgImg, yuv);
        orgImg = yuv[Y];

        dstImg.create(orgImg.rows * 2, orgImg.cols * 2, CV_32FC1);
        if (parallelIO)
            runKernelPF(orgImg, dstImg);
        else
            runKernelF(orgImg, dstImg);

        cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
        cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
}

void Anime4KCPP::OpenCL::ACNet::processGrayscaleF()
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
            if (parallelIO)
                runKernelPF(tmpImg, dstImg);
            else
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
        if (parallelIO)
            runKernelPF(orgImg, dstImg);
        else
            runKernelF(orgImg, dstImg);
    }
}

void Anime4KCPP::OpenCL::ACNet::runKernelB(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    cl_int err = CL_SUCCESS;

    cl_image_format format{};
    cl_image_format tmpFormat{};

    cl_image_desc dstDesc{};
    cl_image_desc tmpDesc{};
    cl_image_desc orgDesc{};

    constexpr size_t orgin[3] = { 0,0,0 };
    const size_t orgRegion[3] = { static_cast<const size_t>(orgImg.cols),static_cast<const size_t>(orgImg.rows),1 };
    const size_t dstRegion[3] = { static_cast<const size_t>(dstImg.cols),static_cast<const size_t>(dstImg.rows),1 };

    const size_t orgSize[2] =
    {
        (((static_cast<const size_t>(orgImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(orgImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };
    const size_t dstSize[2] =
    {
        (((static_cast<const size_t>(dstImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(dstImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };

    //init frame
    format.image_channel_data_type = CL_UNORM_INT8;
    format.image_channel_order = CL_R;

    tmpFormat.image_channel_data_type = CL_HALF_FLOAT;
    tmpFormat.image_channel_order = CL_RGBA;

    orgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    orgDesc.image_height = orgImg.rows;
    orgDesc.image_width = orgImg.cols;
    orgDesc.buffer = nullptr;

    tmpDesc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    tmpDesc.image_height = orgImg.rows;
    tmpDesc.image_width = orgImg.cols;
    tmpDesc.image_array_size = 2;
    tmpDesc.buffer = nullptr;

    dstDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    dstDesc.image_height = dstImg.rows;
    dstDesc.image_width = dstImg.cols;
    dstDesc.buffer = nullptr;

    cl_command_queue commandQueue = commandQueueList[commandQueueCount++];
    if (commandQueueCount >= commandQueueNum)
        commandQueueCount = 0;

    cl_kernel kernelConv1To8L1 = clCreateKernel(program[currACNetypeIndex], "conv1To8", &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L1", err);
    }
    cl_kernel kernelConv8To8L2 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L2", err);
    }
    cl_kernel kernelConv8To8L3 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L3", err);
    }
    cl_kernel kernelConv8To8L4 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L4", err);
    }
    cl_kernel kernelConv8To8L5 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L5", err);
    }
    cl_kernel kernelConv8To8L6 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L6", err);
    }
    cl_kernel kernelConv8To8L7 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L7", err);
    }
    cl_kernel kernelConv8To8L8 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L8", err);
    }
    cl_kernel kernelConv8To8L9 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L9", err);
    }
    cl_kernel kernelConvTranspose8To1L10 = clCreateKernel(program[currACNetypeIndex], "convTranspose8To1", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        clReleaseKernel(kernelConv8To8L9);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L10", err);
    }


    cl_mem imageBufferOrg = clCreateImage(context, CL_MEM_READ_ONLY, &format, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Request imageBufferOrg error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp1 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp1 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp2 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp2 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferDst = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        clReleaseMemObject(imageBufferTmp2);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferDst error, video memory may be insufficient.", err);
    }

    //L1
    err = clSetKernelArg(kernelConv1To8L1, 0, sizeof(cl_mem), &imageBufferOrg);
    err |= clSetKernelArg(kernelConv1To8L1, 1, sizeof(cl_mem), &imageBufferTmp1);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L1 clSetKernelArg error", err)
    //L2
    err = clSetKernelArg(kernelConv8To8L2, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L2, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L2, 2, sizeof(cl_int), &L2);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L2 clSetKernelArg error", err)
    //L3
    err = clSetKernelArg(kernelConv8To8L3, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L3, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L3, 2, sizeof(cl_int), &L3);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L3 clSetKernelArg error", err)
    //L4
    err = clSetKernelArg(kernelConv8To8L4, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L4, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L4, 2, sizeof(cl_int), &L4);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L4 clSetKernelArg error", err)
    //L5
    err = clSetKernelArg(kernelConv8To8L5, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L5, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L5, 2, sizeof(cl_int), &L5);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L5 clSetKernelArg error", err)
    //L6
    err = clSetKernelArg(kernelConv8To8L6, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L6, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L6, 2, sizeof(cl_int), &L6);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L6 clSetKernelArg error", err)
    //L7
    err = clSetKernelArg(kernelConv8To8L7, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L7, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L7, 2, sizeof(cl_int), &L7);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L7 clSetKernelArg error", err)
    //L8
    err = clSetKernelArg(kernelConv8To8L8, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L8, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L8, 2, sizeof(cl_int), &L8);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L8 clSetKernelArg error", err)
    //L9
    err = clSetKernelArg(kernelConv8To8L9, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L9, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L9, 2, sizeof(cl_int), &L9);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L9 clSetKernelArg error", err)
    //L10
    err = clSetKernelArg(kernelConvTranspose8To1L10, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConvTranspose8To1L10, 1, sizeof(cl_mem), &imageBufferDst);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L10 clSetKernelArg error", err)

    clEnqueueWriteImage(commandQueue, imageBufferOrg, CL_FALSE, orgin, orgRegion, orgImg.step, 0, orgImg.data, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv1To8L1, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L2, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L3, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L4, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L5, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L6, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L7, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L8, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L9, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConvTranspose8To1L10, 2, nullptr, dstSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadImage(commandQueue, imageBufferDst, CL_TRUE, orgin, dstRegion, dstImg.step, 0, dstImg.data, 0, nullptr, nullptr);

    //clean
    clReleaseMemObject(imageBufferOrg);
    clReleaseMemObject(imageBufferTmp1);
    clReleaseMemObject(imageBufferTmp2);
    clReleaseMemObject(imageBufferDst);

    clReleaseKernel(kernelConv1To8L1);
    clReleaseKernel(kernelConv8To8L2);
    clReleaseKernel(kernelConv8To8L3);
    clReleaseKernel(kernelConv8To8L4);
    clReleaseKernel(kernelConv8To8L5);
    clReleaseKernel(kernelConv8To8L6);
    clReleaseKernel(kernelConv8To8L7);
    clReleaseKernel(kernelConv8To8L8);
    clReleaseKernel(kernelConv8To8L9);
    clReleaseKernel(kernelConvTranspose8To1L10);
}

void Anime4KCPP::OpenCL::ACNet::runKernelW(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    cl_int err = CL_SUCCESS;

    cl_image_format format{};
    cl_image_format tmpFormat{};

    cl_image_desc dstDesc{};
    cl_image_desc tmpDesc{};
    cl_image_desc orgDesc{};

    constexpr size_t orgin[3] = { 0,0,0 };
    const size_t orgRegion[3] = { static_cast<const size_t>(orgImg.cols),static_cast<const size_t>(orgImg.rows),1 };
    const size_t dstRegion[3] = { static_cast<const size_t>(dstImg.cols),static_cast<const size_t>(dstImg.rows),1 };

    const size_t orgSize[2] =
    {
        (((static_cast<const size_t>(orgImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(orgImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };
    const size_t dstSize[2] =
    {
        (((static_cast<const size_t>(dstImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(dstImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };

    //init frame
    format.image_channel_data_type = CL_UNORM_INT16;
    format.image_channel_order = CL_R;

    tmpFormat.image_channel_data_type = CL_HALF_FLOAT;
    tmpFormat.image_channel_order = CL_RGBA;

    orgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    orgDesc.image_height = orgImg.rows;
    orgDesc.image_width = orgImg.cols;
    orgDesc.buffer = nullptr;

    tmpDesc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    tmpDesc.image_height = orgImg.rows;
    tmpDesc.image_width = orgImg.cols;
    tmpDesc.image_array_size = 2;
    tmpDesc.buffer = nullptr;

    dstDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    dstDesc.image_height = dstImg.rows;
    dstDesc.image_width = dstImg.cols;
    dstDesc.buffer = nullptr;

    cl_command_queue commandQueue = commandQueueList[commandQueueCount++];
    if (commandQueueCount >= commandQueueNum)
        commandQueueCount = 0;

    cl_kernel kernelConv1To8L1 = clCreateKernel(program[currACNetypeIndex], "conv1To8", &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L1", err);
    }
    cl_kernel kernelConv8To8L2 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L2", err);
    }
    cl_kernel kernelConv8To8L3 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L3", err);
    }
    cl_kernel kernelConv8To8L4 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L4", err);
    }
    cl_kernel kernelConv8To8L5 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L5", err);
    }
    cl_kernel kernelConv8To8L6 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L6", err);
    }
    cl_kernel kernelConv8To8L7 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L7", err);
    }
    cl_kernel kernelConv8To8L8 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L8", err);
    }
    cl_kernel kernelConv8To8L9 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L9", err);
    }
    cl_kernel kernelConvTranspose8To1L10 = clCreateKernel(program[currACNetypeIndex], "convTranspose8To1", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        clReleaseKernel(kernelConv8To8L9);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L10", err);
    }


    cl_mem imageBufferOrg = clCreateImage(context, CL_MEM_READ_ONLY, &format, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Request imageBufferOrg error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp1 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp1 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp2 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp2 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferDst = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        clReleaseMemObject(imageBufferTmp2);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferDst error, video memory may be insufficient.", err);
    }

    //L1
    err = clSetKernelArg(kernelConv1To8L1, 0, sizeof(cl_mem), &imageBufferOrg);
    err |= clSetKernelArg(kernelConv1To8L1, 1, sizeof(cl_mem), &imageBufferTmp1);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L1 clSetKernelArg error", err)
    //L2
    err = clSetKernelArg(kernelConv8To8L2, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L2, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L2, 2, sizeof(cl_int), &L2);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L2 clSetKernelArg error", err)
    //L3
    err = clSetKernelArg(kernelConv8To8L3, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L3, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L3, 2, sizeof(cl_int), &L3);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L3 clSetKernelArg error", err)
    //L4
    err = clSetKernelArg(kernelConv8To8L4, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L4, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L4, 2, sizeof(cl_int), &L4);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L4 clSetKernelArg error", err)
    //L5
    err = clSetKernelArg(kernelConv8To8L5, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L5, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L5, 2, sizeof(cl_int), &L5);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L5 clSetKernelArg error", err)
    //L6
    err = clSetKernelArg(kernelConv8To8L6, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L6, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L6, 2, sizeof(cl_int), &L6);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L6 clSetKernelArg error", err)
    //L7
    err = clSetKernelArg(kernelConv8To8L7, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L7, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L7, 2, sizeof(cl_int), &L7);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L7 clSetKernelArg error", err)
    //L8
    err = clSetKernelArg(kernelConv8To8L8, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L8, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L8, 2, sizeof(cl_int), &L8);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L8 clSetKernelArg error", err)
    //L9
    err = clSetKernelArg(kernelConv8To8L9, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L9, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L9, 2, sizeof(cl_int), &L9);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L9 clSetKernelArg error", err)
    //L10
    err = clSetKernelArg(kernelConvTranspose8To1L10, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConvTranspose8To1L10, 1, sizeof(cl_mem), &imageBufferDst);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L10 clSetKernelArg error", err)

    clEnqueueWriteImage(commandQueue, imageBufferOrg, CL_FALSE, orgin, orgRegion, orgImg.step, 0, orgImg.data, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv1To8L1, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L2, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L3, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L4, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L5, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L6, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L7, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L8, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L9, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConvTranspose8To1L10, 2, nullptr, dstSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadImage(commandQueue, imageBufferDst, CL_TRUE, orgin, dstRegion, dstImg.step, 0, dstImg.data, 0, nullptr, nullptr);

    //clean
    clReleaseMemObject(imageBufferOrg);
    clReleaseMemObject(imageBufferTmp1);
    clReleaseMemObject(imageBufferTmp2);
    clReleaseMemObject(imageBufferDst);

    clReleaseKernel(kernelConv1To8L1);
    clReleaseKernel(kernelConv8To8L2);
    clReleaseKernel(kernelConv8To8L3);
    clReleaseKernel(kernelConv8To8L4);
    clReleaseKernel(kernelConv8To8L5);
    clReleaseKernel(kernelConv8To8L6);
    clReleaseKernel(kernelConv8To8L7);
    clReleaseKernel(kernelConv8To8L8);
    clReleaseKernel(kernelConv8To8L9);
    clReleaseKernel(kernelConvTranspose8To1L10);
}

void Anime4KCPP::OpenCL::ACNet::runKernelF(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    cl_int err = CL_SUCCESS;

    cl_image_format format{};
    cl_image_format tmpFormat{};

    cl_image_desc dstDesc{};
    cl_image_desc tmpDesc{};
    cl_image_desc orgDesc{};

    constexpr size_t orgin[3] = { 0,0,0 };
    const size_t orgRegion[3] = { static_cast<const size_t>(orgImg.cols),static_cast<const size_t>(orgImg.rows),1 };
    const size_t dstRegion[3] = { static_cast<const size_t>(dstImg.cols),static_cast<const size_t>(dstImg.rows),1 };

    const size_t orgSize[2] =
    {
        (((static_cast<const size_t>(orgImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(orgImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };
    const size_t dstSize[2] =
    {
        (((static_cast<const size_t>(dstImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(dstImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };

    //init frame
    format.image_channel_data_type = CL_FLOAT;
    format.image_channel_order = CL_R;

    tmpFormat.image_channel_data_type = CL_HALF_FLOAT;
    tmpFormat.image_channel_order = CL_RGBA;

    orgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    orgDesc.image_height = orgImg.rows;
    orgDesc.image_width = orgImg.cols;
    orgDesc.buffer = nullptr;

    tmpDesc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    tmpDesc.image_height = orgImg.rows;
    tmpDesc.image_width = orgImg.cols;
    tmpDesc.image_array_size = 2;
    tmpDesc.buffer = nullptr;

    dstDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    dstDesc.image_height = dstImg.rows;
    dstDesc.image_width = dstImg.cols;
    dstDesc.buffer = nullptr;

    cl_command_queue commandQueue = commandQueueList[commandQueueCount++];
    if (commandQueueCount >= commandQueueNum)
        commandQueueCount = 0;

    cl_kernel kernelConv1To8L1 = clCreateKernel(program[currACNetypeIndex], "conv1To8", &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L1", err);
    }
    cl_kernel kernelConv8To8L2 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L2", err);
    }
    cl_kernel kernelConv8To8L3 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L3", err);
    }
    cl_kernel kernelConv8To8L4 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L4", err);
    }
    cl_kernel kernelConv8To8L5 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L5", err);
    }
    cl_kernel kernelConv8To8L6 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L6", err);
    }
    cl_kernel kernelConv8To8L7 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L7", err);
    }
    cl_kernel kernelConv8To8L8 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L8", err);
    }
    cl_kernel kernelConv8To8L9 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L9", err);
    }
    cl_kernel kernelConvTranspose8To1L10 = clCreateKernel(program[currACNetypeIndex], "convTranspose8To1", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        clReleaseKernel(kernelConv8To8L9);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L10", err);
    }


    cl_mem imageBufferOrg = clCreateImage(context, CL_MEM_READ_ONLY, &format, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Request imageBufferOrg error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp1 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp1 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp2 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp2 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferDst = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        clReleaseMemObject(imageBufferTmp2);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferDst error, video memory may be insufficient.", err);
    }

    //L1
    err = clSetKernelArg(kernelConv1To8L1, 0, sizeof(cl_mem), &imageBufferOrg);
    err |= clSetKernelArg(kernelConv1To8L1, 1, sizeof(cl_mem), &imageBufferTmp1);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L1 clSetKernelArg error", err)
    //L2
    err = clSetKernelArg(kernelConv8To8L2, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L2, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L2, 2, sizeof(cl_int), &L2);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L2 clSetKernelArg error", err)
    //L3
    err = clSetKernelArg(kernelConv8To8L3, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L3, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L3, 2, sizeof(cl_int), &L3);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L3 clSetKernelArg error", err)
    //L4
    err = clSetKernelArg(kernelConv8To8L4, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L4, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L4, 2, sizeof(cl_int), &L4);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L4 clSetKernelArg error", err)
    //L5
    err = clSetKernelArg(kernelConv8To8L5, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L5, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L5, 2, sizeof(cl_int), &L5);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L5 clSetKernelArg error", err)
    //L6
    err = clSetKernelArg(kernelConv8To8L6, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L6, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L6, 2, sizeof(cl_int), &L6);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L6 clSetKernelArg error", err)
    //L7
    err = clSetKernelArg(kernelConv8To8L7, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L7, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L7, 2, sizeof(cl_int), &L7);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L7 clSetKernelArg error", err)
    //L8
    err = clSetKernelArg(kernelConv8To8L8, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L8, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L8, 2, sizeof(cl_int), &L8);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L8 clSetKernelArg error", err)
    //L9
    err = clSetKernelArg(kernelConv8To8L9, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L9, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L9, 2, sizeof(cl_int), &L9);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L9 clSetKernelArg error", err)
    //L10
    err = clSetKernelArg(kernelConvTranspose8To1L10, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConvTranspose8To1L10, 1, sizeof(cl_mem), &imageBufferDst);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L10 clSetKernelArg error", err)

    clEnqueueWriteImage(commandQueue, imageBufferOrg, CL_FALSE, orgin, orgRegion, orgImg.step, 0, orgImg.data, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv1To8L1, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L2, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L3, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L4, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L5, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L6, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L7, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L8, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L9, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConvTranspose8To1L10, 2, nullptr, dstSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadImage(commandQueue, imageBufferDst, CL_TRUE, orgin, dstRegion, dstImg.step, 0, dstImg.data, 0, nullptr, nullptr);

    //clean
    clReleaseMemObject(imageBufferOrg);
    clReleaseMemObject(imageBufferTmp1);
    clReleaseMemObject(imageBufferTmp2);
    clReleaseMemObject(imageBufferDst);

    clReleaseKernel(kernelConv1To8L1);
    clReleaseKernel(kernelConv8To8L2);
    clReleaseKernel(kernelConv8To8L3);
    clReleaseKernel(kernelConv8To8L4);
    clReleaseKernel(kernelConv8To8L5);
    clReleaseKernel(kernelConv8To8L6);
    clReleaseKernel(kernelConv8To8L7);
    clReleaseKernel(kernelConv8To8L8);
    clReleaseKernel(kernelConv8To8L9);
    clReleaseKernel(kernelConvTranspose8To1L10);
}

void Anime4KCPP::OpenCL::ACNet::runKernelPB(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    cl_int err = CL_SUCCESS;

    cl_event writeFinishedEvent = nullptr;
    cl_event readReadyEvent = nullptr;
    cl_event readFinishedEvent = nullptr;

    cl_image_format format{};
    cl_image_format tmpFormat{};

    cl_image_desc dstDesc{};
    cl_image_desc tmpDesc{};
    cl_image_desc orgDesc{};

    constexpr size_t orgin[3] = { 0,0,0 };
    const size_t orgRegion[3] = { static_cast<const size_t>(orgImg.cols),static_cast<const size_t>(orgImg.rows),1 };
    const size_t dstRegion[3] = { static_cast<const size_t>(dstImg.cols),static_cast<const size_t>(dstImg.rows),1 };

    const size_t orgSize[2] =
    {
        (((static_cast<const size_t>(orgImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(orgImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };
    const size_t dstSize[2] =
    {
        (((static_cast<const size_t>(dstImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(dstImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };

    //init frame
    format.image_channel_data_type = CL_UNORM_INT8;
    format.image_channel_order = CL_R;

    tmpFormat.image_channel_data_type = CL_HALF_FLOAT;
    tmpFormat.image_channel_order = CL_RGBA;

    orgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    orgDesc.image_height = orgImg.rows;
    orgDesc.image_width = orgImg.cols;
    orgDesc.buffer = nullptr;

    tmpDesc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    tmpDesc.image_height = orgImg.rows;
    tmpDesc.image_width = orgImg.cols;
    tmpDesc.image_array_size = 2;
    tmpDesc.buffer = nullptr;

    dstDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    dstDesc.image_height = dstImg.rows;
    dstDesc.image_width = dstImg.cols;
    dstDesc.buffer = nullptr;

    cl_command_queue commandQueue = commandQueueList[commandQueueCount++];
    if (commandQueueCount >= commandQueueNum)
        commandQueueCount = 0;

    cl_kernel kernelConv1To8L1 = clCreateKernel(program[currACNetypeIndex], "conv1To8", &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L1", err);
    }
    cl_kernel kernelConv8To8L2 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L2", err);
    }
    cl_kernel kernelConv8To8L3 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L3", err);
    }
    cl_kernel kernelConv8To8L4 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L4", err);
    }
    cl_kernel kernelConv8To8L5 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L5", err);
    }
    cl_kernel kernelConv8To8L6 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L6", err);
    }
    cl_kernel kernelConv8To8L7 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L7", err);
    }
    cl_kernel kernelConv8To8L8 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L8", err);
    }
    cl_kernel kernelConv8To8L9 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L9", err);
    }
    cl_kernel kernelConvTranspose8To1L10 = clCreateKernel(program[currACNetypeIndex], "convTranspose8To1", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        clReleaseKernel(kernelConv8To8L9);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L10", err);
    }


    cl_mem imageBufferOrg = clCreateImage(context, CL_MEM_READ_ONLY, &format, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Request imageBufferOrg error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp1 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp1 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp2 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp2 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferDst = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        clReleaseMemObject(imageBufferTmp2);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferDst error, video memory may be insufficient.", err);
    }

    //L1
    err = clSetKernelArg(kernelConv1To8L1, 0, sizeof(cl_mem), &imageBufferOrg);
    err |= clSetKernelArg(kernelConv1To8L1, 1, sizeof(cl_mem), &imageBufferTmp1);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L1 clSetKernelArg error", err)
        //L2
    err = clSetKernelArg(kernelConv8To8L2, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L2, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L2, 2, sizeof(cl_int), &L2);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L2 clSetKernelArg error", err)
        //L3
    err = clSetKernelArg(kernelConv8To8L3, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L3, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L3, 2, sizeof(cl_int), &L3);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L3 clSetKernelArg error", err)
        //L4
    err = clSetKernelArg(kernelConv8To8L4, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L4, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L4, 2, sizeof(cl_int), &L4);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L4 clSetKernelArg error", err)
        //L5
    err = clSetKernelArg(kernelConv8To8L5, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L5, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L5, 2, sizeof(cl_int), &L5);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L5 clSetKernelArg error", err)
        //L6
    err = clSetKernelArg(kernelConv8To8L6, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L6, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L6, 2, sizeof(cl_int), &L6);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L6 clSetKernelArg error", err)
        //L7
    err = clSetKernelArg(kernelConv8To8L7, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L7, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L7, 2, sizeof(cl_int), &L7);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L7 clSetKernelArg error", err)
        //L8
    err = clSetKernelArg(kernelConv8To8L8, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L8, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L8, 2, sizeof(cl_int), &L8);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L8 clSetKernelArg error", err)
        //L9
    err = clSetKernelArg(kernelConv8To8L9, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L9, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L9, 2, sizeof(cl_int), &L9);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L9 clSetKernelArg error", err)
        //L10
    err = clSetKernelArg(kernelConvTranspose8To1L10, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConvTranspose8To1L10, 1, sizeof(cl_mem), &imageBufferDst);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L10 clSetKernelArg error", err)

    clEnqueueWriteImage(commandQueueIO, imageBufferOrg, CL_FALSE, orgin, orgRegion, orgImg.step, 0, orgImg.data, 0, nullptr, &writeFinishedEvent);
    clEnqueueNDRangeKernel(commandQueue, kernelConv1To8L1, 2, nullptr, orgSize, nullptr, 1, &writeFinishedEvent, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L2, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L3, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L4, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L5, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L6, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L7, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L8, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L9, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConvTranspose8To1L10, 2, nullptr, dstSize, nullptr, 0, nullptr, &readReadyEvent);
    clEnqueueReadImage(commandQueueIO, imageBufferDst, CL_FALSE, orgin, dstRegion, dstImg.step, 0, dstImg.data, 1, &readReadyEvent, &readFinishedEvent);

    clWaitForEvents(1, &readFinishedEvent);

    //clean
    clReleaseMemObject(imageBufferOrg);
    clReleaseMemObject(imageBufferTmp1);
    clReleaseMemObject(imageBufferTmp2);
    clReleaseMemObject(imageBufferDst);

    clReleaseKernel(kernelConv1To8L1);
    clReleaseKernel(kernelConv8To8L2);
    clReleaseKernel(kernelConv8To8L3);
    clReleaseKernel(kernelConv8To8L4);
    clReleaseKernel(kernelConv8To8L5);
    clReleaseKernel(kernelConv8To8L6);
    clReleaseKernel(kernelConv8To8L7);
    clReleaseKernel(kernelConv8To8L8);
    clReleaseKernel(kernelConv8To8L9);
    clReleaseKernel(kernelConvTranspose8To1L10);

    clReleaseEvent(writeFinishedEvent);
    clReleaseEvent(readReadyEvent);
    clReleaseEvent(readFinishedEvent);
}

void Anime4KCPP::OpenCL::ACNet::runKernelPW(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    cl_int err = CL_SUCCESS;

    cl_event writeFinishedEvent = nullptr;
    cl_event readReadyEvent = nullptr;
    cl_event readFinishedEvent = nullptr;

    cl_image_format format{};
    cl_image_format tmpFormat{};

    cl_image_desc dstDesc{};
    cl_image_desc tmpDesc{};
    cl_image_desc orgDesc{};

    constexpr size_t orgin[3] = { 0,0,0 };
    const size_t orgRegion[3] = { static_cast<const size_t>(orgImg.cols),static_cast<const size_t>(orgImg.rows),1 };
    const size_t dstRegion[3] = { static_cast<const size_t>(dstImg.cols),static_cast<const size_t>(dstImg.rows),1 };

    const size_t orgSize[2] =
    {
        (((static_cast<const size_t>(orgImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(orgImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };
    const size_t dstSize[2] =
    {
        (((static_cast<const size_t>(dstImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(dstImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };

    //init frame
    format.image_channel_data_type = CL_UNORM_INT16;
    format.image_channel_order = CL_R;

    tmpFormat.image_channel_data_type = CL_HALF_FLOAT;
    tmpFormat.image_channel_order = CL_RGBA;

    orgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    orgDesc.image_height = orgImg.rows;
    orgDesc.image_width = orgImg.cols;
    orgDesc.buffer = nullptr;

    tmpDesc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    tmpDesc.image_height = orgImg.rows;
    tmpDesc.image_width = orgImg.cols;
    tmpDesc.image_array_size = 2;
    tmpDesc.buffer = nullptr;

    dstDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    dstDesc.image_height = dstImg.rows;
    dstDesc.image_width = dstImg.cols;
    dstDesc.buffer = nullptr;

    cl_command_queue commandQueue = commandQueueList[commandQueueCount++];
    if (commandQueueCount >= commandQueueNum)
        commandQueueCount = 0;

    cl_kernel kernelConv1To8L1 = clCreateKernel(program[currACNetypeIndex], "conv1To8", &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L1", err);
    }
    cl_kernel kernelConv8To8L2 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L2", err);
    }
    cl_kernel kernelConv8To8L3 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L3", err);
    }
    cl_kernel kernelConv8To8L4 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L4", err);
    }
    cl_kernel kernelConv8To8L5 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L5", err);
    }
    cl_kernel kernelConv8To8L6 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L6", err);
    }
    cl_kernel kernelConv8To8L7 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L7", err);
    }
    cl_kernel kernelConv8To8L8 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L8", err);
    }
    cl_kernel kernelConv8To8L9 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L9", err);
    }
    cl_kernel kernelConvTranspose8To1L10 = clCreateKernel(program[currACNetypeIndex], "convTranspose8To1", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        clReleaseKernel(kernelConv8To8L9);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L10", err);
    }


    cl_mem imageBufferOrg = clCreateImage(context, CL_MEM_READ_ONLY, &format, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Request imageBufferOrg error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp1 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp1 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp2 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp2 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferDst = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        clReleaseMemObject(imageBufferTmp2);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferDst error, video memory may be insufficient.", err);
    }

    //L1
    err = clSetKernelArg(kernelConv1To8L1, 0, sizeof(cl_mem), &imageBufferOrg);
    err |= clSetKernelArg(kernelConv1To8L1, 1, sizeof(cl_mem), &imageBufferTmp1);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L1 clSetKernelArg error", err)
    //L2
    err = clSetKernelArg(kernelConv8To8L2, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L2, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L2, 2, sizeof(cl_int), &L2);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L2 clSetKernelArg error", err)
    //L3
    err = clSetKernelArg(kernelConv8To8L3, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L3, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L3, 2, sizeof(cl_int), &L3);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L3 clSetKernelArg error", err)
    //L4
    err = clSetKernelArg(kernelConv8To8L4, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L4, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L4, 2, sizeof(cl_int), &L4);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L4 clSetKernelArg error", err)
    //L5
    err = clSetKernelArg(kernelConv8To8L5, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L5, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L5, 2, sizeof(cl_int), &L5);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L5 clSetKernelArg error", err)
    //L6
    err = clSetKernelArg(kernelConv8To8L6, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L6, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L6, 2, sizeof(cl_int), &L6);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L6 clSetKernelArg error", err)
    //L7
    err = clSetKernelArg(kernelConv8To8L7, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L7, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L7, 2, sizeof(cl_int), &L7);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L7 clSetKernelArg error", err)
    //L8
    err = clSetKernelArg(kernelConv8To8L8, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L8, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L8, 2, sizeof(cl_int), &L8);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L8 clSetKernelArg error", err)
    //L9
    err = clSetKernelArg(kernelConv8To8L9, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L9, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L9, 2, sizeof(cl_int), &L9);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L9 clSetKernelArg error", err)
    //L10
    err = clSetKernelArg(kernelConvTranspose8To1L10, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConvTranspose8To1L10, 1, sizeof(cl_mem), &imageBufferDst);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L10 clSetKernelArg error", err)

    clEnqueueWriteImage(commandQueueIO, imageBufferOrg, CL_FALSE, orgin, orgRegion, orgImg.step, 0, orgImg.data, 0, nullptr, &writeFinishedEvent);
    clEnqueueNDRangeKernel(commandQueue, kernelConv1To8L1, 2, nullptr, orgSize, nullptr, 1, &writeFinishedEvent, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L2, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L3, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L4, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L5, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L6, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L7, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L8, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L9, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConvTranspose8To1L10, 2, nullptr, dstSize, nullptr, 0, nullptr, &readReadyEvent);
    clEnqueueReadImage(commandQueueIO, imageBufferDst, CL_FALSE, orgin, dstRegion, dstImg.step, 0, dstImg.data, 1, &readReadyEvent, &readFinishedEvent);

    clWaitForEvents(1, &readFinishedEvent);

    //clean
    clReleaseMemObject(imageBufferOrg);
    clReleaseMemObject(imageBufferTmp1);
    clReleaseMemObject(imageBufferTmp2);
    clReleaseMemObject(imageBufferDst);

    clReleaseKernel(kernelConv1To8L1);
    clReleaseKernel(kernelConv8To8L2);
    clReleaseKernel(kernelConv8To8L3);
    clReleaseKernel(kernelConv8To8L4);
    clReleaseKernel(kernelConv8To8L5);
    clReleaseKernel(kernelConv8To8L6);
    clReleaseKernel(kernelConv8To8L7);
    clReleaseKernel(kernelConv8To8L8);
    clReleaseKernel(kernelConv8To8L9);
    clReleaseKernel(kernelConvTranspose8To1L10);

    clReleaseEvent(writeFinishedEvent);
    clReleaseEvent(readReadyEvent);
    clReleaseEvent(readFinishedEvent);
}

void Anime4KCPP::OpenCL::ACNet::runKernelPF(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    cl_int err = CL_SUCCESS;

    cl_event writeFinishedEvent = nullptr;
    cl_event readReadyEvent = nullptr;
    cl_event readFinishedEvent = nullptr;

    cl_image_format format{};
    cl_image_format tmpFormat{};

    cl_image_desc dstDesc{};
    cl_image_desc tmpDesc{};
    cl_image_desc orgDesc{};

    constexpr size_t orgin[3] = { 0,0,0 };
    const size_t orgRegion[3] = { static_cast<const size_t>(orgImg.cols),static_cast<const size_t>(orgImg.rows),1 };
    const size_t dstRegion[3] = { static_cast<const size_t>(dstImg.cols),static_cast<const size_t>(dstImg.rows),1 };

    const size_t orgSize[2] =
    {
        (((static_cast<const size_t>(orgImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(orgImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };
    const size_t dstSize[2] =
    {
        (((static_cast<const size_t>(dstImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(dstImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };

    //init frame
    format.image_channel_data_type = CL_FLOAT;
    format.image_channel_order = CL_R;

    tmpFormat.image_channel_data_type = CL_HALF_FLOAT;
    tmpFormat.image_channel_order = CL_RGBA;

    orgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    orgDesc.image_height = orgImg.rows;
    orgDesc.image_width = orgImg.cols;
    orgDesc.buffer = nullptr;

    tmpDesc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    tmpDesc.image_height = orgImg.rows;
    tmpDesc.image_width = orgImg.cols;
    tmpDesc.image_array_size = 2;
    tmpDesc.buffer = nullptr;

    dstDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    dstDesc.image_height = dstImg.rows;
    dstDesc.image_width = dstImg.cols;
    dstDesc.buffer = nullptr;

    cl_command_queue commandQueue = commandQueueList[commandQueueCount++];
    if (commandQueueCount >= commandQueueNum)
        commandQueueCount = 0;

    cl_kernel kernelConv1To8L1 = clCreateKernel(program[currACNetypeIndex], "conv1To8", &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L1", err);
    }
    cl_kernel kernelConv8To8L2 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L2", err);
    }
    cl_kernel kernelConv8To8L3 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L3", err);
    }
    cl_kernel kernelConv8To8L4 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L4", err);
    }
    cl_kernel kernelConv8To8L5 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L5", err);
    }
    cl_kernel kernelConv8To8L6 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L6", err);
    }
    cl_kernel kernelConv8To8L7 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L7", err);
    }
    cl_kernel kernelConv8To8L8 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L8", err);
    }
    cl_kernel kernelConv8To8L9 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L9", err);
    }
    cl_kernel kernelConvTranspose8To1L10 = clCreateKernel(program[currACNetypeIndex], "convTranspose8To1", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        clReleaseKernel(kernelConv8To8L9);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L10", err);
    }


    cl_mem imageBufferOrg = clCreateImage(context, CL_MEM_READ_ONLY, &format, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Request imageBufferOrg error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp1 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp1 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp2 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp2 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferDst = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        clReleaseMemObject(imageBufferTmp2);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferDst error, video memory may be insufficient.", err);
    }

    //L1
    err = clSetKernelArg(kernelConv1To8L1, 0, sizeof(cl_mem), &imageBufferOrg);
    err |= clSetKernelArg(kernelConv1To8L1, 1, sizeof(cl_mem), &imageBufferTmp1);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L1 clSetKernelArg error", err)
        //L2
    err = clSetKernelArg(kernelConv8To8L2, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L2, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L2, 2, sizeof(cl_int), &L2);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L2 clSetKernelArg error", err)
        //L3
    err = clSetKernelArg(kernelConv8To8L3, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L3, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L3, 2, sizeof(cl_int), &L3);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L3 clSetKernelArg error", err)
        //L4
    err = clSetKernelArg(kernelConv8To8L4, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L4, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L4, 2, sizeof(cl_int), &L4);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L4 clSetKernelArg error", err)
        //L5
    err = clSetKernelArg(kernelConv8To8L5, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L5, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L5, 2, sizeof(cl_int), &L5);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L5 clSetKernelArg error", err)
        //L6
    err = clSetKernelArg(kernelConv8To8L6, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L6, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L6, 2, sizeof(cl_int), &L6);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L6 clSetKernelArg error", err)
        //L7
    err = clSetKernelArg(kernelConv8To8L7, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L7, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L7, 2, sizeof(cl_int), &L7);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L7 clSetKernelArg error", err)
        //L8
    err = clSetKernelArg(kernelConv8To8L8, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L8, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L8, 2, sizeof(cl_int), &L8);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L8 clSetKernelArg error", err)
        //L9
    err = clSetKernelArg(kernelConv8To8L9, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L9, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L9, 2, sizeof(cl_int), &L9);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L9 clSetKernelArg error", err)
        //L10
    err = clSetKernelArg(kernelConvTranspose8To1L10, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConvTranspose8To1L10, 1, sizeof(cl_mem), &imageBufferDst);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L10 clSetKernelArg error", err)

    clEnqueueWriteImage(commandQueueIO, imageBufferOrg, CL_FALSE, orgin, orgRegion, orgImg.step, 0, orgImg.data, 0, nullptr, &writeFinishedEvent);
    clEnqueueNDRangeKernel(commandQueue, kernelConv1To8L1, 2, nullptr, orgSize, nullptr, 1, &writeFinishedEvent, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L2, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L3, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L4, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L5, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L6, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L7, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L8, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L9, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConvTranspose8To1L10, 2, nullptr, dstSize, nullptr, 0, nullptr, &readReadyEvent);
    clEnqueueReadImage(commandQueueIO, imageBufferDst, CL_FALSE, orgin, dstRegion, dstImg.step, 0, dstImg.data, 1, &readReadyEvent, &readFinishedEvent);

    clWaitForEvents(1, &readFinishedEvent);

    //clean
    clReleaseMemObject(imageBufferOrg);
    clReleaseMemObject(imageBufferTmp1);
    clReleaseMemObject(imageBufferTmp2);
    clReleaseMemObject(imageBufferDst);

    clReleaseKernel(kernelConv1To8L1);
    clReleaseKernel(kernelConv8To8L2);
    clReleaseKernel(kernelConv8To8L3);
    clReleaseKernel(kernelConv8To8L4);
    clReleaseKernel(kernelConv8To8L5);
    clReleaseKernel(kernelConv8To8L6);
    clReleaseKernel(kernelConv8To8L7);
    clReleaseKernel(kernelConv8To8L8);
    clReleaseKernel(kernelConv8To8L9);
    clReleaseKernel(kernelConvTranspose8To1L10);

    clReleaseEvent(writeFinishedEvent);
    clReleaseEvent(readReadyEvent);
    clReleaseEvent(readFinishedEvent);
}

void Anime4KCPP::OpenCL::ACNet::initOpenCL(const CNNType type)
{
    cl_int err = CL_SUCCESS;
    cl_uint platforms = 0;
    cl_uint devices = 0;
    cl_platform_id currentplatform = nullptr;

    //init platform
    err = clGetPlatformIDs(0, nullptr, &platforms);
    if (err != CL_SUCCESS || !platforms)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to find OpenCL platform", err);
    }

    cl_platform_id* tmpPlatform = new cl_platform_id[platforms];
    err = clGetPlatformIDs(platforms, tmpPlatform, nullptr);
    if (err != CL_SUCCESS)
    {
        delete[] tmpPlatform;
        throw ACException<ExceptionType::GPU, true>("Failed to get OpenCL platform", err);
    }


    if (pID >= 0 && pID < (int)platforms)
        currentplatform = tmpPlatform[pID];
    else
        currentplatform = tmpPlatform[pID = 0];

    delete[] tmpPlatform;

    //init device
    err = clGetDeviceIDs(currentplatform, CL_DEVICE_TYPE_GPU, 0, nullptr, &devices);
    if (err != CL_SUCCESS || !devices)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to find supported GPU", err);
    }

    cl_device_id* tmpDevice = new cl_device_id[devices];
    err = clGetDeviceIDs(currentplatform, CL_DEVICE_TYPE_GPU, devices, tmpDevice, nullptr);
    if (err != CL_SUCCESS)
    {
        delete[] tmpDevice;
        throw ACException<ExceptionType::GPU, true>("GPU initialization error", err);
    }

    if (dID >= 0 && dID < (int)devices)
        device = tmpDevice[dID];
    else
        device = tmpDevice[dID = 0];

    delete[] tmpDevice;

    //init context
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        releaseOpenCL();
        throw ACException<ExceptionType::GPU, true>("Failed to create context", err);
    }

    //init command queue
    commandQueueList.resize(commandQueueNum, nullptr);
#ifndef CL_VERSION_2_0 //for OpenCL SDK older than v2.0 to build
    for (int i = 0; i < commandQueueNum; i++)
    {
        commandQueueList[i] = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS)
        {
            releaseOpenCL();
            throw ACException<ExceptionType::GPU, true>("Failed to create command queue", err);
        }
    }
    if (parallelIO)
    {
        commandQueueIO = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS)
        {
            releaseOpenCL();
            throw ACException<ExceptionType::GPU, true>("Failed to create command queue", err);
        }
    }

#else
    for (int i = 0; i < commandQueueNum; i++)
    {
#ifdef LEGACY_OPENCL_API
        commandQueueList[i] = clCreateCommandQueue(context, device, 0, &err);
#else
        commandQueueList[i] = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
#endif
        if (err != CL_SUCCESS)
        {
            releaseOpenCL();
            throw ACException<ExceptionType::GPU, true>("Failed to create command queue", err);
        }
    }
    if (parallelIO)
    {
#ifdef LEGACY_OPENCL_API
        commandQueueIO = clCreateCommandQueue(context, device, 0, &err);
#else
        commandQueueIO = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
#endif
        if (err != CL_SUCCESS)
        {
            releaseOpenCL();
            throw ACException<ExceptionType::GPU, true>("Failed to create command queue", err);
        }
    }
#endif

#ifndef BUILT_IN_KERNEL
    //read kernel files
    std::string ACNetKernelSourceString[TotalTypeCount];
    std::string kernelFiles[TotalTypeCount] =
    { "ACNetKernel.cl", "ACNetHDNL1Kernel.cl" ,"ACNetHDNL2Kernel.cl" ,"ACNetHDNL3Kernel.cl" };
#endif // BUILT_IN_KERNEL
    const char* ACNetKernelSource[TotalTypeCount];

    cl_kernel tmpKernel = nullptr;
#ifdef ENABLE_FAST_MATH
    const char* buildFlags = "-cl-fast-relaxed-math";
#else
    const char* buildFlags = nullptr;
#endif // ENABLE_FAST_MATH
    switch (type)
    {
    case CNNType::ACNetHDNL0:
#ifndef BUILT_IN_KERNEL
        //read kernel files
        ACNetKernelSourceString[HDNL0] = readKernel(kernelFiles[HDNL0]);
#endif // BUILT_IN_KERNEL
        ACNetKernelSource[HDNL0] = ACNetKernelSourceString[HDNL0].c_str();

        //create program
        program[HDNL0] = clCreateProgramWithSource(context, 1, &ACNetKernelSource[HDNL0], nullptr, &err);
        if (err != CL_SUCCESS)
        {
            releaseOpenCL();
            throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL program", err);
        }

        //build program
        err = clBuildProgram(program[HDNL0], 1, &device, buildFlags, nullptr, nullptr);
        if (err != CL_SUCCESS)
        {
            size_t buildErrorSize = 0;
            clGetProgramBuildInfo(program[HDNL0], device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &buildErrorSize);
            char* buildError = new char[buildErrorSize];
            clGetProgramBuildInfo(program[HDNL0], device, CL_PROGRAM_BUILD_LOG, buildErrorSize, buildError, nullptr);
            releaseOpenCL();
            ACException<ExceptionType::GPU, true> exception("Kernel build error", buildError, err);
            delete[] buildError;
            throw exception;
        }

        tmpKernel = clCreateKernel(program[HDNL0], "conv8To8", &err);
        if (err != CL_SUCCESS)
        {
            clReleaseKernel(tmpKernel);
            throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel for getting workGroupSizeLog", err);
        }
        err = clGetKernelWorkGroupInfo(tmpKernel, device,
            CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), (void*)&workGroupSizeLog, nullptr);
        if (err != CL_SUCCESS)
        {
            clReleaseKernel(tmpKernel);
            throw ACException<ExceptionType::GPU, true>("Failed to get workGroupSize", err);
        }
        workGroupSizeLog = std::log2(workGroupSizeLog);
        break;
    case CNNType::ACNetHDNL1:
#ifndef BUILT_IN_KERNEL
        //read kernel files
        ACNetKernelSourceString[HDNL1] = readKernel(kernelFiles[HDNL1]);
#endif // BUILT_IN_KERNEL
        ACNetKernelSource[HDNL1] = ACNetKernelSourceString[HDNL1].c_str();

        //create program
        program[HDNL1] = clCreateProgramWithSource(context, 1, &ACNetKernelSource[HDNL1], nullptr, &err);
        if (err != CL_SUCCESS)
        {
            releaseOpenCL();
            throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL program", err);
        }

        //build program
        err = clBuildProgram(program[HDNL1], 1, &device, buildFlags, nullptr, nullptr);
        if (err != CL_SUCCESS)
        {
            size_t buildErrorSize = 0;
            clGetProgramBuildInfo(program[HDNL1], device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &buildErrorSize);
            char* buildError = new char[buildErrorSize];
            clGetProgramBuildInfo(program[HDNL1], device, CL_PROGRAM_BUILD_LOG, buildErrorSize, buildError, nullptr);
            releaseOpenCL();
            ACException<ExceptionType::GPU, true> exception("Kernel build error", buildError, err);
            delete[] buildError;
            throw exception;
        }

        tmpKernel = clCreateKernel(program[HDNL1], "conv8To8", &err);
        if (err != CL_SUCCESS)
        {
            clReleaseKernel(tmpKernel);
            throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel for getting workGroupSizeLog", err);
        }
        err = clGetKernelWorkGroupInfo(tmpKernel, device,
            CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), (void*)&workGroupSizeLog, nullptr);
        if (err != CL_SUCCESS)
        {
            clReleaseKernel(tmpKernel);
            throw ACException<ExceptionType::GPU, true>("Failed to get workGroupSize", err);
        }
        workGroupSizeLog = std::log2(workGroupSizeLog);
        break;
    case CNNType::ACNetHDNL2:
#ifndef BUILT_IN_KERNEL
        //read kernel files
        ACNetKernelSourceString[HDNL2] = readKernel(kernelFiles[HDNL2]);
#endif // BUILT_IN_KERNEL
        ACNetKernelSource[HDNL2] = ACNetKernelSourceString[HDNL2].c_str();

        //create program
        program[HDNL2] = clCreateProgramWithSource(context, 1, &ACNetKernelSource[HDNL2], nullptr, &err);
        if (err != CL_SUCCESS)
        {
            releaseOpenCL();
            throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL program", err);
        }

        //build program
        err = clBuildProgram(program[HDNL2], 1, &device, buildFlags, nullptr, nullptr);
        if (err != CL_SUCCESS)
        {
            size_t buildErrorSize = 0;
            clGetProgramBuildInfo(program[HDNL2], device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &buildErrorSize);
            char* buildError = new char[buildErrorSize];
            clGetProgramBuildInfo(program[HDNL2], device, CL_PROGRAM_BUILD_LOG, buildErrorSize, buildError, nullptr);
            releaseOpenCL();
            ACException<ExceptionType::GPU, true> exception("Kernel build error", buildError, err);
            delete[] buildError;
            throw exception;
        }

        tmpKernel = clCreateKernel(program[HDNL2], "conv8To8", &err);
        if (err != CL_SUCCESS)
        {
            clReleaseKernel(tmpKernel);
            throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel for getting workGroupSizeLog", err);
        }
        err = clGetKernelWorkGroupInfo(tmpKernel, device,
            CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), (void*)&workGroupSizeLog, nullptr);
        if (err != CL_SUCCESS)
        {
            clReleaseKernel(tmpKernel);
            throw ACException<ExceptionType::GPU, true>("Failed to get workGroupSize", err);
        }
        workGroupSizeLog = std::log2(workGroupSizeLog);
        break;
    case CNNType::ACNetHDNL3:
#ifndef BUILT_IN_KERNEL
        //read kernel files
        ACNetKernelSourceString[HDNL3] = readKernel(kernelFiles[HDNL3]);
#endif // BUILT_IN_KERNEL
        ACNetKernelSource[HDNL3] = ACNetKernelSourceString[HDNL3].c_str();

        //create program
        program[HDNL3] = clCreateProgramWithSource(context, 1, &ACNetKernelSource[HDNL3], nullptr, &err);
        if (err != CL_SUCCESS)
        {
            releaseOpenCL();
            throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL program", err);
        }

        //build program
        err = clBuildProgram(program[HDNL3], 1, &device, buildFlags, nullptr, nullptr);
        if (err != CL_SUCCESS)
        {
            size_t buildErrorSize = 0;
            clGetProgramBuildInfo(program[HDNL3], device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &buildErrorSize);
            char* buildError = new char[buildErrorSize];
            clGetProgramBuildInfo(program[HDNL3], device, CL_PROGRAM_BUILD_LOG, buildErrorSize, buildError, nullptr);
            releaseOpenCL();
            ACException<ExceptionType::GPU, true> exception("Kernel build error", buildError, err);
            delete[] buildError;
            throw exception;
        }

        tmpKernel = clCreateKernel(program[HDNL3], "conv8To8", &err);
        if (err != CL_SUCCESS)
        {
            clReleaseKernel(tmpKernel);
            throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel for getting workGroupSizeLog", err);
        }
        err = clGetKernelWorkGroupInfo(tmpKernel, device,
            CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), (void*)&workGroupSizeLog, nullptr);
        if (err != CL_SUCCESS)
        {
            clReleaseKernel(tmpKernel);
            throw ACException<ExceptionType::GPU, true>("Failed to get workGroupSize", err);
        }
        workGroupSizeLog = std::log2(workGroupSizeLog);
        break;
    case CNNType::Default:
#ifndef BUILT_IN_KERNEL
        //read kernel files
        for (int i = HDNL0; i < TotalTypeCount; i++)
            ACNetKernelSourceString[i] = readKernel(kernelFiles[i]);
#endif // BUILT_IN_KERNEL
        for (int i = HDNL0; i < TotalTypeCount; i++)
        {
            ACNetKernelSource[i] = ACNetKernelSourceString[i].c_str();

            //create programACNet
            program[i] = clCreateProgramWithSource(context, 1, &ACNetKernelSource[i], nullptr, &err);
            if (err != CL_SUCCESS)
            {
                releaseOpenCL();
                throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL program", err);
            }

            //build programACNet
            err = clBuildProgram(program[i], 1, &device, buildFlags, nullptr, nullptr);
            if (err != CL_SUCCESS)
            {
                size_t buildErrorSize = 0;
                clGetProgramBuildInfo(program[i], device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &buildErrorSize);
                char* buildError = new char[buildErrorSize];
                clGetProgramBuildInfo(program[i], device, CL_PROGRAM_BUILD_LOG, buildErrorSize, buildError, nullptr);
                releaseOpenCL();
                ACException<ExceptionType::GPU, true> exception("Kernel build error", buildError, err);
                delete[] buildError;
                throw exception;
            }
        }

        tmpKernel = clCreateKernel(program[HDNL0], "conv8To8", &err);
        if (err != CL_SUCCESS)
        {
            clReleaseKernel(tmpKernel);
            throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel for getting workGroupSizeLog", err);
        }
        err = clGetKernelWorkGroupInfo(tmpKernel, device,
            CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), (void*)&workGroupSizeLog, nullptr);
        if (err != CL_SUCCESS)
        {
            clReleaseKernel(tmpKernel);
            throw ACException<ExceptionType::GPU, true>("Failed to get workGroupSize", err);
        }
        workGroupSizeLog = std::log2(workGroupSizeLog);
        break;
    }
    clReleaseKernel(tmpKernel);
}

void Anime4KCPP::OpenCL::ACNet::releaseOpenCL() noexcept
{
    for (auto& commandQueue : commandQueueList)
    {
        if (commandQueue != nullptr)
            clReleaseCommandQueue(commandQueue);
    }
    if (commandQueueIO != nullptr)
        clReleaseCommandQueue(commandQueueIO);
    for (int i = HDNL0; i < TotalTypeCount; i++)
    {
        if (program[i] != nullptr)
            clReleaseProgram(program[i]);
    }
    if (context != nullptr)
        clReleaseContext(context);
}

std::string Anime4KCPP::OpenCL::ACNet::readKernel(const std::string& fileName)
{
    std::ifstream kernelFile(fileName);
    if (!kernelFile.is_open())
        throw ACException<ExceptionType::IO>("Failed to open kernel file.");

    std::ostringstream source;
    source << kernelFile.rdbuf();

    return source.str();
}

Anime4KCPP::Processor::Type Anime4KCPP::OpenCL::ACNet::getProcessorType() noexcept
{
    return Processor::Type::OpenCL_ACNet;
}

std::string Anime4KCPP::OpenCL::ACNet::getProcessorInfo()
{
    cl_int err = 0;
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;

    size_t platformNameLength = 0;
    size_t deviceNameLength = 0;

    auto tmpPlatform = std::make_unique<cl_platform_id[]>(static_cast<size_t>(pID) + 1);
    err = clGetPlatformIDs(pID+1, tmpPlatform.get(), nullptr);
    if (err != CL_SUCCESS)
        throw ACException<ExceptionType::GPU, true>("Failed to find OpenCL platforms.", err);

    platform = tmpPlatform[pID];

    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameLength);
    if (err != CL_SUCCESS)
        throw ACException<ExceptionType::GPU, true>("Failed to get OpenCL platform information.", err);

    auto platformName = std::make_unique<char[]>(platformNameLength);
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameLength, platformName.get(), nullptr);
    if (err != CL_SUCCESS)
        throw ACException<ExceptionType::GPU, true>("Failed to get OpenCL platform information.", err);

    auto tmpDevice = std::make_unique<cl_device_id[]>(static_cast<size_t>(dID) + 1);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, dID + 1, tmpDevice.get(), nullptr);
    if (err != CL_SUCCESS)
        throw ACException<ExceptionType::GPU, true>("Failed to find OpenCL devices.", err);

    device = tmpDevice[dID];

    err = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameLength);
    if (err != CL_SUCCESS)
        throw ACException<ExceptionType::GPU, true>("Failed to get OpenCL devices information.", err);

    auto deviceName = std::make_unique<char[]>(deviceNameLength);
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameLength, deviceName.get(), nullptr);
    if (err != CL_SUCCESS)
        throw ACException<ExceptionType::GPU, true>("Failed to get OpenCL devices information.", err);

    std::ostringstream oss;
    oss << "Processor type: " << getProcessorType() << std::endl
        << "Current OpenCL devices:" << std::endl
        << " Platform " + std::to_string(pID) + ": " + platformName.get() << std::endl
        << "  Device " + std::to_string(dID) + ": " + deviceName.get();
    return oss.str();
}

//init OpenCL arguments
bool Anime4KCPP::OpenCL::ACNet::isInitialized = false;
cl_context Anime4KCPP::OpenCL::ACNet::context = nullptr;
int Anime4KCPP::OpenCL::ACNet::commandQueueNum = 4;
int Anime4KCPP::OpenCL::ACNet::commandQueueCount = 0;
std::vector<cl_command_queue> Anime4KCPP::OpenCL::ACNet::commandQueueList(commandQueueNum, nullptr);
bool Anime4KCPP::OpenCL::ACNet::parallelIO = false;
cl_command_queue Anime4KCPP::OpenCL::ACNet::commandQueueIO = nullptr;
cl_program Anime4KCPP::OpenCL::ACNet::program[TotalTypeCount] = { nullptr };
cl_device_id Anime4KCPP::OpenCL::ACNet::device = nullptr;
int Anime4KCPP::OpenCL::ACNet::pID = 0;
int Anime4KCPP::OpenCL::ACNet::dID = 0;
size_t Anime4KCPP::OpenCL::ACNet::workGroupSizeLog = 5;
