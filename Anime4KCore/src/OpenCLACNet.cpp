#ifdef ENABLE_OPENCL

#define DLL

#include<fstream>

#define CL_HPP_ENABLE_EXCEPTIONS
#ifdef LEGACY_OPENCL_API
#define CL_HPP_TARGET_OPENCL_VERSION 120
#else
#define CL_HPP_TARGET_OPENCL_VERSION 200
#endif // LEGACY_OPENCL_API
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#include<CL/opencl.hpp>

#include"ACNetType.hpp"
#include"OpenCLACNet.hpp"
#include"OpenCLACNetKernel.hpp"

#define ALIGN_UP(x, size) (((x) + (size) - 1) & (~((size) - 1)))

constexpr static int L2 = 0, L3 = 1, L4 = 2, L5 = 3, L6 = 4, L7 = 5, L8 = 6, L9 = 7;

//init OpenCL arguments
static bool isInitialized = false;
static cl::Program program[Anime4KCPP::ACNetType::TotalTypeCount];
static cl::Device device;
static cl::Context context;
static cl::CommandQueue commandQueueIO;
static int commandQueueNum = 4;
static int commandQueueCount = 0;
static std::vector<cl::CommandQueue> commandQueueList(commandQueueNum);
static bool parallelIO = false;
static int pID = 0;
static int dID = 0;
static size_t workGroupSizeBase = 32;

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

void Anime4KCPP::OpenCL::ACNet::setParameters(const Parameters& parameters)
{
    AC::setParameters(parameters);
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
        context = nullptr;
        std::fill(commandQueueList.begin(), commandQueueList.end(), nullptr);
        commandQueueIO = nullptr;
        for (int i = HDNL0; i < TotalTypeCount; i++)
            program[i] = nullptr;
        device = nullptr;
        isInitialized = false;
    }
}

bool Anime4KCPP::OpenCL::ACNet::isInitializedGPU() noexcept
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

        cv::Mat tmpImg;
        cv::cvtColor(orgImg, tmpImg, cv::COLOR_BGR2YUV);

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

        cv::Mat tmpImg;
        cv::cvtColor(orgImg, tmpImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(tmpImg, yuv);

        dstImg.create(yuv[Y].rows * 2, yuv[Y].cols * 2, CV_8UC1);
        if (parallelIO)
            runKernelPB(yuv[Y], dstImg);
        else
            runKernelB(yuv[Y], dstImg);

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

        cv::Mat tmpImg;
        cv::cvtColor(orgImg, tmpImg, cv::COLOR_BGR2YUV);

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

        cv::Mat tmpImg;
        cv::cvtColor(orgImg, tmpImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(tmpImg, yuv);

        dstImg.create(yuv[Y].rows * 2, yuv[Y].cols * 2, CV_16UC1);
        if (parallelIO)
            runKernelPW(yuv[Y], dstImg);
        else
            runKernelW(yuv[Y], dstImg);

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

        cv::Mat tmpImg;
        cv::cvtColor(orgImg, tmpImg, cv::COLOR_BGR2YUV);

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

        cv::Mat tmpImg;
        cv::cvtColor(orgImg, tmpImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(tmpImg, yuv);

        dstImg.create(yuv[Y].rows * 2, yuv[Y].cols * 2, CV_32FC1);
        if (parallelIO)
            runKernelPF(yuv[Y], dstImg);
        else
            runKernelF(yuv[Y], dstImg);

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

void Anime4KCPP::OpenCL::ACNet::runKernel(const cv::Mat& orgImg, cv::Mat& dstImg, int channelType)
{
    constexpr std::array<size_t, 3> orgin = { 0,0,0 };
    const std::array<size_t, 3> orgRegion = { static_cast<const size_t>(orgImg.cols),static_cast<const size_t>(orgImg.rows),1 };
    const std::array<size_t, 3> dstRegion = { static_cast<const size_t>(dstImg.cols),static_cast<const size_t>(dstImg.rows),1 };
    const std::array<size_t, 2> orgSize =
    {
        ALIGN_UP(orgImg.cols, workGroupSizeBase),
        ALIGN_UP(orgImg.rows, workGroupSizeBase)
    };
    const std::array<size_t, 2> dstSize =
    {
        ALIGN_UP(dstImg.cols, workGroupSizeBase),
        ALIGN_UP(dstImg.rows, workGroupSizeBase)
    };

    try
    {
        cl::CommandQueue& commandQueue = commandQueueList[commandQueueCount++];
        if (commandQueueCount >= commandQueueNum)
            commandQueueCount = 0;

        //kernel for each thread
        cl::Kernel kernelConv1To8L1(program[currACNetypeIndex], "conv1To8");
        cl::Kernel kernelConv8To8L2(program[currACNetypeIndex], "conv8To8");
        cl::Kernel kernelConv8To8L3(program[currACNetypeIndex], "conv8To8");
        cl::Kernel kernelConv8To8L4(program[currACNetypeIndex], "conv8To8");
        cl::Kernel kernelConv8To8L5(program[currACNetypeIndex], "conv8To8");
        cl::Kernel kernelConv8To8L6(program[currACNetypeIndex], "conv8To8");
        cl::Kernel kernelConv8To8L7(program[currACNetypeIndex], "conv8To8");
        cl::Kernel kernelConv8To8L8(program[currACNetypeIndex], "conv8To8");
        cl::Kernel kernelConv8To8L9(program[currACNetypeIndex], "conv8To8");
        cl::Kernel kernelConvTranspose8To1L10(program[currACNetypeIndex], "convTranspose8To1");

        cl::ImageFormat imageFormat(CL_R, channelType);
        cl::ImageFormat tmpFormat(CL_RGBA, CL_HALF_FLOAT);

        cl::Image2D imageBufferOrg(context, CL_MEM_READ_ONLY, imageFormat, orgImg.cols, orgImg.rows);
        cl::Image2DArray imageBufferTmp1(context, CL_MEM_READ_WRITE, tmpFormat, 2, orgImg.cols, orgImg.rows, 0, 0);
        cl::Image2DArray imageBufferTmp2(context, CL_MEM_READ_WRITE, tmpFormat, 2, orgImg.cols, orgImg.rows, 0, 0);
        cl::Image2D imageBufferDst(context, CL_MEM_WRITE_ONLY, imageFormat, dstImg.cols, dstImg.rows);

        kernelConv1To8L1.setArg(0, sizeof(cl_mem), &imageBufferOrg);
        kernelConv1To8L1.setArg(1, sizeof(cl_mem), &imageBufferTmp1);

        kernelConv8To8L2.setArg(0, sizeof(cl_mem), &imageBufferTmp1);
        kernelConv8To8L2.setArg(1, sizeof(cl_mem), &imageBufferTmp2);
        kernelConv8To8L2.setArg(2, sizeof(cl_int), &L2);

        kernelConv8To8L3.setArg(0, sizeof(cl_mem), &imageBufferTmp2);
        kernelConv8To8L3.setArg(1, sizeof(cl_mem), &imageBufferTmp1);
        kernelConv8To8L3.setArg(2, sizeof(cl_int), &L3);

        kernelConv8To8L4.setArg(0, sizeof(cl_mem), &imageBufferTmp1);
        kernelConv8To8L4.setArg(1, sizeof(cl_mem), &imageBufferTmp2);
        kernelConv8To8L4.setArg(2, sizeof(cl_int), &L4);

        kernelConv8To8L5.setArg(0, sizeof(cl_mem), &imageBufferTmp2);
        kernelConv8To8L5.setArg(1, sizeof(cl_mem), &imageBufferTmp1);
        kernelConv8To8L5.setArg(2, sizeof(cl_int), &L5);

        kernelConv8To8L6.setArg(0, sizeof(cl_mem), &imageBufferTmp1);
        kernelConv8To8L6.setArg(1, sizeof(cl_mem), &imageBufferTmp2);
        kernelConv8To8L6.setArg(2, sizeof(cl_int), &L6);

        kernelConv8To8L7.setArg(0, sizeof(cl_mem), &imageBufferTmp2);
        kernelConv8To8L7.setArg(1, sizeof(cl_mem), &imageBufferTmp1);
        kernelConv8To8L7.setArg(2, sizeof(cl_int), &L7);

        kernelConv8To8L8.setArg(0, sizeof(cl_mem), &imageBufferTmp1);
        kernelConv8To8L8.setArg(1, sizeof(cl_mem), &imageBufferTmp2);
        kernelConv8To8L8.setArg(2, sizeof(cl_int), &L8);

        kernelConv8To8L9.setArg(0, sizeof(cl_mem), &imageBufferTmp2);
        kernelConv8To8L9.setArg(1, sizeof(cl_mem), &imageBufferTmp1);
        kernelConv8To8L9.setArg(2, sizeof(cl_int), &L9);

        kernelConvTranspose8To1L10.setArg(0, sizeof(cl_mem), &imageBufferTmp1);
        kernelConvTranspose8To1L10.setArg(1, sizeof(cl_mem), &imageBufferDst);

        commandQueue.enqueueWriteImage(imageBufferOrg, CL_FALSE, orgin, orgRegion, orgImg.step, 0, orgImg.data);
        commandQueue.enqueueNDRangeKernel(kernelConv1To8L1, cl::NullRange, cl::NDRange(orgSize[0], orgSize[1]));
        commandQueue.enqueueNDRangeKernel(kernelConv8To8L2, cl::NullRange, cl::NDRange(orgSize[0], orgSize[1]));
        commandQueue.enqueueNDRangeKernel(kernelConv8To8L3, cl::NullRange, cl::NDRange(orgSize[0], orgSize[1]));
        commandQueue.enqueueNDRangeKernel(kernelConv8To8L4, cl::NullRange, cl::NDRange(orgSize[0], orgSize[1]));
        commandQueue.enqueueNDRangeKernel(kernelConv8To8L5, cl::NullRange, cl::NDRange(orgSize[0], orgSize[1]));
        commandQueue.enqueueNDRangeKernel(kernelConv8To8L6, cl::NullRange, cl::NDRange(orgSize[0], orgSize[1]));
        commandQueue.enqueueNDRangeKernel(kernelConv8To8L7, cl::NullRange, cl::NDRange(orgSize[0], orgSize[1]));
        commandQueue.enqueueNDRangeKernel(kernelConv8To8L8, cl::NullRange, cl::NDRange(orgSize[0], orgSize[1]));
        commandQueue.enqueueNDRangeKernel(kernelConv8To8L9, cl::NullRange, cl::NDRange(orgSize[0], orgSize[1]));
        commandQueue.enqueueNDRangeKernel(kernelConvTranspose8To1L10, cl::NullRange, cl::NDRange(dstSize[0], dstSize[1]));
        commandQueue.enqueueReadImage(imageBufferDst, CL_TRUE, orgin, dstRegion, dstImg.step, 0, dstImg.data);
    }
    catch (const cl::Error& e)
    {
        throw ACException<ExceptionType::GPU, true>("Faile to run OpenCL ACNet kernel", e.what(), e.err());
    }
}

void Anime4KCPP::OpenCL::ACNet::runKernelP(const cv::Mat& orgImg, cv::Mat& dstImg, int channelType)
{
    constexpr std::array<size_t, 3> orgin = { 0,0,0 };
    const std::array<size_t, 3> orgRegion = { static_cast<const size_t>(orgImg.cols),static_cast<const size_t>(orgImg.rows),1 };
    const std::array<size_t, 3> dstRegion = { static_cast<const size_t>(dstImg.cols),static_cast<const size_t>(dstImg.rows),1 };
    const std::array<size_t, 2> orgSize =
    {
        ALIGN_UP(orgImg.cols, workGroupSizeBase),
        ALIGN_UP(orgImg.rows, workGroupSizeBase)
    };
    const std::array<size_t, 2> dstSize =
    {
        ALIGN_UP(dstImg.cols, workGroupSizeBase),
        ALIGN_UP(dstImg.rows, workGroupSizeBase)
    };

    try
    {
        cl::CommandQueue& commandQueue = commandQueueList[commandQueueCount++];
        if (commandQueueCount >= commandQueueNum)
            commandQueueCount = 0;

        //kernel for each thread
        cl::Kernel kernelConv1To8L1(program[currACNetypeIndex], "conv1To8");
        cl::Kernel kernelConv8To8L2(program[currACNetypeIndex], "conv8To8");
        cl::Kernel kernelConv8To8L3(program[currACNetypeIndex], "conv8To8");
        cl::Kernel kernelConv8To8L4(program[currACNetypeIndex], "conv8To8");
        cl::Kernel kernelConv8To8L5(program[currACNetypeIndex], "conv8To8");
        cl::Kernel kernelConv8To8L6(program[currACNetypeIndex], "conv8To8");
        cl::Kernel kernelConv8To8L7(program[currACNetypeIndex], "conv8To8");
        cl::Kernel kernelConv8To8L8(program[currACNetypeIndex], "conv8To8");
        cl::Kernel kernelConv8To8L9(program[currACNetypeIndex], "conv8To8");
        cl::Kernel kernelConvTranspose8To1L10(program[currACNetypeIndex], "convTranspose8To1");

        cl::ImageFormat imageFormat(CL_R, channelType);
        cl::ImageFormat tmpFormat(CL_RGBA, CL_HALF_FLOAT);

        cl::Image2D imageBufferOrg(context, CL_MEM_READ_ONLY, imageFormat, orgImg.cols, orgImg.rows);
        cl::Image2DArray imageBufferTmp1(context, CL_MEM_READ_WRITE, tmpFormat, 2, orgImg.cols, orgImg.rows, 0, 0);
        cl::Image2DArray imageBufferTmp2(context, CL_MEM_READ_WRITE, tmpFormat, 2, orgImg.cols, orgImg.rows, 0, 0);
        cl::Image2D imageBufferDst(context, CL_MEM_WRITE_ONLY, imageFormat, dstImg.cols, dstImg.rows);

        kernelConv1To8L1.setArg(0, sizeof(cl_mem), &imageBufferOrg);
        kernelConv1To8L1.setArg(1, sizeof(cl_mem), &imageBufferTmp1);

        kernelConv8To8L2.setArg(0, sizeof(cl_mem), &imageBufferTmp1);
        kernelConv8To8L2.setArg(1, sizeof(cl_mem), &imageBufferTmp2);
        kernelConv8To8L2.setArg(2, sizeof(cl_int), &L2);

        kernelConv8To8L3.setArg(0, sizeof(cl_mem), &imageBufferTmp2);
        kernelConv8To8L3.setArg(1, sizeof(cl_mem), &imageBufferTmp1);
        kernelConv8To8L3.setArg(2, sizeof(cl_int), &L3);

        kernelConv8To8L4.setArg(0, sizeof(cl_mem), &imageBufferTmp1);
        kernelConv8To8L4.setArg(1, sizeof(cl_mem), &imageBufferTmp2);
        kernelConv8To8L4.setArg(2, sizeof(cl_int), &L4);

        kernelConv8To8L5.setArg(0, sizeof(cl_mem), &imageBufferTmp2);
        kernelConv8To8L5.setArg(1, sizeof(cl_mem), &imageBufferTmp1);
        kernelConv8To8L5.setArg(2, sizeof(cl_int), &L5);

        kernelConv8To8L6.setArg(0, sizeof(cl_mem), &imageBufferTmp1);
        kernelConv8To8L6.setArg(1, sizeof(cl_mem), &imageBufferTmp2);
        kernelConv8To8L6.setArg(2, sizeof(cl_int), &L6);

        kernelConv8To8L7.setArg(0, sizeof(cl_mem), &imageBufferTmp2);
        kernelConv8To8L7.setArg(1, sizeof(cl_mem), &imageBufferTmp1);
        kernelConv8To8L7.setArg(2, sizeof(cl_int), &L7);

        kernelConv8To8L8.setArg(0, sizeof(cl_mem), &imageBufferTmp1);
        kernelConv8To8L8.setArg(1, sizeof(cl_mem), &imageBufferTmp2);
        kernelConv8To8L8.setArg(2, sizeof(cl_int), &L8);

        kernelConv8To8L9.setArg(0, sizeof(cl_mem), &imageBufferTmp2);
        kernelConv8To8L9.setArg(1, sizeof(cl_mem), &imageBufferTmp1);
        kernelConv8To8L9.setArg(2, sizeof(cl_int), &L9);

        kernelConvTranspose8To1L10.setArg(0, sizeof(cl_mem), &imageBufferTmp1);
        kernelConvTranspose8To1L10.setArg(1, sizeof(cl_mem), &imageBufferDst);


        std::vector<cl::Event> waitForWriteFinishedEvent(1);
        std::vector<cl::Event> waitForReadReadyEvent(1);
        cl::Event& writeFinishedEvent = waitForWriteFinishedEvent.front();
        cl::Event& readReadyEvent = waitForReadReadyEvent.front();
        cl::Event  readFinishedEvent;

        commandQueueIO.enqueueWriteImage(imageBufferOrg, CL_FALSE, orgin, orgRegion, orgImg.step, 0, orgImg.data, nullptr, &writeFinishedEvent);
        commandQueue.enqueueNDRangeKernel(kernelConv1To8L1, cl::NullRange, cl::NDRange(orgSize[0], orgSize[1]), cl::NullRange, &waitForWriteFinishedEvent);
        commandQueue.enqueueNDRangeKernel(kernelConv8To8L2, cl::NullRange, cl::NDRange(orgSize[0], orgSize[1]));
        commandQueue.enqueueNDRangeKernel(kernelConv8To8L3, cl::NullRange, cl::NDRange(orgSize[0], orgSize[1]));
        commandQueue.enqueueNDRangeKernel(kernelConv8To8L4, cl::NullRange, cl::NDRange(orgSize[0], orgSize[1]));
        commandQueue.enqueueNDRangeKernel(kernelConv8To8L5, cl::NullRange, cl::NDRange(orgSize[0], orgSize[1]));
        commandQueue.enqueueNDRangeKernel(kernelConv8To8L6, cl::NullRange, cl::NDRange(orgSize[0], orgSize[1]));
        commandQueue.enqueueNDRangeKernel(kernelConv8To8L7, cl::NullRange, cl::NDRange(orgSize[0], orgSize[1]));
        commandQueue.enqueueNDRangeKernel(kernelConv8To8L8, cl::NullRange, cl::NDRange(orgSize[0], orgSize[1]));
        commandQueue.enqueueNDRangeKernel(kernelConv8To8L9, cl::NullRange, cl::NDRange(orgSize[0], orgSize[1]));
        commandQueue.enqueueNDRangeKernel(kernelConvTranspose8To1L10, cl::NullRange, cl::NDRange(dstSize[0], dstSize[1]), cl::NullRange, nullptr, &readReadyEvent);
        commandQueueIO.enqueueReadImage(imageBufferDst, CL_FALSE, orgin, dstRegion, dstImg.step, 0, dstImg.data, &waitForReadReadyEvent, &readFinishedEvent);
        readFinishedEvent.wait();
    }
    catch (const cl::Error& e)
    {
        throw ACException<ExceptionType::GPU, true>("Faile to run OpenCL ACNet kernel", e.what(), e.err());
    }
}

void Anime4KCPP::OpenCL::ACNet::runKernelB(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    runKernel(orgImg, dstImg, CL_UNORM_INT8);
}

void Anime4KCPP::OpenCL::ACNet::runKernelW(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    runKernel(orgImg, dstImg, CL_UNORM_INT16);
}

void Anime4KCPP::OpenCL::ACNet::runKernelF(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    runKernel(orgImg, dstImg, CL_FLOAT);
}

void Anime4KCPP::OpenCL::ACNet::runKernelPB(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    runKernelP(orgImg, dstImg, CL_UNORM_INT8);
}

void Anime4KCPP::OpenCL::ACNet::runKernelPW(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    runKernelP(orgImg, dstImg, CL_UNORM_INT16);
}

void Anime4KCPP::OpenCL::ACNet::runKernelPF(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    runKernelP(orgImg, dstImg, CL_FLOAT);
}

void Anime4KCPP::OpenCL::ACNet::initOpenCL(const CNNType type)
{
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;

    try
    {
        cl::Platform::get(&platforms);
        ((pID >= 0 && pID < platforms.size()) ? platforms[pID] : platforms[0])
            .getDevices(CL_DEVICE_TYPE_GPU, &devices);

        device = (dID >= 0 && dID < devices.size()) ? devices[dID] : devices[0];

        context = cl::Context(device);

        commandQueueList.resize(commandQueueNum);
        for (int i = 0; i < commandQueueNum; i++)
        {
            commandQueueList[i] = cl::CommandQueue(context, device);
        }
        if (parallelIO)
        {
            commandQueueIO = cl::CommandQueue(context, device);
        }

#ifndef BUILT_IN_KERNEL
        //read kernel files
        std::string ACNetKernelSourceString[TotalTypeCount];
        std::string kernelFiles[TotalTypeCount] =
        { "ACNetKernel.cl", "ACNetHDNL1Kernel.cl" ,"ACNetHDNL2Kernel.cl" ,"ACNetHDNL3Kernel.cl" };
#endif // BUILT_IN_KERNEL

#ifdef ENABLE_FAST_MATH
        const char* buildFlags = "-cl-fast-relaxed-math";
#else
        const char* buildFlags = nullptr;
#endif // ENABLE_FAST_MATH

        switch (type)
        {
        case CNNType::ACNetHDNL0:
        {
#ifndef BUILT_IN_KERNEL
            //read kernel files
            ACNetKernelSourceString[HDNL0] = readKernel(kernelFiles[HDNL0]);
#endif // BUILT_IN_KERNEL
            program[HDNL0] = cl::Program(context, ACNetKernelSourceString[HDNL0]);
            try
            {
                program[HDNL0].build(device, buildFlags);
            }
            catch (const cl::BuildError& e)
            {
                throw ACException<ExceptionType::GPU, true>("Kernel build error", e.getBuildLog().front().second, e.err());
            }

            cl::Kernel tmpKernel{ program[HDNL0], "conv8To8" };
            tmpKernel.getWorkGroupInfo(device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &workGroupSizeBase);
        }
        break;
        case CNNType::ACNetHDNL1:
        {
#ifndef BUILT_IN_KERNEL
            //read kernel files
            ACNetKernelSourceString[HDNL1] = readKernel(kernelFiles[HDNL1]);
#endif // BUILT_IN_KERNEL
            program[HDNL1] = cl::Program(context, ACNetKernelSourceString[HDNL1]);
            try
            {
                program[HDNL1].build(device, buildFlags);
            }
            catch (const cl::BuildError& e)
            {
                throw ACException<ExceptionType::GPU, true>("Kernel build error", e.getBuildLog().front().second, e.err());
            }

            cl::Kernel tmpKernel{ program[HDNL1], "conv8To8" };
            tmpKernel.getWorkGroupInfo(device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &workGroupSizeBase);
        }
            break;
        case CNNType::ACNetHDNL2:
        {
#ifndef BUILT_IN_KERNEL
            //read kernel files
            ACNetKernelSourceString[HDNL2] = readKernel(kernelFiles[HDNL2]);
#endif // BUILT_IN_KERNEL
            program[HDNL2] = cl::Program(context, ACNetKernelSourceString[HDNL2]);
            try
            {
                program[HDNL2].build(device, buildFlags);
            }
            catch (const cl::BuildError& e)
            {
                throw ACException<ExceptionType::GPU, true>("Kernel build error", e.getBuildLog().front().second, e.err());
            }

            cl::Kernel tmpKernel{ program[HDNL2], "conv8To8" };
            tmpKernel.getWorkGroupInfo(device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &workGroupSizeBase);
        }
            break;
        case CNNType::ACNetHDNL3:
        {
#ifndef BUILT_IN_KERNEL
            //read kernel files
            ACNetKernelSourceString[HDNL3] = readKernel(kernelFiles[HDNL3]);
#endif // BUILT_IN_KERNEL
            program[HDNL3] = cl::Program(context, ACNetKernelSourceString[HDNL3]);
            try
            {
                program[HDNL3].build(device, buildFlags);
            }
            catch (const cl::BuildError& e)
            {
                throw ACException<ExceptionType::GPU, true>("Kernel build error", e.getBuildLog().front().second, e.err());
            }

            cl::Kernel tmpKernel{ program[HDNL3], "conv8To8" };
            tmpKernel.getWorkGroupInfo(device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &workGroupSizeBase);
        }
            break;
        case CNNType::Default:
        default:
        {
#ifndef BUILT_IN_KERNEL
            //read kernel files
            for (int i = HDNL0; i < TotalTypeCount; i++)
                ACNetKernelSourceString[i] = readKernel(kernelFiles[i]);
#endif // BUILT_IN_KERNEL
            for (int i = HDNL0; i < TotalTypeCount; i++)
            {
                program[i] = cl::Program(context, ACNetKernelSourceString[i]);
                try
                {
                    program[i].build(device, buildFlags);
                }
                catch (const cl::BuildError& e)
                {
                    throw ACException<ExceptionType::GPU, true>("Kernel build error", e.getBuildLog().front().second, e.err());
                }
            }

            cl::Kernel tmpKernel{ program[HDNL0], "conv8To8" };
            tmpKernel.getWorkGroupInfo(device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &workGroupSizeBase);
        }
            break;
        }
    }
    catch (const cl::Error& e)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to initialize OpenCL", e.what(), e.err());
    }
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
    std::ostringstream oss;
    oss << "Processor type: " << getProcessorType() << std::endl
        << "Current OpenCL devices:" << std::endl
        << " " + device.getInfo<CL_DEVICE_NAME>();
    return oss.str();
}

#endif
