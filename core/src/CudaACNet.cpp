#ifdef ENABLE_CUDA

#include"CudaInterface.hpp"
#include"ACNetType.hpp"
#include"CudaACNet.hpp"

namespace Anime4KCPP::Cuda::detail
{
    static void runKernel(const cv::Mat& orgImg, cv::Mat& dstImg, int index)
    {
        ACCudaParamACNet cuParam{ orgImg.cols, orgImg.rows, orgImg.step };
        ACCudaDataType dataType;

        switch (orgImg.depth())
        {
        case CV_8U:
            dataType = ACCudaDataType::AC_8U;
            break;
        case CV_16U:
            dataType = ACCudaDataType::AC_16U;
            break;
        case CV_32F:
            dataType = ACCudaDataType::AC_32F;
            break;
        default:
            throw ACException<ExceptionType::RunTimeError>("Unsupported image data type");
        }

        switch (index)
        {
        case 0:
            cuRunKernelACNetHDN0(orgImg.data, dstImg.data, dataType, &cuParam);
            break;
        case 1:
            cuRunKernelACNetHDN1(orgImg.data, dstImg.data, dataType, &cuParam);
            break;
        case 2:
            cuRunKernelACNetHDN2(orgImg.data, dstImg.data, dataType, &cuParam);
            break;
        case 3:
            cuRunKernelACNetHDN3(orgImg.data, dstImg.data, dataType, &cuParam);
            break;
        default:
            cuRunKernelACNetHDN1(orgImg.data, dstImg.data, dataType, &cuParam);
            break;
        }
    }
}

Anime4KCPP::Cuda::ACNet::ACNet(const Parameters& parameters) :
    AC(parameters)
{
    ACNetTypeIndex = GET_ACNET_TYPE_INDEX(param.HDN, param.HDNLevel);
}

void Anime4KCPP::Cuda::ACNet::setParameters(const Parameters& parameters)
{
    AC::setParameters(parameters);
    ACNetTypeIndex = GET_ACNET_TYPE_INDEX(param.HDN, param.HDNLevel);
}

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

void Anime4KCPP::Cuda::ACNet::processYUVImage()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        cv::Mat tmpImg = orgImg;
        for (int i = 0; i < scaleTimes; i++)
        {
            dstImg.create(tmpImg.rows * 2, tmpImg.cols * 2, tmpImg.type());
            detail::runKernel(tmpImg, dstImg, ACNetTypeIndex);
            tmpImg = dstImg;
        }
        if (param.isNonIntegerScale())
        {
            cv::resize(dstImg, dstImg, cv::Size(width, height), 0.0, 0.0, cv::INTER_AREA);
        }

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
    }
    else
    {
        cv::Mat tmpImg = orgImg;

        if (param.zoomFactor > 2.0)
            cv::resize(tmpImg, tmpImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(tmpImg, tmpImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        cv::Mat outMat(tmpImg.rows * 2, tmpImg.cols * 2, tmpImg.type());
        detail::runKernel(tmpImg, outMat, ACNetTypeIndex);
        dstImg = outMat;

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
    }
}

void Anime4KCPP::Cuda::ACNet::processRGBImage()
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
            dstImg.create(tmpImg.rows * 2, tmpImg.cols * 2, tmpImg.type());
            detail::runKernel(tmpImg, dstImg, ACNetTypeIndex);
            tmpImg = dstImg;
        }
        if (param.isNonIntegerScale())
        {
            cv::resize(dstImg, dstImg, cv::Size(width, height), 0.0, 0.0, cv::INTER_AREA);
        }

        cv::resize(yuv[U], yuv[U], cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(yuv[V], yuv[V], cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
    else
    {
        cv::Mat tmpImg;
        cv::cvtColor(orgImg, tmpImg, cv::COLOR_BGR2YUV);

        if (param.zoomFactor > 2.0)
            cv::resize(tmpImg, tmpImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(tmpImg, tmpImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        std::vector<cv::Mat> yuv(3);
        cv::split(tmpImg, yuv);

        cv::Mat outMat(yuv[Y].rows * 2, yuv[Y].cols * 2, yuv[Y].type());
        detail::runKernel(yuv[Y], outMat, ACNetTypeIndex);

        cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
        cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);

        cv::merge(std::vector<cv::Mat>{ outMat, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
}

void Anime4KCPP::Cuda::ACNet::processGrayscale()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        cv::Mat tmpImg = orgImg;
        for (int i = 0; i < scaleTimes; i++)
        {
            dstImg.create(tmpImg.rows * 2, tmpImg.cols * 2, tmpImg.type());
            detail::runKernel(tmpImg, dstImg, ACNetTypeIndex);
            tmpImg = dstImg;
        }
        if (param.isNonIntegerScale())
        {
            cv::resize(dstImg, dstImg, cv::Size(width, height), 0.0, 0.0, cv::INTER_AREA);
        }
    }
    else
    {
        cv::Mat tmpImg = orgImg;

        if (param.zoomFactor > 2.0)
            cv::resize(tmpImg, tmpImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(tmpImg, tmpImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        cv::Mat outMat(tmpImg.rows * 2, tmpImg.cols * 2, tmpImg.type());
        detail::runKernel(tmpImg, outMat, ACNetTypeIndex);
        dstImg = outMat;
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
