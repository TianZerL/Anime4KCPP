#ifdef ENABLE_CUDA

#include"CudaInterface.hpp"
#include"FilterProcessor.hpp"
#include"CudaAnime4K09.hpp"

namespace Anime4KCPP::Cuda::detail
{
    static void runKernel(const cv::Mat& orgImg, cv::Mat& dstImg, const Parameters& param)
    {
        ACCudaParamAnime4K09 cuParam{
        orgImg.cols, orgImg.rows,
        dstImg.cols, dstImg.rows,
        orgImg.step,
        param.passes, param.pushColorCount,
        static_cast<float>(param.strengthColor),static_cast<float>(param.strengthGradient)
        };

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

        cuRunKernelAnime4K09(orgImg.data, dstImg.data, dataType, &cuParam);
    }
}

std::string Anime4KCPP::Cuda::Anime4K09::getInfo()
{
    std::ostringstream oss;
    oss << AC::getInfo()
        << "----------------------------------------------" << std::endl
        << "CUDA Device ID:" << cuGetDeviceID() << std::endl
        << "Passes: " << param.passes << std::endl
        << "pushColorCount: " << param.pushColorCount << std::endl
        << "Zoom Factor: " << param.zoomFactor << std::endl
        << "Fast Mode: " << std::boolalpha << param.fastMode << std::endl
        << "Strength Color: " << param.strengthColor << std::endl
        << "Strength Gradient: " << param.strengthGradient << std::endl
        << "----------------------------------------------" << std::endl;
    return oss.str();
}

std::string Anime4KCPP::Cuda::Anime4K09::getFiltersInfo()
{
    std::ostringstream oss;

    oss << AC::getFiltersInfo()
        << "----------------------------------------------" << std::endl
        << "Preprocessing filters list:" << std::endl
        << "----------------------------------------------" << std::endl;
    if (!param.preprocessing)
        oss << "Preprocessing disabled" << std::endl;
    else
    {
        std::vector<std::string>preFiltersString = FilterProcessor::filterToString(param.preFilters);
        if (preFiltersString.empty())
            oss << "Preprocessing disabled" << std::endl;
        else
            for (auto& filters : preFiltersString)
                oss << filters << std::endl;
    }

    oss << "----------------------------------------------" << std::endl
        << "Postprocessing filters list:" << std::endl
        << "----------------------------------------------" << std::endl;
    if (!param.postprocessing)
        oss << "Postprocessing disabled" << std::endl;
    else
    {
        std::vector<std::string>postFiltersString = FilterProcessor::filterToString(param.postFilters);
        if (postFiltersString.empty())
            oss << "Postprocessing disabled" << std::endl;
        else
            for (auto& filters : postFiltersString)
                oss << filters << std::endl;
    }

    return oss.str();
}

void Anime4KCPP::Cuda::Anime4K09::processYUVImage()
{
    cv::Mat tmpImg;
    cv::merge(std::vector<cv::Mat>{ orgImg, orgU, orgV }, tmpImg);
    cv::cvtColor(tmpImg, tmpImg, cv::COLOR_YUV2BGR);

    dstImg.create(height, width, CV_MAKE_TYPE(tmpImg.depth(), 4));
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(tmpImg, param.preFilters).process();
    cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2BGRA);
    detail::runKernel(tmpImg, dstImg, param);
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
    if (param.postprocessing)//Postprocessing(CPU)
        FilterProcessor(dstImg, param.postFilters).process();

    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGR2YUV);
    std::vector<cv::Mat> yuv(3);
    cv::split(dstImg, yuv);
    dstImg = yuv[Y];
    dstU = yuv[U];
    dstV = yuv[V];
}

void Anime4KCPP::Cuda::Anime4K09::processRGBImage()
{
    cv::Mat tmpImg = orgImg;
    dstImg.create(height, width, CV_MAKE_TYPE(tmpImg.depth(), 4));
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(tmpImg, param.preFilters).process();
    cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2BGRA);
    detail::runKernel(tmpImg, dstImg, param);
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
    if (param.postprocessing)//Postprocessing(CPU)
        FilterProcessor(dstImg, param.postFilters).process();
}

void Anime4KCPP::Cuda::Anime4K09::processGrayscale()
{
    cv::Mat tmpImg;
    cv::cvtColor(orgImg, tmpImg, cv::COLOR_GRAY2BGR);

    dstImg.create(height, width, CV_MAKE_TYPE(tmpImg.depth(), 4));
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(tmpImg, param.preFilters).process();
    cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2BGRA);
    detail::runKernel(tmpImg, dstImg, param);
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
    if (param.postprocessing)//Postprocessing(CPU)
        FilterProcessor(dstImg, param.postFilters).process();

    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGR2GRAY);
}

Anime4KCPP::Processor::Type Anime4KCPP::Cuda::Anime4K09::getProcessorType() noexcept
{
    return Processor::Type::Cuda_Anime4K09;
}

std::string Anime4KCPP::Cuda::Anime4K09::getProcessorInfo()
{
    std::ostringstream oss;
    oss << "Processor type: " << getProcessorType() << std::endl
        << "Current CUDA devices:" << std::endl
        << cuGetDeviceInfo(cuGetDeviceID());
    return oss.str();
}

#endif
