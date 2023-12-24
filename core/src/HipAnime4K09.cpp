#ifdef ENABLE_HIP

#include "HipInterface.hpp"
#include "FilterProcessor.hpp"
#include "HipAnime4K09.hpp"

namespace Anime4KCPP::Hip::detail
{
    static void runKernel(const cv::Mat& orgImg, cv::Mat& dstImg, const Parameters& param, hipStream_t& stream)
    {
        ACHipParamAnime4K09 cuParam{
        orgImg.cols, orgImg.rows,
        dstImg.cols, dstImg.rows,
        orgImg.step,
        param.passes, param.pushColorCount,
        static_cast<float>(param.strengthColor),static_cast<float>(param.strengthGradient)
        };

        ACHipDataType dataType;

        switch (orgImg.depth())
        {
        case CV_8U:
            dataType = ACHipDataType::AC_8U;
            break;
        case CV_16U:
            dataType = ACHipDataType::AC_16U;
            break;
        case CV_32F:
            dataType = ACHipDataType::AC_32F;
            break;
        default:
            throw ACException<ExceptionType::RunTimeError>("Unsupported image data type");
        }

        cuRunKernelAnime4K09(orgImg.data, dstImg.data, dataType, &cuParam, stream);
    }
}

namespace Anime4KCPP::Hip 
{
	Anime4K09 :: Anime4K09 (const Parameters& parameters) : Anime4KCPP::AC(parameters)
	{
		hipStreamCreate(&stream);
	}

	Anime4K09 :: ~Anime4K09()
	{
		hipStreamDestroy(stream);
	}
}

std::string Anime4KCPP::Hip::Anime4K09::getInfo() const
{
    std::ostringstream oss;
    oss << AC::getInfo()
        << "----------------------------------------------" << '\n'
        << "HIP Device ID:" << cuGetDeviceID() << '\n'
        << "Passes: " << param.passes << '\n'
        << "pushColorCount: " << param.pushColorCount << '\n'
        << "Zoom Factor: " << param.zoomFactor << '\n'
        << "Fast Mode: " << std::boolalpha << param.fastMode << '\n'
        << "Strength Color: " << param.strengthColor << '\n'
        << "Strength Gradient: " << param.strengthGradient << '\n'
        << "----------------------------------------------" << '\n';
    return oss.str();
}

std::string Anime4KCPP::Hip::Anime4K09::getFiltersInfo() const
{
    std::ostringstream oss;

    oss << AC::getFiltersInfo()
        << "----------------------------------------------" << '\n'
        << "Preprocessing filters list:" << '\n'
        << "----------------------------------------------" << '\n';
    if (!param.preprocessing)
        oss << "Preprocessing disabled" << '\n';
    else
    {
        std::vector<std::string>preFiltersString = FilterProcessor::filterToString(param.preFilters);
        if (preFiltersString.empty())
            oss << "Preprocessing disabled" << '\n';
        else
            for (auto& filters : preFiltersString)
                oss << filters << '\n';
    }

    oss << "----------------------------------------------" << '\n'
        << "Postprocessing filters list:" << '\n'
        << "----------------------------------------------" << '\n';
    if (!param.postprocessing)
        oss << "Postprocessing disabled" << '\n';
    else
    {
        std::vector<std::string>postFiltersString = FilterProcessor::filterToString(param.postFilters);
        if (postFiltersString.empty())
            oss << "Postprocessing disabled" << '\n';
        else
            for (auto& filters : postFiltersString)
                oss << filters << '\n';
    }

    return oss.str();
}

void Anime4KCPP::Hip::Anime4K09::processYUVImage()
{
    cv::Mat tmpImg;
    cv::merge(std::vector<cv::Mat>{ orgImg, orgU, orgV }, tmpImg);
    cv::cvtColor(tmpImg, tmpImg, cv::COLOR_YUV2BGR);

    dstImg.create(height, width, CV_MAKE_TYPE(tmpImg.depth(), 4));
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(tmpImg, param.preFilters).process();
    cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2BGRA);
    detail::runKernel(tmpImg, dstImg, param, stream);
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
    if (param.postprocessing)//Postprocessing(CPU)
        FilterProcessor(dstImg, param.postFilters).process();

    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGR2YUV);
    cv::Mat yuv[3];
    cv::split(dstImg, yuv);
    dstImg = yuv[Y];
    dstU = yuv[U];
    dstV = yuv[V];
}

void Anime4KCPP::Hip::Anime4K09::processRGBImage()
{
    cv::Mat tmpImg = orgImg;
    dstImg.create(height, width, CV_MAKE_TYPE(tmpImg.depth(), 4));
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(tmpImg, param.preFilters).process();
    cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2BGRA);
    detail::runKernel(tmpImg, dstImg, param, stream);
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
    if (param.postprocessing)//Postprocessing(CPU)
        FilterProcessor(dstImg, param.postFilters).process();
}

void Anime4KCPP::Hip::Anime4K09::processGrayscale()
{
    cv::Mat tmpImg;
    cv::cvtColor(orgImg, tmpImg, cv::COLOR_GRAY2BGR);

    dstImg.create(height, width, CV_MAKE_TYPE(tmpImg.depth(), 4));
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(tmpImg, param.preFilters).process();
    cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2BGRA);
    detail::runKernel(tmpImg, dstImg, param, stream);
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
    if (param.postprocessing)//Postprocessing(CPU)
        FilterProcessor(dstImg, param.postFilters).process();

    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGR2GRAY);
}

Anime4KCPP::Processor::Type Anime4KCPP::Hip::Anime4K09::getProcessorType() const noexcept
{
    return Processor::Type::Hip_Anime4K09;
}

std::string Anime4KCPP::Hip::Anime4K09::getProcessorInfo() const
{
    std::ostringstream oss;
    oss << "Processor type: " << getProcessorType() << '\n'
        << "Current HIP devices:" << '\n'
        << cuGetDeviceInfo(cuGetDeviceID());
    return oss.str();
}

#endif
