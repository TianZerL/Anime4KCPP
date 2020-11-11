#ifdef ENABLE_CUDA

#define DLL

#include "CudaAnime4K09.hpp"

Anime4KCPP::Cuda::Anime4K09::Anime4K09(const Parameters& parameters) :
    AC(parameters) {}

std::string Anime4KCPP::Cuda::Anime4K09::getInfo()
{
    std::ostringstream oss;
    oss << AC::getInfo()
        << "----------------------------------------------" << std::endl
        << "Passes: " << param.passes << std::endl
        << "pushColorCount: " << param.pushColorCount << std::endl
        << "Zoom Factor: " << param.zoomFactor << std::endl
        << "Video Mode: " << std::boolalpha << param.videoMode << std::endl
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

inline void Anime4KCPP::Cuda::Anime4K09::runKernel(cv::Mat& orgImg, cv::Mat& dstImg)
{
    ACCudaParamAnime4K09 cuParam{
        orgImg.cols, orgImg.rows,
        dstImg.cols, dstImg.rows,
        param.passes, param.pushColorCount,
        static_cast<float>(param.strengthColor),static_cast<float>(param.strengthGradient)
    };
    cuRunKernelAnime4K09(orgImg.data, dstImg.data, &cuParam);
}

void Anime4KCPP::Cuda::Anime4K09::processYUVImage()
{
    cv::merge(std::vector<cv::Mat>{ orgY, orgU, orgV }, orgImg);
    cv::cvtColor(orgImg, orgImg, cv::COLOR_YUV2BGR);

    dstImg.create(H, W, CV_8UC4);
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(orgImg, param.preFilters).process();
    cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2BGRA);
    runKernel(orgImg, dstImg);
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
    if (param.postprocessing)//Postprocessing(CPU)
        FilterProcessor(dstImg, param.postFilters).process();

    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGR2YUV);
    std::vector<cv::Mat> yuv(3);
    cv::split(dstImg, yuv);
    dstY = yuv[Y];
    dstU = yuv[U];
    dstV = yuv[V];
}

void Anime4KCPP::Cuda::Anime4K09::processRGBImage()
{
    dstImg.create(H, W, CV_8UC4);
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(orgImg, param.preFilters).process();
    cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2BGRA);
    runKernel(orgImg, dstImg);
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
    if (param.postprocessing)//Postprocessing(CPU)
        FilterProcessor(dstImg, param.postFilters).process();
}

void Anime4KCPP::Cuda::Anime4K09::processRGBVideo()
{
    videoIO->init(
        [this]()
        {
            Utils::Frame frame = videoIO->read();
            cv::Mat orgFrame = frame.first;
            cv::Mat dstFrame(H, W, CV_8UC4);
            if (param.preprocessing)
                FilterProcessor(orgFrame, param.preFilters).process();
            cv::cvtColor(orgFrame, orgFrame, cv::COLOR_BGR2BGRA);
            runKernel(orgFrame, dstFrame);
            cv::cvtColor(dstFrame, dstFrame, cv::COLOR_BGRA2BGR);
            if (param.postprocessing)//PostProcessing
                FilterProcessor(dstFrame, param.postFilters).process();
            frame.first = dstFrame;
            videoIO->write(frame);
        }
        , param.maxThreads
            ).process();
}

Anime4KCPP::Processor::Type Anime4KCPP::Cuda::Anime4K09::getProcessorType() noexcept
{
    return Processor::Type::Cuda_Anime4K09;
}

#endif
