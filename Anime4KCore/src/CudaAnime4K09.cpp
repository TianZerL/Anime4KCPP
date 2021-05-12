#ifdef ENABLE_CUDA

#define DLL

#include"CudaInterface.hpp"
#include"FilterProcessor.hpp"
#include"CudaAnime4K09.hpp"

Anime4KCPP::Cuda::Anime4K09::Anime4K09(const Parameters& parameters) :
    AC(parameters) {}

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

inline void Anime4KCPP::Cuda::Anime4K09::runKernelB(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    ACCudaParamAnime4K09 cuParam{
        orgImg.cols, orgImg.rows,
        dstImg.cols, dstImg.rows,
        orgImg.step,
        param.passes, param.pushColorCount,
        static_cast<float>(param.strengthColor),static_cast<float>(param.strengthGradient)
    };
    cuRunKernelAnime4K09B(orgImg.data, dstImg.data, &cuParam);
}

inline void Anime4KCPP::Cuda::Anime4K09::runKernelW(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    ACCudaParamAnime4K09 cuParam{
        orgImg.cols, orgImg.rows,
        dstImg.cols, dstImg.rows,
        orgImg.step,
        param.passes, param.pushColorCount,
        static_cast<float>(param.strengthColor),static_cast<float>(param.strengthGradient)
    };
    cuRunKernelAnime4K09W(reinterpret_cast<const unsigned short*>(orgImg.data), reinterpret_cast<unsigned short*>(dstImg.data), &cuParam);
}

inline void Anime4KCPP::Cuda::Anime4K09::runKernelF(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    ACCudaParamAnime4K09 cuParam{
        orgImg.cols, orgImg.rows,
        dstImg.cols, dstImg.rows,
        orgImg.step,
        param.passes, param.pushColorCount,
        static_cast<float>(param.strengthColor),static_cast<float>(param.strengthGradient)
    };
    cuRunKernelAnime4K09F(reinterpret_cast<const float*>(orgImg.data), reinterpret_cast<float*>(dstImg.data), &cuParam);
}

void Anime4KCPP::Cuda::Anime4K09::processYUVImageB()
{
    cv::merge(std::vector<cv::Mat>{ orgY, orgU, orgV }, orgImg);
    cv::cvtColor(orgImg, orgImg, cv::COLOR_YUV2BGR);

    dstImg.create(H, W, CV_8UC4);
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(orgImg, param.preFilters).process();
    cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2BGRA);
    runKernelB(orgImg, dstImg);
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

void Anime4KCPP::Cuda::Anime4K09::processRGBImageB()
{
    dstImg.create(H, W, CV_8UC4);
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(orgImg, param.preFilters).process();
    cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2BGRA);
    runKernelB(orgImg, dstImg);
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
    if (param.postprocessing)//Postprocessing(CPU)
        FilterProcessor(dstImg, param.postFilters).process();
}

void Anime4KCPP::Cuda::Anime4K09::processGrayscaleB()
{
    cv::Mat tmpImg;
    cv::cvtColor(orgImg, tmpImg, cv::COLOR_GRAY2BGR);

    dstImg.create(H, W, CV_8UC4);
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(tmpImg, param.preFilters).process();
    cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2BGRA);
    runKernelB(tmpImg, dstImg);
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
    if (param.postprocessing)//Postprocessing(CPU)
        FilterProcessor(dstImg, param.postFilters).process();

    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGR2GRAY);
}

void Anime4KCPP::Cuda::Anime4K09::processYUVImageW()
{
    cv::merge(std::vector<cv::Mat>{ orgY, orgU, orgV }, orgImg);
    cv::cvtColor(orgImg, orgImg, cv::COLOR_YUV2BGR);

    dstImg.create(H, W, CV_16UC4);
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(orgImg, param.preFilters).process();
    cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2BGRA);
    runKernelW(orgImg, dstImg);
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

void Anime4KCPP::Cuda::Anime4K09::processRGBImageW()
{
    dstImg.create(H, W, CV_16UC4);
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(orgImg, param.preFilters).process();
    cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2BGRA);
    runKernelW(orgImg, dstImg);
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
    if (param.postprocessing)//Postprocessing(CPU)
        FilterProcessor(dstImg, param.postFilters).process();
}

void Anime4KCPP::Cuda::Anime4K09::processGrayscaleW()
{
    cv::Mat tmpImg;
    cv::cvtColor(orgImg, tmpImg, cv::COLOR_GRAY2BGR);

    dstImg.create(H, W, CV_16UC4);
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(tmpImg, param.preFilters).process();
    cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2BGRA);
    runKernelW(tmpImg, dstImg);
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
    if (param.postprocessing)//Postprocessing(CPU)
        FilterProcessor(dstImg, param.postFilters).process();

    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGR2GRAY);
}

void Anime4KCPP::Cuda::Anime4K09::processYUVImageF()
{
    cv::merge(std::vector<cv::Mat>{ orgY, orgU, orgV }, orgImg);
    cv::cvtColor(orgImg, orgImg, cv::COLOR_YUV2BGR);

    dstImg.create(H, W, CV_32FC4);
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(orgImg, param.preFilters).process();
    cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2BGRA);
    runKernelF(orgImg, dstImg);
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

void Anime4KCPP::Cuda::Anime4K09::processRGBImageF()
{
    dstImg.create(H, W, CV_32FC4);
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(orgImg, param.preFilters).process();
    cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2BGRA);
    runKernelF(orgImg, dstImg);
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
    if (param.postprocessing)//Postprocessing(CPU)
        FilterProcessor(dstImg, param.postFilters).process();
}

void Anime4KCPP::Cuda::Anime4K09::processGrayscaleF()
{
    cv::Mat tmpImg;
    cv::cvtColor(orgImg, tmpImg, cv::COLOR_GRAY2BGR);

    dstImg.create(H, W, CV_32FC4);
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(tmpImg, param.preFilters).process();
    cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2BGRA);
    runKernelF(tmpImg, dstImg);
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
