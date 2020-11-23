#ifdef ENABLE_CUDA

#define DLL

#include "CudaACNet.hpp"

Anime4KCPP::Cuda::ACNet::ACNet(const Parameters& parameters) :
    AC(parameters) {}

std::string Anime4KCPP::Cuda::ACNet::getInfo()
{
    std::ostringstream oss;
    oss << AC::getInfo()
        << "----------------------------------------------" << std::endl
        << "Zoom Factor: " << param.zoomFactor << std::endl
        << "HDN Mode: " << std::boolalpha << param.HDN << std::endl
        << "HDN level: " << (param.HDN ? param.HDNLevel : 0) << std::endl
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

inline void Anime4KCPP::Cuda::ACNet::runKernel(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    ACCudaParamACNet cuParam{ orgImg.cols, orgImg.rows,(param.HDN ? param.HDNLevel : 0) };
    cuRunKernelACNet(orgImg.data, dstImg.data, &cuParam);
}

void Anime4KCPP::Cuda::ACNet::processYUVImage()
{
    if (!param.fastMode)
    {
        double tmpZf = std::log2(param.zoomFactor);
        if (tmpZf < 0.0001)
            tmpZf = 1.0 - 0.0002;
        int tmpZfUp = std::ceil(tmpZf);

        cv::Mat tmpY = orgY;
        dstU = orgU;
        dstV = orgV;
        for (int i = 0; i < tmpZfUp; i++)
        {
            dstY.create(tmpY.rows * 2, tmpY.cols * 2, CV_8UC1);
            runKernel(tmpY, dstY);

            cv::resize(dstU, dstU, cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
            cv::resize(dstV, dstV, cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
            tmpY = dstY;
        }
        if (tmpZfUp - tmpZf > 0.00001)
        {
            double currZf = param.zoomFactor / exp2(tmpZfUp);
            cv::resize(dstY, dstY, cv::Size(0, 0), currZf, currZf, cv::INTER_AREA);
            cv::resize(dstU, dstU, cv::Size(0, 0), currZf, currZf, cv::INTER_AREA);
            cv::resize(dstV, dstV, cv::Size(0, 0), currZf, currZf, cv::INTER_AREA);
        }
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgY, orgY, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_LANCZOS4);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgY, orgY, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        dstY.create(orgY.rows * 2, orgY.cols * 2, CV_8UC1);
        runKernel(orgY, dstY);

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_LANCZOS4);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_LANCZOS4);
    }
}

void Anime4KCPP::Cuda::ACNet::processRGBImage()
{
    if (!param.fastMode)
    {
        double tmpZf = std::log2(param.zoomFactor);
        if (tmpZf < 0.0001)
            tmpZf = 1.0 - 0.0002;
        int tmpZfUp = std::ceil(tmpZf);

        cv::Mat tmpImg = orgImg;
        cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(tmpImg, yuv);
        tmpImg = yuv[Y];

        for (int i = 0; i < tmpZfUp; i++)
        {
            dstImg.create(tmpImg.rows * 2, tmpImg.cols * 2, CV_8UC1);
            runKernel(tmpImg, dstImg);

            cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
            cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
            tmpImg = dstImg;
        }

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
        if (tmpZfUp - tmpZf > 0.00001)
        {
            cv::resize(dstImg, dstImg, cv::Size(W, H), 0, 0, cv::INTER_AREA);
        }
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_LANCZOS4);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(orgImg, yuv);
        orgImg = yuv[Y];

        dstImg.create(orgImg.rows * 2, orgImg.cols * 2, CV_8UC1);
        runKernel(orgImg, dstImg);

        cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
        cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
}

void Anime4KCPP::Cuda::ACNet::processRGBVideo()
{
    if (!param.fastMode)
    {
        double tmpZf = std::log2(param.zoomFactor);
        if (tmpZf < 0.0001)
            tmpZf = 1.0 - 0.0002;
        int tmpZfUp = std::ceil(tmpZf);

        videoIO->init(
            [this, tmpZfUp, tmpZf]()
            {
                Utils::Frame frame = videoIO->read();
                cv::Mat orgFrame = frame.first;
                cv::Mat dstFrame;

                cv::Mat tmpFrame = orgFrame;
                cv::cvtColor(tmpFrame, tmpFrame, cv::COLOR_BGR2YUV);

                std::vector<cv::Mat> yuv(3);
                cv::split(tmpFrame, yuv);
                tmpFrame = yuv[Y];

                for (int i = 0; i < tmpZfUp; i++)
                {
                    dstFrame.create(tmpFrame.rows * 2, tmpFrame.cols * 2, CV_8UC1);
                    runKernel(tmpFrame, dstFrame);

                    cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
                    cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
                    tmpFrame = dstFrame;
                }

                cv::merge(std::vector<cv::Mat>{ dstFrame, yuv[U], yuv[V] }, dstFrame);
                cv::cvtColor(dstFrame, dstFrame, cv::COLOR_YUV2BGR);
                if (tmpZfUp - tmpZf > 0.00001)
                {
                    cv::resize(dstFrame, dstFrame, cv::Size(W, H), 0, 0, cv::INTER_AREA);
                }

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
                    cv::resize(orgFrame, orgFrame, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_LANCZOS4);
                else if (param.zoomFactor < 2.0)
                    cv::resize(orgFrame, orgFrame, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

                cv::cvtColor(orgFrame, orgFrame, cv::COLOR_BGR2YUV);

                std::vector<cv::Mat> yuv(3);
                cv::split(orgFrame, yuv);
                orgFrame = yuv[Y];

                dstFrame.create(orgFrame.rows * 2, orgFrame.cols * 2, CV_8UC1);
                runKernel(orgFrame, dstFrame);

                cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
                cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);

                cv::merge(std::vector<cv::Mat>{ dstFrame, yuv[U], yuv[V] }, dstFrame);
                cv::cvtColor(dstFrame, dstFrame, cv::COLOR_YUV2BGR);

                frame.first = dstFrame;
                videoIO->write(frame);
            }
            , param.maxThreads
                ).process();
    }
}

Anime4KCPP::Processor::Type Anime4KCPP::Cuda::ACNet::getProcessorType() noexcept
{
    return Processor::Type::Cuda_ACNet;
}

#endif
