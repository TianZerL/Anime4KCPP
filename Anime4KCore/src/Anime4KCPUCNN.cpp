#define DLL

#include "Anime4KCPUCNN.h"

Anime4KCPP::Anime4KCPUCNN::Anime4KCPUCNN(const Parameters& parameters) :
    Anime4K(parameters) {}

void Anime4KCPP::Anime4KCPUCNN::process()
{
    CNNProcessor* processor;
    if (HDN)
    {
        processor = CNNCreator::create(CNNType::ACNetHDN);
    }
    else
    {
        processor = CNNCreator::create(CNNType::ACNet);
    }

    if (fm)
    {
        if (!vm)
        {
            if (!inputYUV)
            {
                if (zf > 2.0)
                    cv::resize(orgImg, orgImg, cv::Size(0, 0), zf / 2.0, zf / 2.0, cv::INTER_LANCZOS4);
                else if (zf < 2.0)
                    cv::resize(orgImg, orgImg, cv::Size(0, 0), zf / 2.0, zf / 2.0, cv::INTER_AREA);
                cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2YUV);

                processor->process(orgImg, dstImg);
                std::vector<cv::Mat> yuv(3);
                cv::resize(orgImg, orgImg, cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
                cv::split(orgImg, yuv);
                cv::merge(std::vector{ dstImg,yuv[U],yuv[V] }, dstImg);

                cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
            }
            else
            {
                if (zf > 2.0)
                    cv::resize(orgY, orgY, cv::Size(0, 0), zf / 2.0, zf / 2.0, cv::INTER_LANCZOS4);
                else if (zf < 2.0)
                    cv::resize(orgY, orgY, cv::Size(0, 0), zf / 2.0, zf / 2.0, cv::INTER_AREA);

                processor->process(orgY, dstY);

                cv::resize(orgU, dstU, cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
                cv::resize(orgV, dstV, cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
            }
        }
        else
        {
            videoIO->init(
                [this, processor]()
                {
                    Frame frame = videoIO->read();
                    cv::Mat orgFrame = frame.first;
                    cv::Mat dstFrame;

                    if (zf > 2.0)
                        cv::resize(orgFrame, orgFrame, cv::Size(0, 0), zf / 2.0, zf / 2.0, cv::INTER_LANCZOS4);
                    else if (zf < 2.0)
                        cv::resize(orgFrame, orgFrame, cv::Size(0, 0), zf / 2.0, zf / 2.0, cv::INTER_AREA);
                    cv::cvtColor(orgFrame, orgFrame, cv::COLOR_BGR2YUV);

                    processor->process(orgFrame, dstFrame);
                    std::vector<cv::Mat> yuv(3);
                    cv::resize(orgFrame, orgFrame, cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
                    cv::split(orgFrame, yuv);
                    cv::merge(std::vector{ dstFrame,yuv[U],yuv[V] }, dstFrame);

                    cv::cvtColor(dstFrame, dstFrame, cv::COLOR_YUV2BGR);
                    frame.first = dstFrame;
                    videoIO->write(frame);
                }
                , mt
                    ).process();
        }
    }
    else
    {
        double tmpZf = log2(zf);
        if (tmpZf < 0.0001)
            tmpZf = 1.0 - 0.0002;
        int tmpZfUp = ceil(tmpZf);
        if (!vm)
        {
            if (!inputYUV)
            {
                cv::Mat tmpImg = orgImg;
                cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2YUV);
                for (int i = 0; i < tmpZfUp; i++)
                {
                    processor->process(tmpImg, dstImg);

                    std::vector<cv::Mat> yuv(3);
                    cv::resize(tmpImg, tmpImg, cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
                    cv::split(tmpImg, yuv);
                    cv::merge(std::vector{ dstImg,yuv[U],yuv[V] }, dstImg);
                    tmpImg = dstImg;
                }
                cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
                if (tmpZfUp - tmpZf > 0.00001)
                {
                    cv::resize(dstImg, dstImg, cv::Size(W, H), 0, 0, cv::INTER_AREA);
                }
            }
            else
            {
                cv::Mat tmpY = orgY;
                dstU = orgU;
                dstV = orgV;
                for (int i = 0; i < tmpZfUp; i++)
                {
                    processor->process(tmpY, dstY);

                    cv::resize(dstU, dstU, cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
                    cv::resize(dstV, dstV, cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
                    tmpY = dstY;
                }
                if (tmpZfUp - tmpZf > 0.00001)
                {
                    double currZf = zf / exp2(tmpZfUp);
                    cv::resize(dstY, dstY, cv::Size(0, 0), currZf, currZf, cv::INTER_AREA);
                    cv::resize(dstU, dstU, cv::Size(0, 0), currZf, currZf, cv::INTER_AREA);
                    cv::resize(dstV, dstV, cv::Size(0, 0), currZf, currZf, cv::INTER_AREA);
                }
            }
        }
        else
        {
            videoIO->init(
                [this, tmpZfUp, tmpZf, processor]()
                {
                    Frame frame = videoIO->read();
                    cv::Mat orgFrame = frame.first;
                    cv::Mat dstFrame;

                    cv::Mat tmpFrame = orgFrame;
                    cv::cvtColor(tmpFrame, tmpFrame, cv::COLOR_BGR2YUV);
                    for (int i = 0; i < tmpZfUp; i++)
                    {
                        processor->process(tmpFrame, dstFrame);

                        std::vector<cv::Mat> yuv(3);
                        cv::resize(tmpFrame, tmpFrame, cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
                        cv::split(tmpFrame, yuv);
                        cv::merge(std::vector{ dstFrame,yuv[U],yuv[V] }, dstFrame);
                        tmpFrame = dstFrame;
                    }
                    cv::cvtColor(dstFrame, dstFrame, cv::COLOR_YUV2BGR);
                    if (tmpZfUp - tmpZf > 0.00001)
                    {
                        cv::resize(dstFrame, dstFrame, cv::Size(W, H), 0, 0, cv::INTER_AREA);
                    }
                    frame.first = dstFrame;
                    videoIO->write(frame);
                }
                , mt
                    ).process();
        }
    }
    CNNCreator::release(processor);
}

Anime4KCPP::ProcessorType Anime4KCPP::Anime4KCPUCNN::getProcessorType()
{
    return ProcessorType::CPUCNN;
}
