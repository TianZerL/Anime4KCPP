#define DLL

#include "Anime4KCPUCNN.h"

Anime4KCPP::Anime4KCPUCNN::Anime4KCPUCNN(const Parameters& parameters) :
    Anime4K(parameters) {}

void Anime4KCPP::Anime4KCPUCNN::process()
{
    double tmpZf = log2(zf);
    if (tmpZf < 0.0001)
        tmpZf = 1.0- 0.0002;
    int tmpZfUp = ceil(tmpZf);
    CNNProcessor* processor;
    if (HDN)
    {
        processor = CNNCreator::create(CNNType::ACNetHDN);
    }
    else
    {
        processor = CNNCreator::create(CNNType::ACNet);
    }
    
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
                cv::resize(dstImg, dstImg, cv::Size(W, H), 0, 0, cv::INTER_LANCZOS4);
            }
        }
        else
        {
            cv::Mat tmpImg = orgImg;
            for (int i = 0; i < tmpZfUp; i++)
            {
                processor->process(tmpImg, dstImg);

                std::vector<cv::Mat> yuv(3);
                cv::resize(tmpImg, tmpImg, cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
                cv::split(tmpImg, yuv);
                cv::merge(std::vector{ dstImg,yuv[U],yuv[V] }, dstImg);
                tmpImg = dstImg;
            }
            if (tmpZfUp - tmpZf > 0.00001)
            {
                cv::resize(dstImg, dstImg, cv::Size(W, H), 0, 0, cv::INTER_LANCZOS4);
            }
        }
    }
    else
    {
        VideoIO::instance().init(
            [this, tmpZfUp, tmpZf, processor]()
            {
                Frame frame = VideoIO::instance().read();
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
                    cv::resize(dstFrame, dstFrame, cv::Size(W, H), 0, 0, cv::INTER_LANCZOS4);
                }
                frame.first = dstFrame;
                VideoIO::instance().write(frame);
            }
            , mt
                ).process();
    }
    CNNCreator::release(processor);
}
