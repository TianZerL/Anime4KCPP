#define DLL

#include "CPUACNet.hpp"

Anime4KCPP::CPU::ACNet::ACNet(const Parameters& parameters) :
    AC(parameters) 
{
    if (param.HDN)
        switch (param.HDNLevel)
        {
        case 1:
            processor = createACNetProcessor(CNNType::ACNetHDNL1);
            break;
        case 2:
            processor = createACNetProcessor(CNNType::ACNetHDNL2);
            break;
        case 3:
            processor = createACNetProcessor(CNNType::ACNetHDNL3);
            break;
        default:
            processor = createACNetProcessor(CNNType::ACNetHDNL1);
            break;
        }
    else
        processor = createACNetProcessor(CNNType::ACNetHDNL0);
}

Anime4KCPP::CPU::ACNet::~ACNet()
{
    releaseACNetProcessor(processor);
}

void Anime4KCPP::CPU::ACNet::setArguments(const Parameters& parameters)
{
    AC::setArguments(parameters);
    releaseACNetProcessor(processor);
    if (param.HDN)
        switch (param.HDNLevel)
        {
        case 1:
            processor = createACNetProcessor(CNNType::ACNetHDNL1);
            break;
        case 2:
            processor = createACNetProcessor(CNNType::ACNetHDNL2);
            break;
        case 3:
            processor = createACNetProcessor(CNNType::ACNetHDNL3);
            break;
        default:
            processor = createACNetProcessor(CNNType::ACNetHDNL1);
            break;
        }
    else
        processor = createACNetProcessor(CNNType::ACNetHDNL0);
}

std::string Anime4KCPP::CPU::ACNet::getInfo()
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

std::string Anime4KCPP::CPU::ACNet::getFiltersInfo()
{
    std::ostringstream oss;
    oss << AC::getFiltersInfo()
        << "----------------------------------------------" << std::endl
        << "Filter not supported" << std::endl
        << "----------------------------------------------" << std::endl;
    return oss.str();
}

void Anime4KCPP::CPU::ACNet::processYUVImageB()
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
            processor->processB(tmpY, dstY);

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

        processor->processB(orgY, dstY);

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_LANCZOS4);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_LANCZOS4);
    }
}

void Anime4KCPP::CPU::ACNet::processRGBImageB()
{
    if (!param.fastMode)
    {
        double tmpZf = std::log2(param.zoomFactor);
        if (tmpZf < 0.0001)
            tmpZf = 1.0 - 0.0002;
        int tmpZfUp = std::ceil(tmpZf);

        cv::Mat tmpImg = orgImg;
        cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2YUV);

        for (int i = 0; i < tmpZfUp; i++)
        {
            processor->processB(tmpImg, dstImg);

            cv::resize(orgImg, orgImg, cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
            tmpImg = dstImg;
        }

        cv::mixChannels(dstImg, orgImg, std::vector<int>{0, 0});
        dstImg = orgImg;

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

        processor->processB(orgImg, dstImg);
        std::vector<cv::Mat> yuv(3);
        cv::resize(orgImg, orgImg, cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
        cv::mixChannels(dstImg, orgImg, std::vector<int>{0, 0});
        dstImg = orgImg;
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
}

void Anime4KCPP::CPU::ACNet::processRGBVideoB()
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

                for (int i = 0; i < tmpZfUp; i++)
                {
                    processor->processB(tmpFrame, dstFrame);

                    cv::resize(orgFrame, orgFrame, cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
                    tmpFrame = dstFrame;
                }

                cv::mixChannels(dstFrame, orgFrame, std::vector<int>{0, 0});
                dstFrame = orgFrame;

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

                processor->processB(orgFrame, dstFrame);
                std::vector<cv::Mat> yuv(3);
                cv::resize(orgFrame, orgFrame, cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
                cv::mixChannels(dstFrame, orgFrame, std::vector<int>{0, 0});
                dstFrame = orgFrame;

                cv::cvtColor(dstFrame, dstFrame, cv::COLOR_YUV2BGR);
                frame.first = dstFrame;
                videoIO->write(frame);
            }
            , param.maxThreads
                ).process();
    }
}

void Anime4KCPP::CPU::ACNet::processYUVImageF()
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
            processor->processF(tmpY, dstY);

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

        processor->processF(orgY, dstY);

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_LANCZOS4);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_LANCZOS4);
    }
}

void Anime4KCPP::CPU::ACNet::processRGBImageF()
{
    if (!param.fastMode)
    {
        double tmpZf = std::log2(param.zoomFactor);
        if (tmpZf < 0.0001)
            tmpZf = 1.0 - 0.0002;
        int tmpZfUp = std::ceil(tmpZf);

        cv::Mat tmpImg = orgImg;
        cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2YUV);

        for (int i = 0; i < tmpZfUp; i++)
        {
            processor->processF(tmpImg, dstImg);

            cv::resize(orgImg, orgImg, cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
            tmpImg = dstImg;
        }

        cv::mixChannels(dstImg, orgImg, std::vector<int>{0, 0});
        dstImg = orgImg;

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

        processor->processF(orgImg, dstImg);
        std::vector<cv::Mat> yuv(3);
        cv::resize(orgImg, orgImg, cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
        cv::mixChannels(dstImg, orgImg, std::vector<int>{0, 0});
        dstImg = orgImg;
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
}

Anime4KCPP::Processor::Type Anime4KCPP::CPU::ACNet::getProcessorType() noexcept
{
    return Processor::Type::CPU_ACNet;
}
