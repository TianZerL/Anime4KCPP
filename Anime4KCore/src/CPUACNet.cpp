#define DLL

#include"CPUACNet.hpp"

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

void Anime4KCPP::CPU::ACNet::setParameters(const Parameters& parameters)
{
    AC::setParameters(parameters);
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
        << "HDN Level: " << (param.HDN ? param.HDNLevel : 0) << std::endl
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
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        processor->processB(orgY, dstY, scaleTimes);

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

        processor->processB(orgY, dstY, 1);

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
    }
}

void Anime4KCPP::CPU::ACNet::processRGBImageB()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        cv::Mat tmpImg;
        cv::cvtColor(orgImg, tmpImg, cv::COLOR_BGR2YUV);

        processor->processB(tmpImg, dstImg , scaleTimes);

        if (param.isNonIntegerScale())
        {
            cv::resize(dstImg, dstImg, cv::Size(W, H), param.zoomFactor, param.zoomFactor, cv::INTER_AREA);
        }

        cv::resize(tmpImg, tmpImg, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::mixChannels(dstImg, tmpImg, std::vector<int>{0, 0});

        cv::cvtColor(tmpImg, dstImg, cv::COLOR_YUV2BGR);
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);
        
        cv::Mat tmpImg;
        cv::cvtColor(orgImg, tmpImg, cv::COLOR_BGR2YUV);

        processor->processB(tmpImg, dstImg, 1);

        cv::resize(tmpImg, tmpImg, cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
        cv::mixChannels(dstImg, tmpImg, std::vector<int>{0, 0});

        cv::cvtColor(tmpImg, dstImg, cv::COLOR_YUV2BGR);
    }
}

void Anime4KCPP::CPU::ACNet::processGrayscaleB()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        processor->processB(orgImg, dstImg, scaleTimes);

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

        processor->processB(orgImg, dstImg, 1);
    }
}

void Anime4KCPP::CPU::ACNet::processYUVImageW()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        processor->processW(orgY, dstY, scaleTimes);

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

        processor->processW(orgY, dstY, 1);

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
    }
}

void Anime4KCPP::CPU::ACNet::processRGBImageW()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        cv::Mat tmpImg;
        cv::cvtColor(orgImg, tmpImg, cv::COLOR_BGR2YUV);

        processor->processW(tmpImg, dstImg, scaleTimes);

        if (param.isNonIntegerScale())
        {
            cv::resize(dstImg, dstImg, cv::Size(W, H), param.zoomFactor, param.zoomFactor, cv::INTER_AREA);
        }

        cv::resize(tmpImg, tmpImg, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::mixChannels(dstImg, tmpImg, std::vector<int>{0, 0});

        cv::cvtColor(tmpImg, dstImg, cv::COLOR_YUV2BGR);
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);
        
        cv::Mat tmpImg;
        cv::cvtColor(orgImg, tmpImg, cv::COLOR_BGR2YUV);

        processor->processW(tmpImg, dstImg, 1);

        cv::resize(tmpImg, tmpImg, cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
        cv::mixChannels(dstImg, tmpImg, std::vector<int>{0, 0});

        cv::cvtColor(tmpImg, dstImg, cv::COLOR_YUV2BGR);
    }
}

void Anime4KCPP::CPU::ACNet::processGrayscaleW()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        processor->processW(orgImg, dstImg, scaleTimes);

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

        processor->processW(orgImg, dstImg, 1);
    }
}

void Anime4KCPP::CPU::ACNet::processYUVImageF()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        processor->processF(orgY, dstY , scaleTimes);

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

        processor->processF(orgY, dstY, 1);

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
    }
}

void Anime4KCPP::CPU::ACNet::processRGBImageF()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        cv::Mat tmpImg;
        cv::cvtColor(orgImg, tmpImg, cv::COLOR_BGR2YUV);

        processor->processF(tmpImg, dstImg, scaleTimes);

        if (param.isNonIntegerScale())
        {
            cv::resize(dstImg, dstImg, cv::Size(W, H), param.zoomFactor, param.zoomFactor, cv::INTER_AREA);
        }

        cv::resize(tmpImg, tmpImg, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::mixChannels(dstImg, tmpImg, std::vector<int>{0, 0});

        cv::cvtColor(tmpImg, dstImg, cv::COLOR_YUV2BGR);
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);
        
        cv::Mat tmpImg;
        cv::cvtColor(orgImg, tmpImg, cv::COLOR_BGR2YUV);

        processor->processF(tmpImg, dstImg, 1);

        cv::resize(tmpImg, tmpImg, cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
        cv::mixChannels(dstImg, tmpImg, std::vector<int>{0, 0});

        cv::cvtColor(tmpImg, dstImg, cv::COLOR_YUV2BGR);
    }
}

void Anime4KCPP::CPU::ACNet::processGrayscaleF()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        processor->processF(orgImg, dstImg, scaleTimes);

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

        processor->processF(orgImg, dstImg, 1);
    }
}

Anime4KCPP::Processor::Type Anime4KCPP::CPU::ACNet::getProcessorType() noexcept
{
    return Processor::Type::CPU_ACNet;
}

std::string Anime4KCPP::CPU::ACNet::getProcessorInfo()
{
    std::ostringstream oss;
    oss << "Processor type: " << getProcessorType();
    return oss.str();
}
