#ifdef ENABLE_NCNN

#define DLL

#include"ACNetType.hpp"
#include"NCNNACNet.hpp"
#include"NCNNACNetID.hpp"
#include"NCNNACNetModel.hpp"

static bool isInitializedFlag = false;
static ncnn::VulkanDevice* vkdev = nullptr;
static ncnn::Net net[Anime4KCPP::ACNetType::TotalTypeCount];

Anime4KCPP::NCNN::ACNet::ACNet(const Parameters& parameters) :
    AC(parameters)
{
    if (param.HDN)
    {
        switch (param.HDNLevel)
        {
        case 1:
            currACNetypeIndex = HDNL1;
            break;
        case 2:
            currACNetypeIndex = HDNL2;
            break;
        case 3:
            currACNetypeIndex = HDNL3;
            break;
        default:
            currACNetypeIndex = HDNL1;
            break;
        }
    }
    else
    {
        currACNetypeIndex = HDNL0;
    }
}

void Anime4KCPP::NCNN::ACNet::setParameters(const Parameters& parameters)
{
    AC::setParameters(parameters);
    if (param.HDN)
    {
        switch (param.HDNLevel)
        {
        case 1:
            currACNetypeIndex = HDNL1;
            break;
        case 2:
            currACNetypeIndex = HDNL2;
            break;
        case 3:
            currACNetypeIndex = HDNL3;
            break;
        default:
            currACNetypeIndex = HDNL1;
            break;
        }
    }
    else
    {
        currACNetypeIndex = HDNL0;
    }
}

std::string Anime4KCPP::NCNN::ACNet::getInfo()
{
    std::ostringstream oss;
    oss << AC::getInfo()
        << "----------------------------------------------" << std::endl
        << "NCNN device product ID: " << (vkdev ? std::to_string(vkdev->info.device_id()) : "-1") << std::endl
        << "Zoom Factor: " << param.zoomFactor << std::endl
        << "HDN Mode: " << std::boolalpha << param.HDN << std::endl
        << "HDN Level: " << (param.HDN ? param.HDNLevel : 0) << std::endl
        << "----------------------------------------------" << std::endl;
    return oss.str();
}

std::string Anime4KCPP::NCNN::ACNet::getFiltersInfo()
{
    std::ostringstream oss;
    oss << AC::getFiltersInfo()
        << "----------------------------------------------" << std::endl
        << "Filter not supported" << std::endl
        << "----------------------------------------------" << std::endl;
    return oss.str();
}

void Anime4KCPP::NCNN::ACNet::init(
    std::string& modelPath, std::string& paramPath,
    int type, const int deviceID, const int threads)
{
    if (isInitializedFlag)
        return;

    if (vkdev == nullptr)
        vkdev = deviceID < 0 ? nullptr : ncnn::get_gpu_device((deviceID >= ncnn::get_gpu_count()) ? 0 : deviceID);

    if (type >= ACNetType::TotalTypeCount || type < ACNetType::HDNL0)
        type = ACNetType::HDNL0;

    net[type].set_vulkan_device(vkdev);
    net[type].opt.use_vulkan_compute = vkdev ? true : false;

    net[type].opt.use_fp16_arithmetic = false;
    net[type].opt.use_fp16_packed = true;
    net[type].opt.use_fp16_storage = true;

    net[type].opt.use_int8_packed = true;
    net[type].opt.use_int8_storage = true;
    net[type].opt.use_int8_inference = false;

    net[type].opt.num_threads = threads;

    if (net[type].load_param(paramPath.c_str()) || net[type].load_model(modelPath.c_str()))
    {
        release();
        throw ACException<ExceptionType::IO, true>(
            "Failed to load ncnn model or param",
            std::string("model path: ") + modelPath.c_str() + "\nparam path: " + paramPath.c_str(),
            __LINE__);
    }

    isInitializedFlag = true;
}

void Anime4KCPP::NCNN::ACNet::init(int type, const int deviceID, const int threads)
{
    if (isInitializedFlag)
        return;

    if (vkdev == nullptr)
        vkdev = deviceID < 0 ? nullptr : ncnn::get_gpu_device((deviceID >= ncnn::get_gpu_count()) ? 0 : deviceID);

    if (type >= ACNetType::TotalTypeCount || type < ACNetType::HDNL0)
        type = ACNetType::HDNL0;

    net[type].set_vulkan_device(vkdev);
    net[type].opt.use_vulkan_compute = vkdev ? true : false;

    net[type].opt.use_fp16_arithmetic = false;
    net[type].opt.use_fp16_packed = true;
    net[type].opt.use_fp16_storage = true;

    net[type].opt.use_int8_packed = true;
    net[type].opt.use_int8_storage = true;
    net[type].opt.use_int8_inference = false;

    net[type].opt.num_threads = threads;

    if (!net[type].load_param(ACNetParamBin) || !net[type].load_model(ACNetModelBin[type]))
    {
        release();
        throw ACException<ExceptionType::IO, false>("Failed to load ncnn model or param");
    }

    isInitializedFlag = true;
}

void Anime4KCPP::NCNN::ACNet::init(const int deviceID, const int threads)
{
    if (isInitializedFlag)
        return;

    if (vkdev == nullptr)
        vkdev = deviceID < 0 ? nullptr : ncnn::get_gpu_device((deviceID >= ncnn::get_gpu_count()) ? 0 : deviceID);

    for (int type = ACNetType::HDNL0; type < ACNetType::TotalTypeCount; type++)
    {
        net[type].set_vulkan_device(vkdev);
        net[type].opt.use_vulkan_compute = vkdev ? true : false;

        net[type].opt.use_fp16_arithmetic = false;
        net[type].opt.use_fp16_packed = true;
        net[type].opt.use_fp16_storage = true;

        net[type].opt.use_int8_packed = true;
        net[type].opt.use_int8_storage = true;
        net[type].opt.use_int8_inference = false;

        net[type].opt.num_threads = threads;

        if (!net[type].load_param(ACNetParamBin) || !net[type].load_model(ACNetModelBin[type]))
        {
            release();
            throw ACException<ExceptionType::IO, false>("Failed to load ncnn model or param");
        }
    }

    isInitializedFlag = true;
}

void Anime4KCPP::NCNN::ACNet::release()
{
    if (!isInitializedFlag)
        return;

    for (int i = ACNetType::HDNL0; i < ACNetType::TotalTypeCount; i++)
        net[i].clear();

    if (vkdev)
        ncnn::destroy_gpu_instance();

    vkdev = nullptr;

    isInitializedFlag = false;
}

bool Anime4KCPP::NCNN::ACNet::isInitialized()
{
    return isInitializedFlag;
}

void Anime4KCPP::NCNN::ACNet::processCPU(const cv::Mat& orgImg, cv::Mat& dstImg, const int scaleTimes, ncnn::Mat* dataHolder)
{
    if (dataHolder == nullptr)
        dataHolder = &defaultDataHolder;

    ncnn::Mat in;
    if (orgImg.step == orgImg.cols * sizeof(float))
        in = ncnn::Mat{ orgImg.cols, orgImg.rows, 1, orgImg.data };
    else
    {
        in.create(orgImg.cols, orgImg.rows, 1);
        float* dst = reinterpret_cast<float*>(in.data);
        unsigned char* src = orgImg.data;
        for (int i = 0; i < orgImg.rows; i++)
        {
            std::memcpy(dst, src, orgImg.cols * sizeof(float));
            dst += orgImg.cols;
            src += orgImg.step;
        }
    }

    for (int i = 0; i < scaleTimes; i++)
    {
        ncnn::Extractor ex = net[currACNetypeIndex].create_extractor();
        ex.input(ACNetParamID::BLOB_data, in);
        ex.extract(ACNetParamID::BLOB_output, *dataHolder);
        in = *dataHolder;
    }

    dstImg = cv::Mat{ dataHolder->h, dataHolder->w,CV_32FC1, dataHolder->data };
}

void Anime4KCPP::NCNN::ACNet::processVK(const cv::Mat& orgImg, cv::Mat& dstImg, const int scaleTimes, ncnn::Mat* dataHolder)
{
    if (dataHolder == nullptr)
        dataHolder = &defaultDataHolder;

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = net[currACNetypeIndex].opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    ncnn::Mat in;
    if (orgImg.step == orgImg.cols * sizeof(float))
        in = ncnn::Mat{ orgImg.cols, orgImg.rows, 1, orgImg.data };
    else
    {
        in.create(orgImg.cols, orgImg.rows, 1);
        float* dst = reinterpret_cast<float*>(in.data);
        unsigned char* src = orgImg.data;
        for (int i = 0; i < orgImg.rows; i++)
        {
            std::memcpy(dst, src, orgImg.cols * sizeof(float));
            dst += orgImg.cols;
            src += orgImg.step;
        }
    }

    ncnn::VkCompute cmd(vkdev);

    ncnn::VkMat vkIn;
    ncnn::VkMat vkOut;

    cmd.record_upload(in, vkIn, opt);
    for (int i = 0; i < scaleTimes; i++)
    {
        ncnn::Extractor ex = net[currACNetypeIndex].create_extractor();

        ex.set_blob_vkallocator(blob_vkallocator);
        ex.set_workspace_vkallocator(blob_vkallocator);
        ex.set_staging_vkallocator(staging_vkallocator);

        ex.input(ACNetParamID::BLOB_data, vkIn);
        ex.extract(ACNetParamID::BLOB_output, vkOut, cmd);
        vkIn = vkOut;
    }
    cmd.record_download(vkOut, *dataHolder, opt);

    cmd.submit_and_wait();

    dstImg = cv::Mat{ dataHolder->h, dataHolder->w,CV_32FC1, dataHolder->data };

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);
}

void Anime4KCPP::NCNN::ACNet::processYUVImageB()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        orgY.convertTo(orgY, CV_32FC1, 1.0 / 255.0);
        if (vkdev == nullptr)
        {
            processCPU(orgY, dstY, scaleTimes);
        }
        else
        {
            processVK(orgY, dstY, scaleTimes);
        }
        dstY.convertTo(dstY, CV_8UC1, 255.0);

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

        orgY.convertTo(orgY, CV_32FC1, 1.0 / 255.0);
        if (vkdev == nullptr)
        {
            processCPU(orgY, dstY);
        }
        else
        {
            processVK(orgY, dstY);
        }
        dstY.convertTo(dstY, CV_8UC1, 255.0);

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
    }
}

void Anime4KCPP::NCNN::ACNet::processRGBImageB()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        cv::Mat tmpImg = orgImg;
        cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(tmpImg, yuv);
        tmpImg = yuv[Y];

        tmpImg.convertTo(tmpImg, CV_32FC1, 1.0 / 255.0);
        if (vkdev == nullptr)
        {
            processCPU(tmpImg, dstImg, scaleTimes);
        }
        else
        {
            processVK(tmpImg, dstImg, scaleTimes);
        }
        dstImg.convertTo(dstImg, CV_8UC1, 255.0);

        if (param.isNonIntegerScale())
        {
            cv::resize(dstImg, dstImg, cv::Size(W, H), param.zoomFactor, param.zoomFactor, cv::INTER_AREA);
        }

        cv::resize(yuv[U], yuv[U], cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(yuv[V], yuv[V], cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(orgImg, yuv);
        orgImg = yuv[Y];

        orgImg.convertTo(orgImg, CV_32FC1, 1.0 / 255.0);
        if (vkdev == nullptr)
        {
            processCPU(orgImg, dstImg);
        }
        else
        {
            processVK(orgImg, dstImg);
        }
        dstImg.convertTo(dstImg, CV_8UC1, 255.0);

        cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
        cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
}

void Anime4KCPP::NCNN::ACNet::processGrayscaleB()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        orgImg.convertTo(orgImg, CV_32FC1, 1.0 / 255.0);
        if (vkdev == nullptr)
        {
            processCPU(orgImg, dstImg, scaleTimes);
        }
        else
        {
            processVK(orgImg, dstImg, scaleTimes);
        }
        dstImg.convertTo(dstImg, CV_8UC1, 255.0);

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

        orgImg.convertTo(orgImg, CV_32FC1, 1.0 / 255.0);
        if (vkdev == nullptr)
        {
            processCPU(orgImg, dstImg);
        }
        else
        {
            processVK(orgImg, dstImg);
        }
        dstImg.convertTo(dstImg, CV_8UC1, 255.0);
    }
}

void Anime4KCPP::NCNN::ACNet::processYUVImageW()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        orgY.convertTo(orgY, CV_32FC1, 1.0 / 65535.0);
        if (vkdev == nullptr)
        {
            processCPU(orgY, dstY, scaleTimes);
        }
        else
        {
            processVK(orgY, dstY, scaleTimes);
        }
        dstY.convertTo(dstY, CV_16UC1, 65535.0);

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

        orgY.convertTo(orgY, CV_32FC1, 1.0 / 65535.0);
        if (vkdev == nullptr)
        {
            processCPU(orgY, dstY);
        }
        else
        {
            processVK(orgY, dstY);
        }
        dstY.convertTo(dstY, CV_16UC1, 65535.0);

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
    }
}

void Anime4KCPP::NCNN::ACNet::processRGBImageW()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        cv::Mat tmpImg = orgImg;
        cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(tmpImg, yuv);
        tmpImg = yuv[Y];

        tmpImg.convertTo(tmpImg, CV_32FC1, 1.0 / 65535.0);
        if (vkdev == nullptr)
        {
            processCPU(tmpImg, dstImg, scaleTimes);
        }
        else
        {
            processVK(tmpImg, dstImg, scaleTimes);
        }
        dstImg.convertTo(dstImg, CV_16UC1, 65535.0);

        if (param.isNonIntegerScale())
        {
            cv::resize(dstImg, dstImg, cv::Size(W, H), param.zoomFactor, param.zoomFactor, cv::INTER_AREA);
        }

        cv::resize(yuv[U], yuv[U], cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(yuv[V], yuv[V], cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(orgImg, yuv);
        orgImg = yuv[Y];

        orgImg.convertTo(orgImg, CV_32FC1, 1.0 / 65535.0);
        if (vkdev == nullptr)
        {
            processCPU(orgImg, dstImg);
        }
        else
        {
            processVK(orgImg, dstImg);
        }
        dstImg.convertTo(dstImg, CV_16UC1, 65535.0);

        cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
        cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
}

void Anime4KCPP::NCNN::ACNet::processGrayscaleW()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        orgImg.convertTo(orgImg, CV_32FC1, 1.0 / 65535.0);
        if (vkdev == nullptr)
        {
            processCPU(orgImg, dstImg, scaleTimes);
        }
        else
        {
            processVK(orgImg, dstImg, scaleTimes);
        }
        dstImg.convertTo(dstImg, CV_16UC1, 65535.0);

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

        orgImg.convertTo(orgImg, CV_32FC1, 1.0 / 65535.0);
        if (vkdev == nullptr)
        {
            processCPU(orgImg, dstImg);
        }
        else
        {
            processVK(orgImg, dstImg);
        }
        dstImg.convertTo(dstImg, CV_16UC1, 65535.0);
    }
}

void Anime4KCPP::NCNN::ACNet::processYUVImageF()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        if (vkdev == nullptr)
        {
            processCPU(orgY, dstY, scaleTimes);
        }
        else
        {
            processVK(orgY, dstY, scaleTimes);
        }

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

        if (vkdev == nullptr)
        {
            processCPU(orgY, dstY);
        }
        else
        {
            processVK(orgY, dstY);
        }

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
    }
}

void Anime4KCPP::NCNN::ACNet::processRGBImageF()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        cv::Mat tmpImg = orgImg;
        cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(tmpImg, yuv);
        tmpImg = yuv[Y];

        if (vkdev == nullptr)
        {
            processCPU(tmpImg, dstImg, scaleTimes);
        }
        else
        {
            processVK(tmpImg, dstImg, scaleTimes);
        }

        if (param.isNonIntegerScale())
        {
            cv::resize(dstImg, dstImg, cv::Size(W, H), param.zoomFactor, param.zoomFactor, cv::INTER_AREA);
        }

        cv::resize(yuv[U], yuv[U], cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(yuv[V], yuv[V], cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(orgImg, yuv);
        orgImg = yuv[Y];

        if (vkdev == nullptr)
        {
            processCPU(orgImg, dstImg);
        }
        else
        {
            processVK(orgImg, dstImg);
        }

        cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
        cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
}

void Anime4KCPP::NCNN::ACNet::processGrayscaleF()
{
    if (!param.fastMode)
    {
        int scaleTimes = Utils::fastCeilLog2(param.zoomFactor);
        if (!scaleTimes)
            scaleTimes++;

        if (vkdev == nullptr)
        {
            processCPU(orgImg, dstImg, scaleTimes);
        }
        else
        {
            processVK(orgImg, dstImg, scaleTimes);
        }

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

        if (vkdev == nullptr)
        {
            processCPU(orgImg, dstImg);
        }
        else
        {
            processVK(orgImg, dstImg);
        }
    }
}

Anime4KCPP::Processor::Type Anime4KCPP::NCNN::ACNet::getProcessorType() noexcept
{
    return Processor::Type::NCNN_ACNet;
}

std::string Anime4KCPP::NCNN::ACNet::getProcessorInfo()
{
    std::ostringstream oss;
    oss << "Processor type: " << getProcessorType() << std::endl
        << "Current NCNN devices:" << std::endl
        << (vkdev ? (std::string{ " Type: " } + vkdev->info.device_name()) : " Type: CPU");
    return oss.str();
}

#endif
