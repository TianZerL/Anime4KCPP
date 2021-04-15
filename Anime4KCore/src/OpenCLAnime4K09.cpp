#define DLL

#include<fstream>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include<OpenCL/opencl.h>
#else
#include<CL/cl.h>
#endif // SPECIAL OS

#include"FilterProcessor.hpp"
#include "OpenCLAnime4K09.hpp"
#include "OpenCLAnime4K09Kernel.hpp"

#define CLEAN_KERNEL_AND_THROW_ERROR(err, errCode) \
{\
clReleaseMemObject(imageBuffer3); \
clReleaseMemObject(imageBuffer2); \
clReleaseMemObject(imageBuffer1); \
clReleaseMemObject(imageBuffer0); \
clReleaseKernel(kernelGetGray); \
clReleaseKernel(kernelPushColor); \
clReleaseKernel(kernelGetGradient); \
clReleaseKernel(kernelPushGradient); \
throw ACException<ExceptionType::GPU, true>(err, errCode); \
}

//init OpenCL arguments
static bool isInitialized = false;
static cl_context context = nullptr;
static int commandQueueNum = 4;
static int commandQueueCount = 0;
static std::vector<cl_command_queue> commandQueueList(commandQueueNum, nullptr);
static bool parallelIO = false;
static cl_command_queue commandQueueIO = nullptr;
static cl_program program = nullptr;
static cl_device_id device = nullptr;
static int pID = 0;
static int dID = 0;
static size_t workGroupSizeLog = 5;

Anime4KCPP::OpenCL::Anime4K09::Anime4K09(const Parameters& parameters) :
    AC(parameters), nWidth(0.0), nHeight(0.0) {};

void Anime4KCPP::OpenCL::Anime4K09::initGPU(const int platformID, const int deviceID, const int OpenCLQueueNum, const bool OpenCLParallelIO)
{
    if (!isInitialized)
    {
        pID = platformID;
        dID = deviceID;
        commandQueueNum = OpenCLQueueNum >= 1 ? OpenCLQueueNum : 1;
        parallelIO = OpenCLParallelIO;
        initOpenCL();
        isInitialized = true;
    }
}

void Anime4KCPP::OpenCL::Anime4K09::releaseGPU() noexcept
{
    if (isInitialized)
    {
        releaseOpenCL();
        context = nullptr;
        std::fill(commandQueueList.begin(), commandQueueList.end(), nullptr);
        commandQueueIO = nullptr;
        program = nullptr;
        device = nullptr;
        isInitialized = false;
    }
}

bool Anime4KCPP::OpenCL::Anime4K09::isInitializedGPU() noexcept
{
    return isInitialized;
}

std::string Anime4KCPP::OpenCL::Anime4K09::getInfo()
{
    std::ostringstream oss;
    oss << AC::getInfo()
        << "----------------------------------------------" << std::endl
        << "OpenCL Platform ID:" << pID << std::endl
        << "OpenCL Device ID:" << dID << std::endl
        << "Passes: " << param.passes << std::endl
        << "pushColorCount: " << param.pushColorCount << std::endl
        << "Zoom Factor: " << param.zoomFactor << std::endl
        << "Video Mode: " << std::boolalpha << param.videoMode << std::endl
        << "Fast Mode: " << std::boolalpha << param.fastMode << std::endl
        << "Strength Color: " << param.strengthColor << std::endl
        << "Strength Gradient: " << param.strengthGradient << std::endl
        << "Number of OpenCL Command Queues:" << commandQueueNum << std::endl
        << "OpenCL Parallel IO Command Queues:" << std::boolalpha << parallelIO << std::endl
        << "----------------------------------------------" << std::endl;
    return oss.str();
}

std::string Anime4KCPP::OpenCL::Anime4K09::getFiltersInfo()
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

void Anime4KCPP::OpenCL::Anime4K09::processYUVImageB()
{
    if (param.zoomFactor == 2.0)
    {
        nWidth = 1.0 / static_cast<double>(W);
        nHeight = 1.0 / static_cast<double>(H);
    }
    else
    {
        nWidth = static_cast<double>(orgW) / static_cast<double>(W);
        nHeight = static_cast<double>(orgH) / static_cast<double>(H);
    }

    cv::merge(std::vector<cv::Mat>{ orgY, orgU, orgV }, orgImg);
    cv::cvtColor(orgImg, orgImg, cv::COLOR_YUV2BGR);

    dstImg.create(H, W, CV_8UC4);
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(orgImg, param.preFilters).process();
    cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2BGRA);
    if (parallelIO)
        runKernelPB(orgImg, dstImg);
    else
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

void Anime4KCPP::OpenCL::Anime4K09::processRGBImageB()
{
    if (param.zoomFactor == 2.0)
    {
        nWidth = 1.0 / static_cast<double>(W);
        nHeight = 1.0 / static_cast<double>(H);
    }
    else
    {
        nWidth = static_cast<double>(orgW) / static_cast<double>(W);
        nHeight = static_cast<double>(orgH) / static_cast<double>(H);
    }

    dstImg.create(H, W, CV_8UC4);
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(orgImg, param.preFilters).process();
    cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2BGRA);
    if (parallelIO)
        runKernelPB(orgImg, dstImg);
    else
        runKernelB(orgImg, dstImg);
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
    if (param.postprocessing)//Postprocessing(CPU)
        FilterProcessor(dstImg, param.postFilters).process();
}

void Anime4KCPP::OpenCL::Anime4K09::processGrayscaleB()
{
    if (param.zoomFactor == 2.0)
    {
        nWidth = 1.0 / static_cast<double>(W);
        nHeight = 1.0 / static_cast<double>(H);
    }
    else
    {
        nWidth = static_cast<double>(orgW) / static_cast<double>(W);
        nHeight = static_cast<double>(orgH) / static_cast<double>(H);
    }

    cv::cvtColor(orgImg, orgImg, cv::COLOR_GRAY2BGR);

    dstImg.create(H, W, CV_8UC4);
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(orgImg, param.preFilters).process();
    cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2BGRA);
    if (parallelIO)
        runKernelPB(orgImg, dstImg);
    else
        runKernelB(orgImg, dstImg);
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
    if (param.postprocessing)//Postprocessing(CPU)
        FilterProcessor(dstImg, param.postFilters).process();

    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGR2GRAY);
}

void Anime4KCPP::OpenCL::Anime4K09::processRGBVideoB()
{
    if (param.zoomFactor == 2.0)
    {
        nWidth = 1.0 / static_cast<double>(W);
        nHeight = 1.0 / static_cast<double>(H);
    }
    else
    {
        nWidth = static_cast<double>(orgW) / static_cast<double>(W);
        nHeight = static_cast<double>(orgH) / static_cast<double>(H);
    }

    videoIO->init(
        [this]()
        {
            Utils::Frame frame = videoIO->read();
            cv::Mat orgFrame = frame.first;
            cv::Mat dstFrame(H, W, CV_8UC4);
            if (param.preprocessing)
                FilterProcessor(orgFrame, param.preFilters).process();
            cv::cvtColor(orgFrame, orgFrame, cv::COLOR_BGR2BGRA);
            if (parallelIO)
                runKernelPB(orgFrame, dstFrame);
            else
                runKernelB(orgFrame, dstFrame);
            cv::cvtColor(dstFrame, dstFrame, cv::COLOR_BGRA2BGR);
            if (param.postprocessing)//PostProcessing
                FilterProcessor(dstFrame, param.postFilters).process();
            frame.first = dstFrame;
            videoIO->write(frame);
        }
        , param.maxThreads
            ).process();
}

void Anime4KCPP::OpenCL::Anime4K09::processYUVImageW()
{
    if (param.zoomFactor == 2.0)
    {
        nWidth = 1.0 / static_cast<double>(W);
        nHeight = 1.0 / static_cast<double>(H);
    }
    else
    {
        nWidth = static_cast<double>(orgW) / static_cast<double>(W);
        nHeight = static_cast<double>(orgH) / static_cast<double>(H);
    }

    cv::merge(std::vector<cv::Mat>{ orgY, orgU, orgV }, orgImg);
    cv::cvtColor(orgImg, orgImg, cv::COLOR_YUV2BGR);

    dstImg.create(H, W, CV_16UC4);
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(orgImg, param.preFilters).process();
    cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2BGRA);
    if (parallelIO)
        runKernelPW(orgImg, dstImg);
    else
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

void Anime4KCPP::OpenCL::Anime4K09::processRGBImageW()
{
    if (param.zoomFactor == 2.0)
    {
        nWidth = 1.0 / static_cast<double>(W);
        nHeight = 1.0 / static_cast<double>(H);
    }
    else
    {
        nWidth = static_cast<double>(orgW) / static_cast<double>(W);
        nHeight = static_cast<double>(orgH) / static_cast<double>(H);
    }

    dstImg.create(H, W, CV_16UC4);
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(orgImg, param.preFilters).process();
    cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2BGRA);
    if (parallelIO)
        runKernelPW(orgImg, dstImg);
    else
        runKernelW(orgImg, dstImg);
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
    if (param.postprocessing)//Postprocessing(CPU)
        FilterProcessor(dstImg, param.postFilters).process();
}

void Anime4KCPP::OpenCL::Anime4K09::processGrayscaleW()
{
    if (param.zoomFactor == 2.0)
    {
        nWidth = 1.0 / static_cast<double>(W);
        nHeight = 1.0 / static_cast<double>(H);
    }
    else
    {
        nWidth = static_cast<double>(orgW) / static_cast<double>(W);
        nHeight = static_cast<double>(orgH) / static_cast<double>(H);
    }

    cv::cvtColor(orgImg, orgImg, cv::COLOR_GRAY2BGR);

    dstImg.create(H, W, CV_16UC4);
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(orgImg, param.preFilters).process();
    cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2BGRA);
    if (parallelIO)
        runKernelPW(orgImg, dstImg);
    else
        runKernelW(orgImg, dstImg);
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
    if (param.postprocessing)//Postprocessing(CPU)
        FilterProcessor(dstImg, param.postFilters).process();

    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGR2GRAY);
}

void Anime4KCPP::OpenCL::Anime4K09::processYUVImageF()
{
    if (param.zoomFactor == 2.0)
    {
        nWidth = 1.0 / static_cast<double>(W);
        nHeight = 1.0 / static_cast<double>(H);
    }
    else
    {
        nWidth = static_cast<double>(orgW) / static_cast<double>(W);
        nHeight = static_cast<double>(orgH) / static_cast<double>(H);
    }

    cv::merge(std::vector<cv::Mat>{ orgY, orgU, orgV }, orgImg);
    cv::cvtColor(orgImg, orgImg, cv::COLOR_YUV2BGR);

    dstImg.create(H, W, CV_32FC4);
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(orgImg, param.preFilters).process();
    cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2BGRA);
    if (parallelIO)
        runKernelPF(orgImg, dstImg);
    else
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

void Anime4KCPP::OpenCL::Anime4K09::processRGBImageF()
{
    if (param.zoomFactor == 2.0)
    {
        nWidth = 1.0 / static_cast<double>(W);
        nHeight = 1.0 / static_cast<double>(H);
    }
    else
    {
        nWidth = static_cast<double>(orgW) / static_cast<double>(W);
        nHeight = static_cast<double>(orgH) / static_cast<double>(H);
    }

    dstImg.create(H, W, CV_32FC4);
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(orgImg, param.preFilters).process();
    cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2BGRA);
    if (parallelIO)
        runKernelPF(orgImg, dstImg);
    else
        runKernelF(orgImg, dstImg);
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
    if (param.postprocessing)//Postprocessing(CPU)
        FilterProcessor(dstImg, param.postFilters).process();
}

void Anime4KCPP::OpenCL::Anime4K09::processGrayscaleF()
{
    if (param.zoomFactor == 2.0)
    {
        nWidth = 1.0 / static_cast<double>(W);
        nHeight = 1.0 / static_cast<double>(H);
    }
    else
    {
        nWidth = static_cast<double>(orgW) / static_cast<double>(W);
        nHeight = static_cast<double>(orgH) / static_cast<double>(H);
    }

    cv::cvtColor(orgImg, orgImg, cv::COLOR_GRAY2BGR);

    dstImg.create(H, W, CV_32FC4);
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(orgImg, param.preFilters).process();
    cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2BGRA);
    if (parallelIO)
        runKernelPF(orgImg, dstImg);
    else
        runKernelF(orgImg, dstImg);
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
    if (param.postprocessing)//Postprocessing(CPU)
        FilterProcessor(dstImg, param.postFilters).process();

    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGR2GRAY);
}

void Anime4KCPP::OpenCL::Anime4K09::runKernelB(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    cl_int err = CL_SUCCESS;

    int i;

    cl_image_format format{};

    cl_image_desc dstDesc{};
    cl_image_desc orgDesc{};

    constexpr size_t orgin[3] = { 0,0,0 };
    const size_t orgRegion[3] = { static_cast<const size_t>(orgImg.cols),static_cast<const size_t>(orgImg.rows),1 };
    const size_t dstRegion[3] = { static_cast<const size_t>(dstImg.cols),static_cast<const size_t>(dstImg.rows),1 };
    const size_t size[2] =
    { 
        (((static_cast<const size_t>(dstImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(dstImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };

    const cl_float pushColorStrength = static_cast<const cl_float>(param.strengthColor);
    const cl_float pushGradientStrength = static_cast<const cl_float>(param.strengthGradient);
    const cl_float normalizedWidth = static_cast<const cl_float>(nWidth);
    const cl_float normalizedHeight = static_cast<const cl_float>(nHeight);

    //init frame
    format.image_channel_data_type = CL_UNORM_INT8;
    format.image_channel_order = CL_RGBA;

    orgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    orgDesc.image_height = orgImg.rows;
    orgDesc.image_width = orgImg.cols;
    orgDesc.buffer = nullptr;

    dstDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    dstDesc.image_height = dstImg.rows;
    dstDesc.image_width = dstImg.cols;
    dstDesc.buffer = nullptr;

    cl_command_queue commandQueue = commandQueueList[commandQueueCount++];
    if (commandQueueCount >= commandQueueNum)
        commandQueueCount = 0;

    //kernel for each thread
    cl_kernel kernelGetGray = nullptr;
    if (param.zoomFactor == 2.0)
        kernelGetGray = clCreateKernel(program, "getGray", &err);
    else
        kernelGetGray = clCreateKernel(program, "getGrayLanczos4", &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel getGray", err);
    }
    cl_kernel kernelPushColor = clCreateKernel(program, "pushColor", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelGetGray);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel pushColor", err);
    }
    cl_kernel kernelGetGradient = clCreateKernel(program, "getGradient", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelGetGray);
        clReleaseKernel(kernelPushColor);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel getGradient", err);
    }
    cl_kernel kernelPushGradient = clCreateKernel(program, "pushGradient", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelGetGray);
        clReleaseKernel(kernelPushColor);
        clReleaseKernel(kernelGetGradient);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel pushGradient", err);
    }

    //imageBuffer
    //for getGray
    cl_mem imageBuffer0 = clCreateImage(context, CL_MEM_READ_ONLY, &format, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer0 error, video memory may be insufficient.", err);
    }
    //tmp buffer 1
    cl_mem imageBuffer1 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBuffer0);
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer1 error, video memory may be insufficient.", err);
    }
    //tmp buffer 2
    cl_mem imageBuffer2 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBuffer0);
        clReleaseMemObject(imageBuffer1);
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer2 error, video memory may be insufficient.", err);
    }
    //tmp buffer 3
    cl_mem imageBuffer3 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBuffer0);
        clReleaseMemObject(imageBuffer1);
        clReleaseMemObject(imageBuffer2);
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer3 error, video memory may be insufficient.", err);
    }

    //set arguments
    //getGray
    err = clSetKernelArg(kernelGetGray, 0, sizeof(cl_mem), &imageBuffer0);
    err |= clSetKernelArg(kernelGetGray, 1, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(kernelGetGray, 2, sizeof(cl_float), &normalizedWidth);
    err |= clSetKernelArg(kernelGetGray, 3, sizeof(cl_float), &normalizedHeight);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: getGray error", err)
    //pushColor
    err = clSetKernelArg(kernelPushColor, 0, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(kernelPushColor, 1, sizeof(cl_mem), &imageBuffer2);
    err |= clSetKernelArg(kernelPushColor, 2, sizeof(cl_float), &pushColorStrength);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: pushColor error", err)
    //getGradient
    err = clSetKernelArg(kernelGetGradient, 0, sizeof(cl_mem), &imageBuffer2);
    err |= clSetKernelArg(kernelGetGradient, 1, sizeof(cl_mem), &imageBuffer3);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: getGradient error", err)
    //pushGradient
    err = clSetKernelArg(kernelPushGradient, 0, sizeof(cl_mem), &imageBuffer3);
    err |= clSetKernelArg(kernelPushGradient, 1, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(kernelPushGradient, 2, sizeof(cl_float), &pushGradientStrength);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: pushGradient error", err)

    //enqueue
    clEnqueueWriteImage(commandQueue, imageBuffer0, CL_FALSE, orgin, orgRegion, orgImg.step, 0, orgImg.data, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelGetGray, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
    for (i = 0; i < param.passes && i < param.pushColorCount; i++)//pcc for push color count
    {
        clEnqueueNDRangeKernel(commandQueue, kernelPushColor, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        clEnqueueNDRangeKernel(commandQueue, kernelGetGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        clEnqueueNDRangeKernel(commandQueue, kernelPushGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
    }
    if (i < param.passes)
    {
        //reset getGradient
        err = clSetKernelArg(kernelGetGradient, 0, sizeof(cl_mem), &imageBuffer1);
        err |= clSetKernelArg(kernelGetGradient, 1, sizeof(cl_mem), &imageBuffer2);
        if (err != CL_SUCCESS)
            CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: reset getGradient error", err)
        //reset pushGradient
        err = clSetKernelArg(kernelPushGradient, 0, sizeof(cl_mem), &imageBuffer2);
        err |= clSetKernelArg(kernelPushGradient, 1, sizeof(cl_mem), &imageBuffer1);
        err |= clSetKernelArg(kernelPushGradient, 2, sizeof(cl_float), &pushGradientStrength);
        if (err != CL_SUCCESS)
            CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: reset pushGradient error", err)

        while (i++ < param.passes)
        {
            clEnqueueNDRangeKernel(commandQueue, kernelGetGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
            clEnqueueNDRangeKernel(commandQueue, kernelPushGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        }
    }
    //blocking read
    clEnqueueReadImage(commandQueue, imageBuffer1, CL_TRUE, orgin, dstRegion, dstImg.step, 0, dstImg.data, 0, nullptr, nullptr);

    //clean
    clReleaseMemObject(imageBuffer3);
    clReleaseMemObject(imageBuffer2);
    clReleaseMemObject(imageBuffer1);
    clReleaseMemObject(imageBuffer0);

    clReleaseKernel(kernelGetGray);
    clReleaseKernel(kernelPushColor);
    clReleaseKernel(kernelGetGradient);
    clReleaseKernel(kernelPushGradient);
}

void Anime4KCPP::OpenCL::Anime4K09::runKernelW(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    cl_int err = CL_SUCCESS;

    int i;

    cl_image_format format{};

    cl_image_desc dstDesc{};
    cl_image_desc orgDesc{};

    constexpr size_t orgin[3] = { 0,0,0 };
    const size_t orgRegion[3] = { static_cast<const size_t>(orgImg.cols),static_cast<const size_t>(orgImg.rows),1 };
    const size_t dstRegion[3] = { static_cast<const size_t>(dstImg.cols),static_cast<const size_t>(dstImg.rows),1 };
    const size_t size[2] =
    {
        (((static_cast<const size_t>(dstImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(dstImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };

    const cl_float pushColorStrength = static_cast<const cl_float>(param.strengthColor);
    const cl_float pushGradientStrength = static_cast<const cl_float>(param.strengthGradient);
    const cl_float normalizedWidth = static_cast<const cl_float>(nWidth);
    const cl_float normalizedHeight = static_cast<const cl_float>(nHeight);

    //init frame
    format.image_channel_data_type = CL_UNORM_INT16;
    format.image_channel_order = CL_RGBA;

    orgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    orgDesc.image_height = orgImg.rows;
    orgDesc.image_width = orgImg.cols;
    orgDesc.buffer = nullptr;

    dstDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    dstDesc.image_height = dstImg.rows;
    dstDesc.image_width = dstImg.cols;
    dstDesc.buffer = nullptr;

    cl_command_queue commandQueue = commandQueueList[commandQueueCount++];
    if (commandQueueCount >= commandQueueNum)
        commandQueueCount = 0;

    //kernel for each thread
    cl_kernel kernelGetGray = nullptr;
    if (param.zoomFactor == 2.0)
        kernelGetGray = clCreateKernel(program, "getGray", &err);
    else
        kernelGetGray = clCreateKernel(program, "getGrayLanczos4", &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel getGray", err);
    }
    cl_kernel kernelPushColor = clCreateKernel(program, "pushColor", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelGetGray);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel pushColor", err);
    }
    cl_kernel kernelGetGradient = clCreateKernel(program, "getGradient", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelGetGray);
        clReleaseKernel(kernelPushColor);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel getGradient", err);
    }
    cl_kernel kernelPushGradient = clCreateKernel(program, "pushGradient", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelGetGray);
        clReleaseKernel(kernelPushColor);
        clReleaseKernel(kernelGetGradient);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel pushGradient", err);
    }

    //imageBuffer
    //for getGray
    cl_mem imageBuffer0 = clCreateImage(context, CL_MEM_READ_ONLY, &format, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer0 error, video memory may be insufficient.", err);
    }
    //tmp buffer 1
    cl_mem imageBuffer1 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBuffer0);
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer1 error, video memory may be insufficient.", err);
    }
    //tmp buffer 2
    cl_mem imageBuffer2 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBuffer0);
        clReleaseMemObject(imageBuffer1);
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer2 error, video memory may be insufficient.", err);
    }
    //tmp buffer 3
    cl_mem imageBuffer3 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBuffer0);
        clReleaseMemObject(imageBuffer1);
        clReleaseMemObject(imageBuffer2);
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer3 error, video memory may be insufficient.", err);
    }

    //set arguments
    //getGray
    err = clSetKernelArg(kernelGetGray, 0, sizeof(cl_mem), &imageBuffer0);
    err |= clSetKernelArg(kernelGetGray, 1, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(kernelGetGray, 2, sizeof(cl_float), &normalizedWidth);
    err |= clSetKernelArg(kernelGetGray, 3, sizeof(cl_float), &normalizedHeight);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: getGray error", err)
    //pushColor
    err = clSetKernelArg(kernelPushColor, 0, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(kernelPushColor, 1, sizeof(cl_mem), &imageBuffer2);
    err |= clSetKernelArg(kernelPushColor, 2, sizeof(cl_float), &pushColorStrength);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: pushColor error", err)
    //getGradient
    err = clSetKernelArg(kernelGetGradient, 0, sizeof(cl_mem), &imageBuffer2);
    err |= clSetKernelArg(kernelGetGradient, 1, sizeof(cl_mem), &imageBuffer3);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: getGradient error", err)
    //pushGradient
    err = clSetKernelArg(kernelPushGradient, 0, sizeof(cl_mem), &imageBuffer3);
    err |= clSetKernelArg(kernelPushGradient, 1, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(kernelPushGradient, 2, sizeof(cl_float), &pushGradientStrength);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: pushGradient error", err)

    //enqueue
    clEnqueueWriteImage(commandQueue, imageBuffer0, CL_FALSE, orgin, orgRegion, orgImg.step, 0, orgImg.data, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelGetGray, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
    for (i = 0; i < param.passes && i < param.pushColorCount; i++)//pcc for push color count
    {
        clEnqueueNDRangeKernel(commandQueue, kernelPushColor, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        clEnqueueNDRangeKernel(commandQueue, kernelGetGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        clEnqueueNDRangeKernel(commandQueue, kernelPushGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
    }
    if (i < param.passes)
    {
        //reset getGradient
        err = clSetKernelArg(kernelGetGradient, 0, sizeof(cl_mem), &imageBuffer1);
        err |= clSetKernelArg(kernelGetGradient, 1, sizeof(cl_mem), &imageBuffer2);
        if (err != CL_SUCCESS)
            CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: reset getGradient error", err)
        //reset pushGradient
        err = clSetKernelArg(kernelPushGradient, 0, sizeof(cl_mem), &imageBuffer2);
        err |= clSetKernelArg(kernelPushGradient, 1, sizeof(cl_mem), &imageBuffer1);
        err |= clSetKernelArg(kernelPushGradient, 2, sizeof(cl_float), &pushGradientStrength);
        if (err != CL_SUCCESS)
            CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: reset pushGradient error", err)

        while (i++ < param.passes)
        {
            clEnqueueNDRangeKernel(commandQueue, kernelGetGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
            clEnqueueNDRangeKernel(commandQueue, kernelPushGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        }
    }
    //blocking read
    clEnqueueReadImage(commandQueue, imageBuffer1, CL_TRUE, orgin, dstRegion, dstImg.step, 0, dstImg.data, 0, nullptr, nullptr);

    //clean
    clReleaseMemObject(imageBuffer3);
    clReleaseMemObject(imageBuffer2);
    clReleaseMemObject(imageBuffer1);
    clReleaseMemObject(imageBuffer0);

    clReleaseKernel(kernelGetGray);
    clReleaseKernel(kernelPushColor);
    clReleaseKernel(kernelGetGradient);
    clReleaseKernel(kernelPushGradient);
}

void Anime4KCPP::OpenCL::Anime4K09::runKernelF(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    cl_int err = CL_SUCCESS;
    int i;

    cl_image_format format{};

    cl_image_desc dstDesc{};
    cl_image_desc orgDesc{};

    constexpr size_t orgin[3] = { 0,0,0 };
    const size_t orgRegion[3] = { static_cast<const size_t>(orgImg.cols),static_cast<const size_t>(orgImg.rows),1 };
    const size_t dstRegion[3] = { static_cast<const size_t>(dstImg.cols),static_cast<const size_t>(dstImg.rows),1 };
    const size_t size[2] =
    {
        (((static_cast<const size_t>(dstImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(dstImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };

    const cl_float pushColorStrength = static_cast<const cl_float>(param.strengthColor);
    const cl_float pushGradientStrength = static_cast<const cl_float>(param.strengthGradient);
    const cl_float normalizedWidth = static_cast<const cl_float>(nWidth);
    const cl_float normalizedHeight = static_cast<const cl_float>(nHeight);

    //init frame
    format.image_channel_data_type = CL_FLOAT;
    format.image_channel_order = CL_RGBA;

    orgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    orgDesc.image_height = orgImg.rows;
    orgDesc.image_width = orgImg.cols;
    orgDesc.buffer = nullptr;

    dstDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    dstDesc.image_height = dstImg.rows;
    dstDesc.image_width = dstImg.cols;
    dstDesc.buffer = nullptr;

    cl_command_queue commandQueue = commandQueueList[commandQueueCount++];
    if (commandQueueCount >= commandQueueNum)
        commandQueueCount = 0;

    //kernel for each thread
    cl_kernel kernelGetGray = nullptr;
    if (param.zoomFactor == 2.0)
        kernelGetGray = clCreateKernel(program, "getGray", &err);
    else
        kernelGetGray = clCreateKernel(program, "getGrayLanczos4", &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel getGray", err);
    }
    cl_kernel kernelPushColor = clCreateKernel(program, "pushColor", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelGetGray);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel pushColor", err);
    }
    cl_kernel kernelGetGradient = clCreateKernel(program, "getGradient", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelGetGray);
        clReleaseKernel(kernelPushColor);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel getGradient", err);
    }
    cl_kernel kernelPushGradient = clCreateKernel(program, "pushGradient", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelGetGray);
        clReleaseKernel(kernelPushColor);
        clReleaseKernel(kernelGetGradient);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel pushGradient", err);
    }

    //imageBuffer
    //for getGray
    cl_mem imageBuffer0 = clCreateImage(context, CL_MEM_READ_ONLY, &format, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer0 error, video memory may be insufficient.", err);
    }
    //tmp buffer 1
    cl_mem imageBuffer1 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBuffer0);
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer1 error, video memory may be insufficient.", err);
    }
    //tmp buffer 2
    cl_mem imageBuffer2 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBuffer0);
        clReleaseMemObject(imageBuffer1);
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer2 error, video memory may be insufficient.", err);
    }
    //tmp buffer 3
    cl_mem imageBuffer3 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBuffer0);
        clReleaseMemObject(imageBuffer1);
        clReleaseMemObject(imageBuffer2);
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer3 error, video memory may be insufficient.", err);
    }

    //set arguments
    //getGray
    err = clSetKernelArg(kernelGetGray, 0, sizeof(cl_mem), &imageBuffer0);
    err |= clSetKernelArg(kernelGetGray, 1, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(kernelGetGray, 2, sizeof(cl_float), &normalizedWidth);
    err |= clSetKernelArg(kernelGetGray, 3, sizeof(cl_float), &normalizedHeight);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: getGray error", err)
    //pushColor
    err = clSetKernelArg(kernelPushColor, 0, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(kernelPushColor, 1, sizeof(cl_mem), &imageBuffer2);
    err |= clSetKernelArg(kernelPushColor, 2, sizeof(cl_float), &pushColorStrength);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: pushColor error", err)
    //getGradient
    err = clSetKernelArg(kernelGetGradient, 0, sizeof(cl_mem), &imageBuffer2);
    err |= clSetKernelArg(kernelGetGradient, 1, sizeof(cl_mem), &imageBuffer3);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: getGradient error", err)
    //pushGradient
    err = clSetKernelArg(kernelPushGradient, 0, sizeof(cl_mem), &imageBuffer3);
    err |= clSetKernelArg(kernelPushGradient, 1, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(kernelPushGradient, 2, sizeof(cl_float), &pushGradientStrength);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: pushGradient error", err)

    //enqueue
    clEnqueueWriteImage(commandQueue, imageBuffer0, CL_FALSE, orgin, orgRegion, orgImg.step, 0, orgImg.data, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelGetGray, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
    for (i = 0; i < param.passes && i < param.pushColorCount; i++)//pcc for push color count
    {
        clEnqueueNDRangeKernel(commandQueue, kernelPushColor, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        clEnqueueNDRangeKernel(commandQueue, kernelGetGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        clEnqueueNDRangeKernel(commandQueue, kernelPushGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
    }
    if (i < param.passes)
    {
        //reset getGradient
        err = clSetKernelArg(kernelGetGradient, 0, sizeof(cl_mem), &imageBuffer1);
        err |= clSetKernelArg(kernelGetGradient, 1, sizeof(cl_mem), &imageBuffer2);
        if (err != CL_SUCCESS)
            CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: reset getGradient error", err)
        //reset pushGradient
        err = clSetKernelArg(kernelPushGradient, 0, sizeof(cl_mem), &imageBuffer2);
        err |= clSetKernelArg(kernelPushGradient, 1, sizeof(cl_mem), &imageBuffer1);
        err |= clSetKernelArg(kernelPushGradient, 2, sizeof(cl_float), &pushGradientStrength);
        if (err != CL_SUCCESS)
            CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: reset pushGradient error", err)

        while (i++ < param.passes)
        {
            clEnqueueNDRangeKernel(commandQueue, kernelGetGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
            clEnqueueNDRangeKernel(commandQueue, kernelPushGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        }
    }
    //blocking read
    clEnqueueReadImage(commandQueue, imageBuffer1, CL_TRUE, orgin, dstRegion, dstImg.step, 0, dstImg.data, 0, nullptr, nullptr);

    //clean
    clReleaseMemObject(imageBuffer3);
    clReleaseMemObject(imageBuffer2);
    clReleaseMemObject(imageBuffer1);
    clReleaseMemObject(imageBuffer0);

    clReleaseKernel(kernelGetGray);
    clReleaseKernel(kernelPushColor);
    clReleaseKernel(kernelGetGradient);
    clReleaseKernel(kernelPushGradient);
}

void Anime4KCPP::OpenCL::Anime4K09::runKernelPB(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    cl_int err = CL_SUCCESS;
    cl_event writeFinishedEvent = nullptr;
    cl_event readReadyEvent = nullptr;
    cl_event readFinishedEvent = nullptr;
    int i;

    cl_image_format format{};

    cl_image_desc dstDesc{};
    cl_image_desc orgDesc{};

    constexpr size_t orgin[3] = { 0,0,0 };
    const size_t orgRegion[3] = { static_cast<const size_t>(orgImg.cols),static_cast<const size_t>(orgImg.rows),1 };
    const size_t dstRegion[3] = { static_cast<const size_t>(dstImg.cols),static_cast<const size_t>(dstImg.rows),1 };
    const size_t size[2] =
    {
        (((static_cast<const size_t>(dstImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(dstImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };

    const cl_float pushColorStrength = static_cast<const cl_float>(param.strengthColor);
    const cl_float pushGradientStrength = static_cast<const cl_float>(param.strengthGradient);
    const cl_float normalizedWidth = static_cast<const cl_float>(nWidth);
    const cl_float normalizedHeight = static_cast<const cl_float>(nHeight);

    //init frame
    format.image_channel_data_type = CL_UNORM_INT8;
    format.image_channel_order = CL_RGBA;

    orgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    orgDesc.image_height = orgImg.rows;
    orgDesc.image_width = orgImg.cols;
    orgDesc.buffer = nullptr;

    dstDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    dstDesc.image_height = dstImg.rows;
    dstDesc.image_width = dstImg.cols;
    dstDesc.buffer = nullptr;

    cl_command_queue commandQueue = commandQueueList[commandQueueCount++];
    if (commandQueueCount >= commandQueueNum)
        commandQueueCount = 0;

    //kernel for each thread
    cl_kernel kernelGetGray = nullptr;
    if (param.zoomFactor == 2.0)
        kernelGetGray = clCreateKernel(program, "getGray", &err);
    else
        kernelGetGray = clCreateKernel(program, "getGrayLanczos4", &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel getGray", err);
    }
    cl_kernel kernelPushColor = clCreateKernel(program, "pushColor", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelGetGray);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel pushColor", err);
    }
    cl_kernel kernelGetGradient = clCreateKernel(program, "getGradient", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelGetGray);
        clReleaseKernel(kernelPushColor);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel getGradient", err);
    }
    cl_kernel kernelPushGradient = clCreateKernel(program, "pushGradient", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelGetGray);
        clReleaseKernel(kernelPushColor);
        clReleaseKernel(kernelGetGradient);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel pushGradient", err);
    }

    //imageBuffer
    //for getGray
    cl_mem imageBuffer0 = clCreateImage(context, CL_MEM_READ_ONLY, &format, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer0 error, video memory may be insufficient.", err);
    }
    //tmp buffer 1
    cl_mem imageBuffer1 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBuffer0);
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer1 error, video memory may be insufficient.", err);
    }
    //tmp buffer 2
    cl_mem imageBuffer2 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBuffer0);
        clReleaseMemObject(imageBuffer1);
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer2 error, video memory may be insufficient.", err);
    }
    //tmp buffer 3
    cl_mem imageBuffer3 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBuffer0);
        clReleaseMemObject(imageBuffer1);
        clReleaseMemObject(imageBuffer2);
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer3 error, video memory may be insufficient.", err);
    }

    //set arguments
    //getGray
    err = clSetKernelArg(kernelGetGray, 0, sizeof(cl_mem), &imageBuffer0);
    err |= clSetKernelArg(kernelGetGray, 1, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(kernelGetGray, 2, sizeof(cl_float), &normalizedWidth);
    err |= clSetKernelArg(kernelGetGray, 3, sizeof(cl_float), &normalizedHeight);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: getGray error", err)
        //pushColor
    err = clSetKernelArg(kernelPushColor, 0, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(kernelPushColor, 1, sizeof(cl_mem), &imageBuffer2);
    err |= clSetKernelArg(kernelPushColor, 2, sizeof(cl_float), &pushColorStrength);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: pushColor error", err)
        //getGradient
    err = clSetKernelArg(kernelGetGradient, 0, sizeof(cl_mem), &imageBuffer2);
    err |= clSetKernelArg(kernelGetGradient, 1, sizeof(cl_mem), &imageBuffer3);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: getGradient error", err)
        //pushGradient
    err = clSetKernelArg(kernelPushGradient, 0, sizeof(cl_mem), &imageBuffer3);
    err |= clSetKernelArg(kernelPushGradient, 1, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(kernelPushGradient, 2, sizeof(cl_float), &pushGradientStrength);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: pushGradient error", err)

    //enqueue
    clEnqueueWriteImage(commandQueueIO, imageBuffer0, CL_FALSE, orgin, orgRegion, orgImg.step, 0, orgImg.data, 0, nullptr, &writeFinishedEvent);
    clEnqueueNDRangeKernel(commandQueue, kernelGetGray, 2, nullptr, size, nullptr, 1, &writeFinishedEvent, nullptr);
    for (i = 0; i < param.passes && i < param.pushColorCount; i++)//pcc for push color count
    {
        clEnqueueNDRangeKernel(commandQueue, kernelPushColor, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        clEnqueueNDRangeKernel(commandQueue, kernelGetGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        clEnqueueNDRangeKernel(commandQueue, kernelPushGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
    }
    if (i < param.passes)
    {
        //reset getGradient
        err = clSetKernelArg(kernelGetGradient, 0, sizeof(cl_mem), &imageBuffer1);
        err |= clSetKernelArg(kernelGetGradient, 1, sizeof(cl_mem), &imageBuffer2);
        if (err != CL_SUCCESS)
            CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: reset getGradient error", err)
        //reset pushGradient
        err = clSetKernelArg(kernelPushGradient, 0, sizeof(cl_mem), &imageBuffer2);
        err |= clSetKernelArg(kernelPushGradient, 1, sizeof(cl_mem), &imageBuffer1);
        err |= clSetKernelArg(kernelPushGradient, 2, sizeof(cl_float), &pushGradientStrength);
        if (err != CL_SUCCESS)
            CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: reset pushGradient error", err)

        while (i++ < param.passes)
        {
            clEnqueueNDRangeKernel(commandQueue, kernelGetGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
            clEnqueueNDRangeKernel(commandQueue, kernelPushGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        }
    }
    clEnqueueMarkerWithWaitList(commandQueue, 0, nullptr, &readReadyEvent);
    clEnqueueReadImage(commandQueueIO, imageBuffer1, CL_FALSE, orgin, dstRegion, dstImg.step, 0, dstImg.data, 1, &readReadyEvent, &readFinishedEvent);

    clWaitForEvents(1, &readFinishedEvent);

    //clean
    clReleaseMemObject(imageBuffer3);
    clReleaseMemObject(imageBuffer2);
    clReleaseMemObject(imageBuffer1);
    clReleaseMemObject(imageBuffer0);

    clReleaseKernel(kernelGetGray);
    clReleaseKernel(kernelPushColor);
    clReleaseKernel(kernelGetGradient);
    clReleaseKernel(kernelPushGradient);

    clReleaseEvent(writeFinishedEvent);
    clReleaseEvent(readReadyEvent);
    clReleaseEvent(readFinishedEvent);
}

void Anime4KCPP::OpenCL::Anime4K09::runKernelPW(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    cl_int err = CL_SUCCESS;
    cl_event writeFinishedEvent = nullptr;
    cl_event readReadyEvent = nullptr;
    cl_event readFinishedEvent = nullptr;
    int i;

    cl_image_format format{};

    cl_image_desc dstDesc{};
    cl_image_desc orgDesc{};

    constexpr size_t orgin[3] = { 0,0,0 };
    const size_t orgRegion[3] = { static_cast<const size_t>(orgImg.cols),static_cast<const size_t>(orgImg.rows),1 };
    const size_t dstRegion[3] = { static_cast<const size_t>(dstImg.cols),static_cast<const size_t>(dstImg.rows),1 };
    const size_t size[2] =
    {
        (((static_cast<const size_t>(dstImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(dstImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };

    const cl_float pushColorStrength = static_cast<const cl_float>(param.strengthColor);
    const cl_float pushGradientStrength = static_cast<const cl_float>(param.strengthGradient);
    const cl_float normalizedWidth = static_cast<const cl_float>(nWidth);
    const cl_float normalizedHeight = static_cast<const cl_float>(nHeight);

    //init frame
    format.image_channel_data_type = CL_UNORM_INT16;
    format.image_channel_order = CL_RGBA;

    orgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    orgDesc.image_height = orgImg.rows;
    orgDesc.image_width = orgImg.cols;
    orgDesc.buffer = nullptr;

    dstDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    dstDesc.image_height = dstImg.rows;
    dstDesc.image_width = dstImg.cols;
    dstDesc.buffer = nullptr;

    cl_command_queue commandQueue = commandQueueList[commandQueueCount++];
    if (commandQueueCount >= commandQueueNum)
        commandQueueCount = 0;

    //kernel for each thread
    cl_kernel kernelGetGray = nullptr;
    if (param.zoomFactor == 2.0)
        kernelGetGray = clCreateKernel(program, "getGray", &err);
    else
        kernelGetGray = clCreateKernel(program, "getGrayLanczos4", &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel getGray", err);
    }
    cl_kernel kernelPushColor = clCreateKernel(program, "pushColor", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelGetGray);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel pushColor", err);
    }
    cl_kernel kernelGetGradient = clCreateKernel(program, "getGradient", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelGetGray);
        clReleaseKernel(kernelPushColor);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel getGradient", err);
    }
    cl_kernel kernelPushGradient = clCreateKernel(program, "pushGradient", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelGetGray);
        clReleaseKernel(kernelPushColor);
        clReleaseKernel(kernelGetGradient);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel pushGradient", err);
    }

    //imageBuffer
    //for getGray
    cl_mem imageBuffer0 = clCreateImage(context, CL_MEM_READ_ONLY, &format, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer0 error, video memory may be insufficient.", err);
    }
    //tmp buffer 1
    cl_mem imageBuffer1 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBuffer0);
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer1 error, video memory may be insufficient.", err);
    }
    //tmp buffer 2
    cl_mem imageBuffer2 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBuffer0);
        clReleaseMemObject(imageBuffer1);
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer2 error, video memory may be insufficient.", err);
    }
    //tmp buffer 3
    cl_mem imageBuffer3 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBuffer0);
        clReleaseMemObject(imageBuffer1);
        clReleaseMemObject(imageBuffer2);
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer3 error, video memory may be insufficient.", err);
    }

    //set arguments
    //getGray
    err = clSetKernelArg(kernelGetGray, 0, sizeof(cl_mem), &imageBuffer0);
    err |= clSetKernelArg(kernelGetGray, 1, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(kernelGetGray, 2, sizeof(cl_float), &normalizedWidth);
    err |= clSetKernelArg(kernelGetGray, 3, sizeof(cl_float), &normalizedHeight);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: getGray error", err)
    //pushColor
    err = clSetKernelArg(kernelPushColor, 0, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(kernelPushColor, 1, sizeof(cl_mem), &imageBuffer2);
    err |= clSetKernelArg(kernelPushColor, 2, sizeof(cl_float), &pushColorStrength);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: pushColor error", err)
    //getGradient
    err = clSetKernelArg(kernelGetGradient, 0, sizeof(cl_mem), &imageBuffer2);
    err |= clSetKernelArg(kernelGetGradient, 1, sizeof(cl_mem), &imageBuffer3);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: getGradient error", err)
    //pushGradient
    err = clSetKernelArg(kernelPushGradient, 0, sizeof(cl_mem), &imageBuffer3);
    err |= clSetKernelArg(kernelPushGradient, 1, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(kernelPushGradient, 2, sizeof(cl_float), &pushGradientStrength);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: pushGradient error", err)

    //enqueue
    clEnqueueWriteImage(commandQueueIO, imageBuffer0, CL_FALSE, orgin, orgRegion, orgImg.step, 0, orgImg.data, 0, nullptr, &writeFinishedEvent);
    clEnqueueNDRangeKernel(commandQueue, kernelGetGray, 2, nullptr, size, nullptr, 1, &writeFinishedEvent, nullptr);
    for (i = 0; i < param.passes && i < param.pushColorCount; i++)//pcc for push color count
    {
        clEnqueueNDRangeKernel(commandQueue, kernelPushColor, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        clEnqueueNDRangeKernel(commandQueue, kernelGetGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        clEnqueueNDRangeKernel(commandQueue, kernelPushGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
    }
    if (i < param.passes)
    {
        //reset getGradient
        err = clSetKernelArg(kernelGetGradient, 0, sizeof(cl_mem), &imageBuffer1);
        err |= clSetKernelArg(kernelGetGradient, 1, sizeof(cl_mem), &imageBuffer2);
        if (err != CL_SUCCESS)
            CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: reset getGradient error", err)
        //reset pushGradient
        err = clSetKernelArg(kernelPushGradient, 0, sizeof(cl_mem), &imageBuffer2);
        err |= clSetKernelArg(kernelPushGradient, 1, sizeof(cl_mem), &imageBuffer1);
        err |= clSetKernelArg(kernelPushGradient, 2, sizeof(cl_float), &pushGradientStrength);
        if (err != CL_SUCCESS)
            CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: reset pushGradient error", err)

        while (i++ < param.passes)
        {
            clEnqueueNDRangeKernel(commandQueue, kernelGetGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
            clEnqueueNDRangeKernel(commandQueue, kernelPushGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        }
    }
    clEnqueueMarkerWithWaitList(commandQueue, 0, nullptr, &readReadyEvent);
    clEnqueueReadImage(commandQueueIO, imageBuffer1, CL_FALSE, orgin, dstRegion, dstImg.step, 0, dstImg.data, 1, &readReadyEvent, &readFinishedEvent);

    clWaitForEvents(1, &readFinishedEvent);

    //clean
    clReleaseMemObject(imageBuffer3);
    clReleaseMemObject(imageBuffer2);
    clReleaseMemObject(imageBuffer1);
    clReleaseMemObject(imageBuffer0);

    clReleaseKernel(kernelGetGray);
    clReleaseKernel(kernelPushColor);
    clReleaseKernel(kernelGetGradient);
    clReleaseKernel(kernelPushGradient);

    clReleaseEvent(writeFinishedEvent);
    clReleaseEvent(readReadyEvent);
    clReleaseEvent(readFinishedEvent);
}

void Anime4KCPP::OpenCL::Anime4K09::runKernelPF(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    cl_int err = CL_SUCCESS;
    cl_event writeFinishedEvent = nullptr;
    cl_event readReadyEvent = nullptr;
    cl_event readFinishedEvent = nullptr;
    int i;

    cl_image_format format{};

    cl_image_desc dstDesc{};
    cl_image_desc orgDesc{};

    constexpr size_t orgin[3] = { 0,0,0 };
    const size_t orgRegion[3] = { static_cast<const size_t>(orgImg.cols),static_cast<const size_t>(orgImg.rows),1 };
    const size_t dstRegion[3] = { static_cast<const size_t>(dstImg.cols),static_cast<const size_t>(dstImg.rows),1 };
    const size_t size[2] =
    {
        (((static_cast<const size_t>(dstImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(dstImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };

    const cl_float pushColorStrength = static_cast<const cl_float>(param.strengthColor);
    const cl_float pushGradientStrength = static_cast<const cl_float>(param.strengthGradient);
    const cl_float normalizedWidth = static_cast<const cl_float>(nWidth);
    const cl_float normalizedHeight = static_cast<const cl_float>(nHeight);

    //init frame
    format.image_channel_data_type = CL_FLOAT;
    format.image_channel_order = CL_RGBA;

    orgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    orgDesc.image_height = orgImg.rows;
    orgDesc.image_width = orgImg.cols;
    orgDesc.buffer = nullptr;

    dstDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    dstDesc.image_height = dstImg.rows;
    dstDesc.image_width = dstImg.cols;
    dstDesc.buffer = nullptr;

    cl_command_queue commandQueue = commandQueueList[commandQueueCount++];
    if (commandQueueCount >= commandQueueNum)
        commandQueueCount = 0;

    //kernel for each thread
    cl_kernel kernelGetGray = nullptr;
    if (param.zoomFactor == 2.0)
        kernelGetGray = clCreateKernel(program, "getGray", &err);
    else
        kernelGetGray = clCreateKernel(program, "getGrayLanczos4", &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel getGray", err);
    }
    cl_kernel kernelPushColor = clCreateKernel(program, "pushColor", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelGetGray);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel pushColor", err);
    }
    cl_kernel kernelGetGradient = clCreateKernel(program, "getGradient", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelGetGray);
        clReleaseKernel(kernelPushColor);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel getGradient", err);
    }
    cl_kernel kernelPushGradient = clCreateKernel(program, "pushGradient", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelGetGray);
        clReleaseKernel(kernelPushColor);
        clReleaseKernel(kernelGetGradient);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel pushGradient", err);
    }

    //imageBuffer
    //for getGray
    cl_mem imageBuffer0 = clCreateImage(context, CL_MEM_READ_ONLY, &format, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer0 error, video memory may be insufficient.", err);
    }
    //tmp buffer 1
    cl_mem imageBuffer1 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBuffer0);
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer1 error, video memory may be insufficient.", err);
    }
    //tmp buffer 2
    cl_mem imageBuffer2 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBuffer0);
        clReleaseMemObject(imageBuffer1);
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer2 error, video memory may be insufficient.", err);
    }
    //tmp buffer 3
    cl_mem imageBuffer3 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBuffer0);
        clReleaseMemObject(imageBuffer1);
        clReleaseMemObject(imageBuffer2);
        throw ACException<ExceptionType::GPU, true>("Request imageBuffer3 error, video memory may be insufficient.", err);
    }

    //set arguments
    //getGray
    err = clSetKernelArg(kernelGetGray, 0, sizeof(cl_mem), &imageBuffer0);
    err |= clSetKernelArg(kernelGetGray, 1, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(kernelGetGray, 2, sizeof(cl_float), &normalizedWidth);
    err |= clSetKernelArg(kernelGetGray, 3, sizeof(cl_float), &normalizedHeight);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: getGray error", err)
    //pushColor
    err = clSetKernelArg(kernelPushColor, 0, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(kernelPushColor, 1, sizeof(cl_mem), &imageBuffer2);
    err |= clSetKernelArg(kernelPushColor, 2, sizeof(cl_float), &pushColorStrength);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: pushColor error", err)
    //getGradient
    err = clSetKernelArg(kernelGetGradient, 0, sizeof(cl_mem), &imageBuffer2);
    err |= clSetKernelArg(kernelGetGradient, 1, sizeof(cl_mem), &imageBuffer3);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: getGradient error", err)
    //pushGradient
    err = clSetKernelArg(kernelPushGradient, 0, sizeof(cl_mem), &imageBuffer3);
    err |= clSetKernelArg(kernelPushGradient, 1, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(kernelPushGradient, 2, sizeof(cl_float), &pushGradientStrength);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: pushGradient error", err)

    //enqueue
    clEnqueueWriteImage(commandQueueIO, imageBuffer0, CL_FALSE, orgin, orgRegion, orgImg.step, 0, orgImg.data, 0, nullptr, &writeFinishedEvent);
    clEnqueueNDRangeKernel(commandQueue, kernelGetGray, 2, nullptr, size, nullptr, 1, &writeFinishedEvent, nullptr);
    for (i = 0; i < param.passes && i < param.pushColorCount; i++)//pcc for push color count
    {
        clEnqueueNDRangeKernel(commandQueue, kernelPushColor, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        clEnqueueNDRangeKernel(commandQueue, kernelGetGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        clEnqueueNDRangeKernel(commandQueue, kernelPushGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
    }
    if (i < param.passes)
    {
        //reset getGradient
        err = clSetKernelArg(kernelGetGradient, 0, sizeof(cl_mem), &imageBuffer1);
        err |= clSetKernelArg(kernelGetGradient, 1, sizeof(cl_mem), &imageBuffer2);
        if (err != CL_SUCCESS)
            CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: reset getGradient error", err)
        //reset pushGradient
        err = clSetKernelArg(kernelPushGradient, 0, sizeof(cl_mem), &imageBuffer2);
        err |= clSetKernelArg(kernelPushGradient, 1, sizeof(cl_mem), &imageBuffer1);
        err |= clSetKernelArg(kernelPushGradient, 2, sizeof(cl_float), &pushGradientStrength);
        if (err != CL_SUCCESS)
            CLEAN_KERNEL_AND_THROW_ERROR("clSetKernelArg: reset pushGradient error", err)

        while (i++ < param.passes)
        {
            clEnqueueNDRangeKernel(commandQueue, kernelGetGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
            clEnqueueNDRangeKernel(commandQueue, kernelPushGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        }
    }
    clEnqueueMarkerWithWaitList(commandQueue, 0, nullptr, &readReadyEvent);
    clEnqueueReadImage(commandQueueIO, imageBuffer1, CL_FALSE, orgin, dstRegion, dstImg.step, 0, dstImg.data, 1, &readReadyEvent, &readFinishedEvent);

    clWaitForEvents(1, &readFinishedEvent);

    //clean
    clReleaseMemObject(imageBuffer3);
    clReleaseMemObject(imageBuffer2);
    clReleaseMemObject(imageBuffer1);
    clReleaseMemObject(imageBuffer0);

    clReleaseKernel(kernelGetGray);
    clReleaseKernel(kernelPushColor);
    clReleaseKernel(kernelGetGradient);
    clReleaseKernel(kernelPushGradient);

    clReleaseEvent(writeFinishedEvent);
    clReleaseEvent(readReadyEvent);
    clReleaseEvent(readFinishedEvent);
}

void Anime4KCPP::OpenCL::Anime4K09::initOpenCL()
{
    cl_int err = CL_SUCCESS;
    cl_uint platforms = 0;
    cl_uint devices = 0;
    cl_platform_id currentplatform = nullptr;

    //init platform
    err = clGetPlatformIDs(0, nullptr, &platforms);
    if (err != CL_SUCCESS || !platforms)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to find OpenCL platform", err);
    }

    cl_platform_id* tmpPlatform = new cl_platform_id[platforms];
    err = clGetPlatformIDs(platforms, tmpPlatform, nullptr);
    if (err != CL_SUCCESS)
    {
        delete[] tmpPlatform;
        throw ACException<ExceptionType::GPU, true>("Failed to get OpenCL platform", err);
    }

    if (pID >= 0 && pID < (int)platforms)
        currentplatform = tmpPlatform[pID];
    else
        currentplatform = tmpPlatform[pID = 0];

    delete[] tmpPlatform;

    //init device
    err = clGetDeviceIDs(currentplatform, CL_DEVICE_TYPE_GPU, 0, nullptr, &devices);
    if (err != CL_SUCCESS || !devices)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to find supported GPU", err);
    }

    cl_device_id* tmpDevice = new cl_device_id[devices];
    err = clGetDeviceIDs(currentplatform, CL_DEVICE_TYPE_GPU, devices, tmpDevice, nullptr);
    if (err != CL_SUCCESS)
    {
        delete[] tmpDevice;
        throw ACException<ExceptionType::GPU, true>("GPU initialization error", err);
    }

    if (dID >= 0 && dID < (int)devices)
        device = tmpDevice[dID];
    else
        device = tmpDevice[dID = 0];

    delete[] tmpDevice;

    //init context
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        releaseOpenCL();
        throw ACException<ExceptionType::GPU, true>("Failed to create context", err);
    }

    //init command queue
    commandQueueList.resize(commandQueueNum, nullptr);
#ifndef CL_VERSION_2_0 //for OpenCL SDK older than v2.0 to build
    for (int i = 0; i < commandQueueNum; i++)
    {
        commandQueueList[i] = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS)
        {
            releaseOpenCL();
            throw ACException<ExceptionType::GPU, true>("Failed to create command queue", err);
        }
    }
    if (parallelIO)
    {
        commandQueueIO = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS)
        {
            releaseOpenCL();
            throw ACException<ExceptionType::GPU, true>("Failed to create command queue", err);
        }
    }

#else
    for (int i = 0; i < commandQueueNum; i++)
    {
#ifdef LEGACY_OPENCL_API
        commandQueueList[i] = clCreateCommandQueue(context, device, 0, &err);
#else
        commandQueueList[i] = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
#endif
        if (err != CL_SUCCESS)
        {
            releaseOpenCL();
            throw ACException<ExceptionType::GPU, true>("Failed to create command queue", err);
        }
    }
    if (parallelIO)
    {
#ifdef LEGACY_OPENCL_API
        commandQueueIO = clCreateCommandQueue(context, device, 0, &err);
#else
        commandQueueIO = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
#endif
        if (err != CL_SUCCESS)
        {
            releaseOpenCL();
            throw ACException<ExceptionType::GPU, true>("Failed to create command queue", err);
        }
    }
#endif

#ifndef BUILT_IN_KERNEL
    //read kernel files
    std::string Anime4KCPPKernelSourceString = readKernel("Anime4KCPPKernel.cl");
    const char* Anime4KCPPKernelSource = Anime4KCPPKernelSourceString.c_str();
#else
    const char* Anime4KCPPKernelSource = Anime4KCPPKernelSourceString;
#endif // BUILT_IN_KERNEL

    //create program
    program = clCreateProgramWithSource(context, 1, &Anime4KCPPKernelSource, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        releaseOpenCL();
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL program", err);
    }

#ifdef ENABLE_FAST_MATH
    const char* buildFlags = "-cl-fast-relaxed-math";
#else
    const char* buildFlags = nullptr;
#endif // ENABLE_FAST_MATH
    //build program
    err = clBuildProgram(program, 1, &device, buildFlags, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        size_t buildErrorSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &buildErrorSize);
        char* buildError = new char[buildErrorSize];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, buildErrorSize, buildError, nullptr);
        releaseOpenCL();
        ACException<ExceptionType::GPU, true> exception("Kernel build error", buildError, err);
        delete[] buildError;
        throw exception;
    }

    cl_kernel tmpKernel = clCreateKernel(program, "pushColor", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(tmpKernel);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel for getting workGroupSizeLog", err);
    }
    err = clGetKernelWorkGroupInfo(tmpKernel, device,
        CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), (void*)&workGroupSizeLog, nullptr);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(tmpKernel);
        throw ACException<ExceptionType::GPU, true>("Failed to get workGroupSize", err);
    }
    workGroupSizeLog = std::log2(workGroupSizeLog);
    clReleaseKernel(tmpKernel);
}

void Anime4KCPP::OpenCL::Anime4K09::releaseOpenCL() noexcept
{
    for (auto& commandQueue : commandQueueList)
    {
        if (commandQueue != nullptr)
            clReleaseCommandQueue(commandQueue);
    }
    if (commandQueueIO != nullptr)
        clReleaseCommandQueue(commandQueueIO);
    if (program != nullptr)
        clReleaseProgram(program);
    if (context != nullptr)
        clReleaseContext(context);
}

std::string Anime4KCPP::OpenCL::Anime4K09::readKernel(const std::string& fileName)
{
    std::ifstream kernelFile(fileName);
    if (!kernelFile.is_open())
        throw ACException<ExceptionType::IO>("Failed to open kernel file.");

    std::ostringstream source;
    source << kernelFile.rdbuf();

    return source.str();
}

Anime4KCPP::Processor::Type Anime4KCPP::OpenCL::Anime4K09::getProcessorType() noexcept
{
    return Processor::Type::OpenCL_Anime4K09;
}

std::string Anime4KCPP::OpenCL::Anime4K09::getProcessorInfo()
{
    cl_int err = 0;
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;

    size_t platformNameLength = 0;
    size_t deviceNameLength = 0;

    auto tmpPlatform = std::make_unique<cl_platform_id[]>(static_cast<size_t>(pID) + 1);
    err = clGetPlatformIDs(pID + 1, tmpPlatform.get(), nullptr);
    if (err != CL_SUCCESS)
        throw ACException<ExceptionType::GPU, true>("Failed to find OpenCL platforms.", err);

    platform = tmpPlatform[pID];

    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameLength);
    if (err != CL_SUCCESS)
        throw ACException<ExceptionType::GPU, true>("Failed to get OpenCL platform information.", err);

    auto platformName = std::make_unique<char[]>(platformNameLength);
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameLength, platformName.get(), nullptr);
    if (err != CL_SUCCESS)
        throw ACException<ExceptionType::GPU, true>("Failed to get OpenCL platform information.", err);

    auto tmpDevice = std::make_unique<cl_device_id[]>(static_cast<size_t>(dID) + 1);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, dID + 1, tmpDevice.get(), nullptr);
    if (err != CL_SUCCESS)
        throw ACException<ExceptionType::GPU, true>("Failed to find OpenCL devices.", err);

    device = tmpDevice[dID];

    err = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameLength);
    if (err != CL_SUCCESS)
        throw ACException<ExceptionType::GPU, true>("Failed to get OpenCL devices information.", err);

    auto deviceName = std::make_unique<char[]>(deviceNameLength);
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameLength, deviceName.get(), nullptr);
    if (err != CL_SUCCESS)
        throw ACException<ExceptionType::GPU, true>("Failed to get OpenCL devices information.", err);

    std::ostringstream oss;
    oss << "Processor type: " << getProcessorType() << std::endl
        << "Current OpenCL devices:" << std::endl
        << " Platform " + std::to_string(pID) + ": " + platformName.get() << std::endl
        << "  Device " + std::to_string(dID) + ": " + deviceName.get();
    return oss.str();
}
