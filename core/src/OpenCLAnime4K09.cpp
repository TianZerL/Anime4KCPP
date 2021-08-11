#ifdef ENABLE_OPENCL

#include<fstream>

#define CL_HPP_ENABLE_EXCEPTIONS
#ifdef LEGACY_OPENCL_API
#define CL_HPP_TARGET_OPENCL_VERSION 120
#else
#define CL_HPP_TARGET_OPENCL_VERSION 200
#endif // LEGACY_OPENCL_API
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#include<CL/opencl.hpp>

#include"FilterProcessor.hpp"
#include"OpenCLAnime4K09.hpp"
#include"OpenCLAnime4K09Kernel.hpp"

#define ALIGN_UP(x, size) (((x) + (size) - 1) & (~((size) - 1)))

namespace Anime4KCPP::OpenCL::detail
{
    //init OpenCL arguments
    static bool isInitializedFlag = false;
    static cl::Program program;
    static cl::Device device;
    static cl::Context context;
    static cl::CommandQueue commandQueueIO;
    static int commandQueueNum = 4;
    static int commandQueueCount = 0;
    static std::vector<cl::CommandQueue> commandQueueList(commandQueueNum);
    static bool parallelIO = false;
    static int pID = 0;
    static int dID = 0;
    static size_t workGroupSizeBase = 32;

    static std::string readKernel(const std::string& fileName)
    {
        std::ifstream kernelFile(fileName);
        if (!kernelFile.is_open())
            throw ACException<ExceptionType::IO>("Failed to open kernel file.");

        std::ostringstream source;
        source << kernelFile.rdbuf();

        return source.str();
    }

    static void runKernelN(const cv::Mat& orgImg, cv::Mat& dstImg, int channelType, const Parameters& param)
    {
        double nWidth, nHeight;
        if (param.zoomFactor == 2.0)
        {
            nWidth = 1.0 / dstImg.cols;
            nHeight = 1.0 / dstImg.rows;
        }
        else
        {
            nWidth = static_cast<double>(orgImg.cols) / dstImg.cols;
            nHeight = static_cast<double>(orgImg.rows) / dstImg.rows;
        }

        constexpr std::array<size_t, 3> orgin = { 0,0,0 };
        const std::array<size_t, 3> orgRegion = { static_cast<const size_t>(orgImg.cols),static_cast<const size_t>(orgImg.rows),1 };
        const std::array<size_t, 3> dstRegion = { static_cast<const size_t>(dstImg.cols),static_cast<const size_t>(dstImg.rows),1 };
        const std::array<size_t, 2> size =
        {
            ALIGN_UP(dstImg.cols, workGroupSizeBase),
            ALIGN_UP(dstImg.rows, workGroupSizeBase)
        };

        const cl_float pushColorStrength = static_cast<const cl_float>(param.strengthColor);
        const cl_float pushGradientStrength = static_cast<const cl_float>(param.strengthGradient);
        const cl_float normalizedWidth = static_cast<const cl_float>(nWidth);
        const cl_float normalizedHeight = static_cast<const cl_float>(nHeight);

        try
        {
            //init frame
            cl::CommandQueue& commandQueue = commandQueueList[commandQueueCount++];
            if (commandQueueCount >= commandQueueNum)
                commandQueueCount = 0;

            //kernel for each thread
            cl::Kernel kernelGetGray(program, (param.zoomFactor == 2.0) ? "getGray" : "getGrayLanczos4");
            cl::Kernel kernelPushColor(program, "pushColor");
            cl::Kernel kernelGetGradient(program, "getGradient");
            cl::Kernel kernelPushGradient(program, "pushGradient");

            cl::ImageFormat imageFormat(CL_RGBA, channelType);
            cl::ImageFormat tmpFormat(CL_RGBA, channelType == CL_FLOAT ? CL_HALF_FLOAT : channelType);

            //imageBuffer
            cl::Image2D imageBuffer0(context, CL_MEM_READ_ONLY, imageFormat, orgImg.cols, orgImg.rows);
            cl::Image2D imageBuffer1(context, CL_MEM_READ_WRITE, imageFormat, dstImg.cols, dstImg.rows);
            cl::Image2D imageBuffer2(context, CL_MEM_READ_WRITE, tmpFormat, dstImg.cols, dstImg.rows);
            cl::Image2D imageBuffer3(context, CL_MEM_READ_WRITE, tmpFormat, dstImg.cols, dstImg.rows);

            //set arguments
            //getGray
            kernelGetGray.setArg(0, sizeof(cl_mem), &imageBuffer0);
            kernelGetGray.setArg(1, sizeof(cl_mem), &imageBuffer1);
            kernelGetGray.setArg(2, sizeof(cl_float), &normalizedWidth);
            kernelGetGray.setArg(3, sizeof(cl_float), &normalizedHeight);

            //pushColor
            kernelPushColor.setArg(0, sizeof(cl_mem), &imageBuffer1);
            kernelPushColor.setArg(1, sizeof(cl_mem), &imageBuffer2);
            kernelPushColor.setArg(2, sizeof(cl_float), &pushColorStrength);

            //getGradient
            kernelGetGradient.setArg(0, sizeof(cl_mem), &imageBuffer2);
            kernelGetGradient.setArg(1, sizeof(cl_mem), &imageBuffer3);

            //pushGradient
            kernelPushGradient.setArg(0, sizeof(cl_mem), &imageBuffer3);
            kernelPushGradient.setArg(1, sizeof(cl_mem), &imageBuffer1);
            kernelPushGradient.setArg(2, sizeof(cl_float), &pushGradientStrength);

            int i;
            //enqueue
            commandQueue.enqueueWriteImage(imageBuffer0, CL_FALSE, orgin, orgRegion, orgImg.step, 0, orgImg.data);
            commandQueue.enqueueNDRangeKernel(kernelGetGray, cl::NullRange, cl::NDRange(size[0], size[1]));
            for (i = 0; i < param.passes && i < param.pushColorCount; i++)//pcc for push color count
            {
                commandQueue.enqueueNDRangeKernel(kernelPushColor, cl::NullRange, cl::NDRange(size[0], size[1]));
                commandQueue.enqueueNDRangeKernel(kernelGetGradient, cl::NullRange, cl::NDRange(size[0], size[1]));
                commandQueue.enqueueNDRangeKernel(kernelPushGradient, cl::NullRange, cl::NDRange(size[0], size[1]));
            }
            if (i < param.passes)
            {
                //reset getGradient
                kernelGetGradient.setArg(0, sizeof(cl_mem), &imageBuffer1);
                kernelGetGradient.setArg(1, sizeof(cl_mem), &imageBuffer2);
                //reset pushGradient
                kernelPushGradient.setArg(0, sizeof(cl_mem), &imageBuffer2);
                kernelPushGradient.setArg(1, sizeof(cl_mem), &imageBuffer1);
                kernelPushGradient.setArg(2, sizeof(cl_float), &pushGradientStrength);

                while (i++ < param.passes)
                {
                    commandQueue.enqueueNDRangeKernel(kernelGetGradient, cl::NullRange, cl::NDRange(size[0], size[1]));
                    commandQueue.enqueueNDRangeKernel(kernelPushGradient, cl::NullRange, cl::NDRange(size[0], size[1]));
                }
            }
            //blocking read
            commandQueue.enqueueReadImage(imageBuffer1, CL_TRUE, orgin, dstRegion, dstImg.step, 0, dstImg.data);
        }
        catch (const cl::Error& e)
        {
            throw ACException<ExceptionType::GPU, true>("Failed to run OpenCL Anime4K09 kernel", e.what(), e.err());
        }
    }

    static void runKernelP(const cv::Mat& orgImg, cv::Mat& dstImg, int channelType, const Parameters& param)
    {
        double nWidth, nHeight;
        if (param.zoomFactor == 2.0)
        {
            nWidth = 1.0 / dstImg.cols;
            nHeight = 1.0 / dstImg.rows;
        }
        else
        {
            nWidth = static_cast<double>(orgImg.cols) / dstImg.cols;
            nHeight = static_cast<double>(orgImg.rows) / dstImg.rows;
        }

        constexpr std::array<size_t, 3> orgin = { 0,0,0 };
        const std::array<size_t, 3> orgRegion = { static_cast<const size_t>(orgImg.cols),static_cast<const size_t>(orgImg.rows),1 };
        const std::array<size_t, 3> dstRegion = { static_cast<const size_t>(dstImg.cols),static_cast<const size_t>(dstImg.rows),1 };
        const std::array<size_t, 2> size =
        {
            ALIGN_UP(dstImg.cols, workGroupSizeBase),
            ALIGN_UP(dstImg.rows, workGroupSizeBase)
        };

        const cl_float pushColorStrength = static_cast<const cl_float>(param.strengthColor);
        const cl_float pushGradientStrength = static_cast<const cl_float>(param.strengthGradient);
        const cl_float normalizedWidth = static_cast<const cl_float>(nWidth);
        const cl_float normalizedHeight = static_cast<const cl_float>(nHeight);

        try
        {
            //init frame
            cl::CommandQueue& commandQueue = commandQueueList[commandQueueCount++];
            if (commandQueueCount >= commandQueueNum)
                commandQueueCount = 0;

            //kernel for each thread
            cl::Kernel kernelGetGray(program, (param.zoomFactor == 2.0) ? "getGray" : "getGrayLanczos4");
            cl::Kernel kernelPushColor(program, "pushColor");
            cl::Kernel kernelGetGradient(program, "getGradient");
            cl::Kernel kernelPushGradient(program, "pushGradient");

            cl::ImageFormat imageFormat(CL_RGBA, channelType);
            cl::ImageFormat tmpFormat(CL_RGBA, channelType == CL_FLOAT ? CL_HALF_FLOAT : channelType);

            //imageBuffer
            cl::Image2D imageBuffer0(context, CL_MEM_READ_ONLY, imageFormat, orgImg.cols, orgImg.rows);
            cl::Image2D imageBuffer1(context, CL_MEM_READ_WRITE, imageFormat, dstImg.cols, dstImg.rows);
            cl::Image2D imageBuffer2(context, CL_MEM_READ_WRITE, tmpFormat, dstImg.cols, dstImg.rows);
            cl::Image2D imageBuffer3(context, CL_MEM_READ_WRITE, tmpFormat, dstImg.cols, dstImg.rows);

            //set arguments
            //getGray
            kernelGetGray.setArg(0, sizeof(cl_mem), &imageBuffer0);
            kernelGetGray.setArg(1, sizeof(cl_mem), &imageBuffer1);
            kernelGetGray.setArg(2, sizeof(cl_float), &normalizedWidth);
            kernelGetGray.setArg(3, sizeof(cl_float), &normalizedHeight);

            //pushColor
            kernelPushColor.setArg(0, sizeof(cl_mem), &imageBuffer1);
            kernelPushColor.setArg(1, sizeof(cl_mem), &imageBuffer2);
            kernelPushColor.setArg(2, sizeof(cl_float), &pushColorStrength);

            //getGradient
            kernelGetGradient.setArg(0, sizeof(cl_mem), &imageBuffer2);
            kernelGetGradient.setArg(1, sizeof(cl_mem), &imageBuffer3);

            //pushGradient
            kernelPushGradient.setArg(0, sizeof(cl_mem), &imageBuffer3);
            kernelPushGradient.setArg(1, sizeof(cl_mem), &imageBuffer1);
            kernelPushGradient.setArg(2, sizeof(cl_float), &pushGradientStrength);

            int i;
            std::vector<cl::Event> waitForWriteFinishedEvent(1);
            std::vector<cl::Event> waitForReadReadyEvent(1);
            cl::Event& writeFinishedEvent = waitForWriteFinishedEvent.front();
            cl::Event& readReadyEvent = waitForReadReadyEvent.front();
            cl::Event  readFinishedEvent;

            //enqueue
            commandQueueIO.enqueueWriteImage(imageBuffer0, CL_FALSE, orgin, orgRegion, orgImg.step, 0, orgImg.data, nullptr, &writeFinishedEvent);
            commandQueue.enqueueNDRangeKernel(kernelGetGray, cl::NullRange, cl::NDRange(size[0], size[1]), cl::NullRange, &waitForWriteFinishedEvent);
            for (i = 0; i < param.passes && i < param.pushColorCount; i++)//pcc for push color count
            {
                commandQueue.enqueueNDRangeKernel(kernelPushColor, cl::NullRange, cl::NDRange(size[0], size[1]));
                commandQueue.enqueueNDRangeKernel(kernelGetGradient, cl::NullRange, cl::NDRange(size[0], size[1]));
                commandQueue.enqueueNDRangeKernel(kernelPushGradient, cl::NullRange, cl::NDRange(size[0], size[1]));
            }
            if (i < param.passes)
            {
                //reset getGradient
                kernelGetGradient.setArg(0, sizeof(cl_mem), &imageBuffer1);
                kernelGetGradient.setArg(1, sizeof(cl_mem), &imageBuffer2);
                //reset pushGradient
                kernelPushGradient.setArg(0, sizeof(cl_mem), &imageBuffer2);
                kernelPushGradient.setArg(1, sizeof(cl_mem), &imageBuffer1);
                kernelPushGradient.setArg(2, sizeof(cl_float), &pushGradientStrength);

                while (i++ < param.passes)
                {
                    commandQueue.enqueueNDRangeKernel(kernelGetGradient, cl::NullRange, cl::NDRange(size[0], size[1]));
                    commandQueue.enqueueNDRangeKernel(kernelPushGradient, cl::NullRange, cl::NDRange(size[0], size[1]));
                }
            }
            //blocking read
            commandQueue.enqueueMarkerWithWaitList(nullptr, &readReadyEvent);
            commandQueueIO.enqueueReadImage(imageBuffer1, CL_FALSE, orgin, dstRegion, dstImg.step, 0, dstImg.data, &waitForReadReadyEvent, &readFinishedEvent);
            readFinishedEvent.wait();
        }
        catch (const cl::Error& e)
        {
            throw ACException<ExceptionType::GPU, true>("Failed to run OpenCL Anime4K09 kernel", e.what(), e.err());
        }
    }

    static void runKernel(const cv::Mat& orgImg, cv::Mat& dstImg, const Parameters& param)
    {
        int channelType;
        switch (orgImg.depth())
        {
        case CV_8U:
            channelType = CL_UNORM_INT8;
            break;
        case CV_16U:
            channelType = CL_UNORM_INT16;
            break;
        case CV_32F:
            channelType = CL_FLOAT;
            break;
        default:
            throw ACException<ExceptionType::RunTimeError>("Unsupported image data type");
        }

        if (parallelIO)
            runKernelP(orgImg, dstImg, channelType, param);
        else
            runKernelN(orgImg, dstImg, channelType, param);
    }
}

void Anime4KCPP::OpenCL::Anime4K09::init(const int platformID, const int deviceID, const int OpenCLQueueNum, const bool OpenCLParallelIO)
{
    if (!detail::isInitializedFlag)
    {
        detail::pID = platformID;
        detail::dID = deviceID;
        detail::commandQueueNum = OpenCLQueueNum >= 1 ? OpenCLQueueNum : 1;
        detail::parallelIO = OpenCLParallelIO;
        initOpenCL();
        detail::isInitializedFlag = true;
    }
}

void Anime4KCPP::OpenCL::Anime4K09::release() noexcept
{
    if (detail::isInitializedFlag)
    {
        detail::context = nullptr;
        std::fill(detail::commandQueueList.begin(), detail::commandQueueList.end(), nullptr);
        detail::commandQueueIO = nullptr;
        detail::program = nullptr;
        detail::device = nullptr;
        detail::isInitializedFlag = false;
    }
}

bool Anime4KCPP::OpenCL::Anime4K09::isInitialized() noexcept
{
    return detail::isInitializedFlag;
}

std::string Anime4KCPP::OpenCL::Anime4K09::getInfo()
{
    std::ostringstream oss;
    oss << AC::getInfo()
        << "----------------------------------------------" << std::endl
        << "OpenCL Platform ID:" << detail::pID << std::endl
        << "OpenCL Device ID:" << detail::dID << std::endl
        << "Passes: " << param.passes << std::endl
        << "pushColorCount: " << param.pushColorCount << std::endl
        << "Zoom Factor: " << param.zoomFactor << std::endl
        << "Fast Mode: " << std::boolalpha << param.fastMode << std::endl
        << "Strength Color: " << param.strengthColor << std::endl
        << "Strength Gradient: " << param.strengthGradient << std::endl
        << "Number of OpenCL Command Queues:" << detail::commandQueueNum << std::endl
        << "OpenCL Parallel IO Command Queues:" << std::boolalpha << detail::parallelIO << std::endl
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

void Anime4KCPP::OpenCL::Anime4K09::processYUVImage()
{
    cv::Mat tmpImg;
    cv::merge(std::vector<cv::Mat>{ orgImg, orgU, orgV }, tmpImg);
    cv::cvtColor(tmpImg, tmpImg, cv::COLOR_YUV2BGR);

    dstImg.create(height , width, CV_MAKE_TYPE(tmpImg.depth() ,4));
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(tmpImg, param.preFilters).process();
    cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2BGRA);
    detail::runKernel(tmpImg, dstImg, param);
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
    if (param.postprocessing)//Postprocessing(CPU)
        FilterProcessor(dstImg, param.postFilters).process();

    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGR2YUV);
    std::vector<cv::Mat> yuv(3);
    cv::split(dstImg, yuv);
    dstImg = yuv[Y];
    dstU = yuv[U];
    dstV = yuv[V];
}

void Anime4KCPP::OpenCL::Anime4K09::processRGBImage()
{
    cv::Mat tmpImg = orgImg;
    dstImg.create(height, width, CV_MAKE_TYPE(tmpImg.depth(), 4));
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(tmpImg, param.preFilters).process();
    cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2BGRA);
    detail::runKernel(tmpImg, dstImg, param);
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
    if (param.postprocessing)//Postprocessing(CPU)
        FilterProcessor(dstImg, param.postFilters).process();
}

void Anime4KCPP::OpenCL::Anime4K09::processGrayscale()
{
    cv::Mat tmpImg;
    cv::cvtColor(orgImg, tmpImg, cv::COLOR_GRAY2BGR);

    dstImg.create(height, width, CV_MAKE_TYPE(tmpImg.depth(), 4));
    if (param.preprocessing)//Pretprocessing(CPU)
        FilterProcessor(tmpImg, param.preFilters).process();
    cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2BGRA);
    detail::runKernel(tmpImg, dstImg, param);
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
    if (param.postprocessing)//Postprocessing(CPU)
        FilterProcessor(dstImg, param.postFilters).process();

    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGR2GRAY);
}

void Anime4KCPP::OpenCL::Anime4K09::initOpenCL()
{
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;

    try
    {
        cl::Platform::get(&platforms);
        ((detail::pID >= 0 && detail::pID < platforms.size()) ? platforms[detail::pID] : platforms[0])
            .getDevices(CL_DEVICE_TYPE_ALL, &devices);

        detail::device = (detail::dID >= 0 && detail::dID < devices.size()) ? devices[detail::dID] : devices[0];

        detail::context = cl::Context(detail::device);

        detail::commandQueueList.resize(detail::commandQueueNum);
        for (int i = 0; i < detail::commandQueueNum; i++)
        {
            detail::commandQueueList[i] = cl::CommandQueue(detail::context, detail::device);
        }
        if (detail::parallelIO)
        {
            detail::commandQueueIO = cl::CommandQueue(detail::context, detail::device);
        }

#ifndef BUILT_IN_KERNEL
        //read kernel files
        std::string Anime4KCPPKernelSource = detail::readKernel("Anime4KCPPKernel.cl");
#else
        std::string Anime4KCPPKernelSource = Anime4KCPPKernelSourceString;
#endif // BUILT_IN_KERNEL
        detail::program = cl::Program(detail::context, Anime4KCPPKernelSource);

#ifdef ENABLE_FAST_MATH
        const char* buildFlags = "-cl-fast-relaxed-math";
#else
        const char* buildFlags = nullptr;
#endif // ENABLE_FAST_MATH
        try
        {
            detail::program.build(detail::device, buildFlags);
        }
        catch (const cl::BuildError& e)
        {
            throw ACException<ExceptionType::GPU, true>("Kernel build error", e.getBuildLog().front().second, e.err());
        }

        cl::Kernel tmpKernel{ detail::program, "pushColor" };
        tmpKernel.getWorkGroupInfo(detail::device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &detail::workGroupSizeBase);
    }
    catch (const cl::Error& e)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to initialize OpenCL", e.what(), e.err());
    }
}

Anime4KCPP::Processor::Type Anime4KCPP::OpenCL::Anime4K09::getProcessorType() noexcept
{
    return Processor::Type::OpenCL_Anime4K09;
}

std::string Anime4KCPP::OpenCL::Anime4K09::getProcessorInfo()
{
    std::ostringstream oss;
    oss << "Processor type: " << getProcessorType() << std::endl
        << "Current OpenCL devices:" << std::endl
        << " " + detail::device.getInfo<CL_DEVICE_NAME>();
    return oss.str();
}

#endif
