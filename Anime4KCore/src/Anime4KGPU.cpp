#include "Anime4KGPU.h"

Anime4KGPU::Anime4KGPU(
    int passes,
    int pushColorCount,
    double strengthColor,
    double strengthGradient,
    double zoomFactor,
    bool fastMode,
    bool videoMode,
    bool PreProcessing,
    bool postProcessing,
    uint8_t preFilters,
    uint8_t postFilters,
    unsigned int maxThreads
) :Anime4K(
    passes,
    pushColorCount,
    strengthColor,
    strengthGradient,
    zoomFactor,
    fastMode,
    videoMode,
    PreProcessing,
    postProcessing,
    preFilters,
    postFilters,
    maxThreads
),
context(nullptr), commandQueue(nullptr),
program(nullptr), device(nullptr),
kernelGetGray(nullptr), kernelPushColor(nullptr),
kernelGetGradient(nullptr), kernelPushGradient(nullptr)
{
    initOpenCL();
}

Anime4KGPU::~Anime4KGPU()
{
    releaseOpenCL();
}

void Anime4KGPU::process()
{
    //init
    format.image_channel_data_type = CL_UNORM_INT8;
    format.image_channel_order = CL_RGBA;
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_height = H;
    desc.image_width = W;
    desc.image_row_pitch = 0;
    desc.image_slice_pitch = 0;
    desc.num_mip_levels = 0;
    desc.num_samples = 0;
    desc.buffer = nullptr;

    if (!vm)
    {
        cv::resize(orgImg, dstImg, cv::Size(0, 0), zf, zf, cv::INTER_CUBIC);
        if (pre)
            FilterProcessor(dstImg, pref).process();
        cv::cvtColor(dstImg, dstImg, cv::COLOR_BGR2BGRA);
        runKernel(dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
        if (post)//PostProcessing
            FilterProcessor(dstImg, postf).process();
    }
    else
    {
        cv::Mat orgFrame, dstFrame;
        std::queue<cv::Mat> frames;
        std::queue<cv::Mat> results;
        bool breakFlag = false;
        int size = std::ceil(fps);
        while (true)
        {
            for (int i = 0; i < size; i++)
            {
                if(video.read(orgFrame))
                {
                    cv::resize(orgFrame, dstFrame, cv::Size(0, 0), zf, zf, cv::INTER_CUBIC);
                    cv::cvtColor(dstFrame, dstFrame, cv::COLOR_BGR2BGRA);
                    frames.emplace(dstFrame);
                }
                else
                {
                    breakFlag = true;
                    size = i;
                    break;
                }
            }

            runKernelForVideo(frames,results);

            for (int i = 0; i < size; i++)
            {
                dstFrame = results.front();
                cv::cvtColor(dstFrame, dstFrame, cv::COLOR_BGRA2BGR);
                videoWriter.write(dstFrame);
                results.pop();
            }
            if (breakFlag)
                break;
        }
        //cv::Mat orgFrame, dstFrame;
        //ThreadPool pool(mt);
        //size_t curFrame = 0;
        //while (true)
        //{
        //    curFrame = video.get(cv::CAP_PROP_POS_FRAMES);
        //    if (!video.read(orgFrame))
        //    {
        //        while (frameCount < totalFrameCount)
        //            std::this_thread::yield;
        //        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        //        break;
        //    }

        //    pool.exec<std::function<void()>>([orgFrame = orgFrame.clone(), dstFrame = dstFrame.clone(), this, curFrame, tmpPcc = this->pcc]()mutable
        //    {
        //        cv::resize(orgFrame, dstFrame, cv::Size(0, 0), zf, zf, cv::INTER_CUBIC);
        //        if (pre)
        //            FilterProcessor(dstFrame, pref).process();
        //        cv::cvtColor(dstFrame, dstFrame, cv::COLOR_BGR2BGRA);
        //        runKernel(dstFrame);
        //        cv::cvtColor(dstFrame, dstFrame, cv::COLOR_BGRA2BGR);
        //        if (post)//PostProcessing
        //            FilterProcessor(dstFrame, postf).process();
        //        std::unique_lock<std::mutex> lock(videoMtx);
        //        while (true)
        //        {
        //            if (curFrame == frameCount)
        //            {
        //                videoWriter.write(dstFrame);
        //                frameCount++;
        //                break;
        //            }
        //            else
        //            {
        //                cnd.wait(lock);
        //            }
        //        }
        //        cnd.notify_all();
        //    });
        //}
    }
}

void Anime4KGPU::runKernel(cv::InputArray img)
{
    cl_int err;
    const size_t orgin[3] = { 0,0,0 };
    const size_t region[3] = { W,H,1 };
    const size_t size[2] = { W,H };
    cv::Mat image = img.getMat();

    cl_mem imagebuffer1 = clCreateImage(context, CL_MEM_READ_WRITE , &format, &desc, nullptr, &err);
    if (err != CL_SUCCESS)
        throw"imagebuffer1 error";
    cl_mem imagebuffer2 = clCreateImage(context, CL_MEM_READ_WRITE , &format, &desc, nullptr, &err);
    if (err != CL_SUCCESS)
        throw"imagebuffer2 error";

    //getGray
    err = clSetKernelArg(kernelGetGray, 0, sizeof(cl_mem), &imagebuffer1);
    err = clSetKernelArg(kernelGetGray, 1, sizeof(cl_mem), &imagebuffer2);
    if (err != CL_SUCCESS)
        throw"getGray clSetKernelArg error";
    //pushColor
    err = clSetKernelArg(kernelPushColor, 0, sizeof(cl_mem), &imagebuffer2);
    err = clSetKernelArg(kernelPushColor, 1, sizeof(cl_mem), &imagebuffer1);
    if (err != CL_SUCCESS)
        throw"pushColor clSetKernelArg error";
    //getGradient
    err = clSetKernelArg(kernelGetGradient, 0, sizeof(cl_mem), &imagebuffer1);
    err = clSetKernelArg(kernelGetGradient, 1, sizeof(cl_mem), &imagebuffer2);
    if (err != CL_SUCCESS)
        throw"getGradient clSetKernelArg error";
    //pushGradient
    err = clSetKernelArg(kernelPushGradient, 0, sizeof(cl_mem), &imagebuffer2);
    err = clSetKernelArg(kernelPushGradient, 1, sizeof(cl_mem), &imagebuffer1);
    if (err != CL_SUCCESS)
        throw"pushGradient clSetKernelArg error";
    
    clEnqueueWriteImage(commandQueue, imagebuffer1, CL_FALSE, orgin, region, image.step, 0, image.data, 0, nullptr, nullptr);
    for (int i = 0; i < ps; i++)
    {
        clEnqueueNDRangeKernel(commandQueue, kernelGetGray, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        clEnqueueNDRangeKernel(commandQueue, kernelPushColor, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        clEnqueueNDRangeKernel(commandQueue, kernelGetGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        clEnqueueNDRangeKernel(commandQueue, kernelPushGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
    }
    clEnqueueReadImage(commandQueue, imagebuffer1, CL_TRUE, orgin, region, image.step, 0, image.data, 0, nullptr, nullptr);

    clReleaseMemObject(imagebuffer2);
    clReleaseMemObject(imagebuffer1);
}

void Anime4KGPU::runKernelForVideo(std::queue<cv::Mat> &frames, std::queue<cv::Mat>& results)
{
    cl_int err;
    const size_t orgin[3] = { 0,0,0 };
    const size_t region[3] = { W,H,1 };
    const size_t size[2] = { W,H };
    
    int frameNum = frames.size();
    
    cl_mem imagebuffer1 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, nullptr, &err);
    if (err != CL_SUCCESS)
        throw"imagebuffer1 error";
    cl_mem imagebuffer2 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, nullptr, &err);
    if (err != CL_SUCCESS)
        throw"imagebuffer2 error";

    //getGray
    err = clSetKernelArg(kernelGetGray, 0, sizeof(cl_mem), &imagebuffer1);
    err = clSetKernelArg(kernelGetGray, 1, sizeof(cl_mem), &imagebuffer2);
    if (err != CL_SUCCESS)
        throw"getGray clSetKernelArg error";
    //pushColor
    err = clSetKernelArg(kernelPushColor, 0, sizeof(cl_mem), &imagebuffer2);
    err = clSetKernelArg(kernelPushColor, 1, sizeof(cl_mem), &imagebuffer1);
    if (err != CL_SUCCESS)
        throw"pushColor clSetKernelArg error";
    //getGradient
    err = clSetKernelArg(kernelGetGradient, 0, sizeof(cl_mem), &imagebuffer1);
    err = clSetKernelArg(kernelGetGradient, 1, sizeof(cl_mem), &imagebuffer2);
    if (err != CL_SUCCESS)
        throw"getGradient clSetKernelArg error";
    //pushGradient
    err = clSetKernelArg(kernelPushGradient, 0, sizeof(cl_mem), &imagebuffer2);
    err = clSetKernelArg(kernelPushGradient, 1, sizeof(cl_mem), &imagebuffer1);
    if (err != CL_SUCCESS)
        throw"pushGradient clSetKernelArg error";

    for (int i = 0; i < frameNum; i++)
    {
        cv::Mat currFrame = frames.front();
        clEnqueueWriteImage(commandQueue, imagebuffer1, CL_FALSE, orgin, region, currFrame.step, 0, currFrame.data, 0, nullptr, nullptr);
        for (int i = 0; i < ps; i++)
        {
            clEnqueueNDRangeKernel(commandQueue, kernelGetGray, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
            clEnqueueNDRangeKernel(commandQueue, kernelPushColor, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
            clEnqueueNDRangeKernel(commandQueue, kernelGetGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
            clEnqueueNDRangeKernel(commandQueue, kernelPushGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        }
        clEnqueueReadImage(commandQueue, imagebuffer1, CL_FALSE, orgin, region, currFrame.step, 0, currFrame.data, 0, nullptr, nullptr);
        results.emplace(currFrame);
        frames.pop();
    }

    clFinish(commandQueue);

    clReleaseMemObject(imagebuffer2);
    clReleaseMemObject(imagebuffer1);
}

inline void Anime4KGPU::initOpenCL()
{
    cl_int err = 0;
    cl_uint plateforms = 0;
    cl_platform_id currentPlateform = nullptr;
    //init plateform
    err = clGetPlatformIDs(1, &currentPlateform, &plateforms);
    if (err != CL_SUCCESS || !plateforms)
        throw"Failed to find OpenCL plateform";
    //init device
    err = clGetDeviceIDs(currentPlateform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS)
        throw"Unsupport GPU";
    //init context
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        releaseOpenCL();
        throw"Failed to create context";
    }
    //init command queue
    commandQueue = clCreateCommandQueueWithProperties(context, device, 0, nullptr);
    if (commandQueue == nullptr)
    {
        releaseOpenCL();
        throw"Failed to create command queue";
    }

    //read kernel files
    std::string Anime4KCPPKernelSourceString = readKernel("Anime4KCPPKernel.cl");
    const char* Anime4KCPPKernelSource = Anime4KCPPKernelSourceString.c_str();

    //create program
    program = clCreateProgramWithSource(context, 1, &Anime4KCPPKernelSource, nullptr, nullptr);
    if (program == nullptr)
    {
        releaseOpenCL();
        throw"Failed to create OpenCL program";
    }

    //build program
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        releaseOpenCL();
        throw"Kernel build error";
    }

    kernelGetGray = clCreateKernel(program, "getGray", nullptr);
    kernelPushColor = clCreateKernel(program, "pushColor", nullptr);
    kernelGetGradient = clCreateKernel(program, "getGradient", nullptr);
    kernelPushGradient = clCreateKernel(program, "pushGradient", nullptr);

    if (kernelGetGray == nullptr || kernelPushColor == nullptr || kernelGetGradient == nullptr || kernelPushGradient == nullptr)
    {
        releaseOpenCL();
        throw"Failed to create OpenCL kernel";
    }
}

void Anime4KGPU::releaseOpenCL()
{
    if (kernelGetGray != nullptr)
        clReleaseKernel(kernelGetGray);
    if (kernelPushColor != nullptr)
        clReleaseKernel(kernelPushColor);
    if (kernelGetGradient != nullptr)
        clReleaseKernel(kernelGetGradient);
    if (kernelPushGradient != nullptr)
        clReleaseKernel(kernelPushGradient);
    if (program != nullptr)
        clReleaseProgram(program);
    if (context != nullptr)
        clReleaseContext(context);
    if (commandQueue != nullptr)
        clReleaseCommandQueue(commandQueue);
}

std::string Anime4KGPU::readKernel(const std::string& fileName)
{
    std::ifstream kernelFile(fileName);
    if (!kernelFile.is_open())
        throw"Read kernel error";
    std::ostringstream source;
    source << kernelFile.rdbuf();
    return std::string(source.str());
}
