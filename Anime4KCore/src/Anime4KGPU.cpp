#define DLL

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
program(nullptr), device(nullptr), frameGPUDoneCount(0)
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

    orgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    orgDesc.image_height = orgH;
    orgDesc.image_width = orgW;
    orgDesc.image_row_pitch = 0;
    orgDesc.image_slice_pitch = 0;
    orgDesc.num_mip_levels = 0;
    orgDesc.num_samples = 0;
    orgDesc.buffer = nullptr;

    dstDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    dstDesc.image_height = H;
    dstDesc.image_width = W;
    dstDesc.image_row_pitch = 0;
    dstDesc.image_slice_pitch = 0;
    dstDesc.num_mip_levels = 0;
    dstDesc.num_samples = 0;
    dstDesc.buffer = nullptr;

    nWidth = 1.0 / double(W);
    nHeight = 1.0 / double(H);

    if (!vm)
    {
        dstImg.create(H, W, CV_8UC4);
        if (pre)//Pretprocessing(CPU)
            FilterProcessor(orgImg, pref).process();
        cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2BGRA);
        runKernel(orgImg, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
        if (post)//Postprocessing(CPU)
            FilterProcessor(dstImg, postf).process();
    }
    else
    {
        uint64_t count = mt;
        cv::Mat orgFrame;
        ThreadPool pool(mt);
        uint64_t curFrame = 0, doneFrameCount = 0;
        frameGPUDoneCount = frameCount = 0;
        while (true)
        {
            curFrame = video.get(cv::CAP_PROP_POS_FRAMES);
            if (!video.read(orgFrame))
            {
                while (frameCount < totalFrameCount)
                    std::this_thread::yield();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                break;
            }

            pool.exec<std::function<void()>>([orgFrame = orgFrame.clone(), this, curFrame, tmpPcc = this->pcc]()mutable
            {
                cv::Mat dstFrame(H, W, CV_8UC4);
                if (pre)
                    FilterProcessor(orgFrame, pref).process();
                cv::cvtColor(orgFrame, orgFrame, cv::COLOR_BGR2BGRA);
                runKernel(orgFrame, dstFrame);
                cv::cvtColor(dstFrame, dstFrame, cv::COLOR_BGRA2BGR);
                if (post)//PostProcessing
                    FilterProcessor(dstFrame, postf).process();
                std::unique_lock<std::mutex> lock(videoMtx);
                frameGPUDoneCount++;
                while (true)
                {
                    if (curFrame == frameCount)
                    {
                        videoWriter.write(dstFrame);
                        dstFrame.release();
                        frameCount++;
                        break;
                    }
                    else
                    {
                        cnd.wait(lock);
                    }
                }
                cnd.notify_all();
            });
            //limit RAM usage
            if (!(--count))
            {
                while (frameGPUDoneCount == doneFrameCount)
                    std::this_thread::yield();
                count = frameGPUDoneCount - doneFrameCount;
                doneFrameCount = frameGPUDoneCount;
            }
        }
    }
}

std::pair<bool, std::string> Anime4KGPU::checkGPUSupport()
{
    cl_int err = 0;
    cl_uint plateforms = 0;
    cl_uint devices = 0;
    cl_platform_id firstPlatform = nullptr;
    cl_device_id device = nullptr;

    size_t platformNameLength = 0;
    size_t DeviceNameLength = 0;
    char* platformName = nullptr;
    char* DeviceName = nullptr;

    err = clGetPlatformIDs(1, &firstPlatform, &plateforms);
    if (err != CL_SUCCESS || !plateforms)
        return std::pair<bool, std::string>(false, "No suppoted platform");

    err = clGetPlatformInfo(firstPlatform, CL_PLATFORM_NAME, 0, nullptr, &platformNameLength);
    if (err != CL_SUCCESS)
        return std::pair<bool, std::string>(false, "Failed to get platform name length infomation");

    platformName = new char[platformNameLength];
    err = clGetPlatformInfo(firstPlatform, CL_PLATFORM_NAME, platformNameLength, platformName, nullptr);
    if (err != CL_SUCCESS)
    {
        delete[] platformName;
        return std::pair<bool, std::string>(false, "Failed to get platform name infomation");
    }


    err = clGetDeviceIDs(firstPlatform, CL_DEVICE_TYPE_GPU, 1, &device, &devices);
    if (err != CL_SUCCESS || !devices)
    {
        delete[] platformName;
        return std::pair<bool, std::string>(false, "No supported GPU");
    }
        

    err = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &DeviceNameLength);
    if (err != CL_SUCCESS)
    {
        delete[] platformName;
        clReleaseDevice(device);
        return std::pair<bool, std::string>(false, "Failed to get device name length infomation");
    }


    DeviceName = new char[DeviceNameLength];
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, DeviceNameLength, DeviceName, nullptr);
    if (err != CL_SUCCESS)
    {
        delete[] DeviceName;
        delete[] platformName;
        clReleaseDevice(device);
        return std::pair<bool, std::string>(false, "Failed to get device name infomation");
    }

    std::pair<bool, std::string> ret(true,
        std::string("Plateform: ") + platformName + "\n" + "Device: " + DeviceName);

    delete[] DeviceName;
    delete[] platformName;
    clReleaseDevice(device);

    return ret;
}

void Anime4KGPU::runKernel(cv::InputArray orgImg, cv::OutputArray dstImg)
{
    cl_int err;
    int i;

    const size_t orgin[3] = { 0,0,0 };
    const size_t orgRegion[3] = { size_t(orgW),size_t(orgH),1 };
    const size_t dstRegion[3] = { size_t(W),size_t(H),1 };
    const size_t size[2] = { size_t(W),size_t(H) };

    const cl_float pushColorStrength = sc;
    const cl_float pushGradientStrength = sg;
    const cl_float normalizedWidth = cl_float(nWidth);
    const cl_float normalizedHeight = cl_float(nHeight);

    cv::Mat orgImage = orgImg.getMat();
    cv::Mat dstImage = dstImg.getMat();

    //kernel for each thread
    cl_kernel kernelGetGray = clCreateKernel(program, "getGray", &err);
    if (err != CL_SUCCESS)
    {
        throw"Failed to create OpenCL kernel getGray";
    }
    cl_kernel kernelPushColor = clCreateKernel(program, "pushColor", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelGetGray);
        throw"Failed to create OpenCL kernel pushColor";
    }
    cl_kernel kernelGetGradient = clCreateKernel(program, "getGradient", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelGetGray);
        clReleaseKernel(kernelPushColor);
        throw"Failed to create OpenCL kernel getGradient";
    }
    cl_kernel kernelPushGradient = clCreateKernel(program, "pushGradient", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelGetGray);
        clReleaseKernel(kernelPushColor);
        clReleaseKernel(kernelGetGradient);
        throw"Failed to create OpenCL kernel pushGradient";
    }

    //imageBuffer
    //for getGray
    cl_mem imageBuffer0 = clCreateImage(context, CL_MEM_READ_ONLY, &format, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        throw"imageBuffer0 error";
    }
    //tmp buffer 1
    cl_mem imageBuffer1 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        throw"imageBuffer1 error";
    }
    //tmp buffer 2
    cl_mem imageBuffer2 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBuffer1);
        throw"imageBuffer2 error";
    }
    //tmp buffer 3
    cl_mem imageBuffer3 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBuffer1);
        throw"imageBuffer3 error";
    }

    //set arguments
    //getGray
    err = clSetKernelArg(kernelGetGray, 0, sizeof(cl_mem), &imageBuffer0);
    err |= clSetKernelArg(kernelGetGray, 1, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(kernelGetGray, 2, sizeof(cl_float), &normalizedWidth);
    err |= clSetKernelArg(kernelGetGray, 3, sizeof(cl_float), &normalizedHeight);
    if (err != CL_SUCCESS)
        throw"getGray clSetKernelArg error";
    //pushColor
    err = clSetKernelArg(kernelPushColor, 0, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(kernelPushColor, 1, sizeof(cl_mem), &imageBuffer2);
    err |= clSetKernelArg(kernelPushColor, 2, sizeof(cl_float), &pushColorStrength);
    if (err != CL_SUCCESS)
        throw"pushColor clSetKernelArg error";
    //getGradient
    err = clSetKernelArg(kernelGetGradient, 0, sizeof(cl_mem), &imageBuffer2);
    err |= clSetKernelArg(kernelGetGradient, 1, sizeof(cl_mem), &imageBuffer3);
    if (err != CL_SUCCESS)
        throw"getGradient clSetKernelArg error";
    //pushGradient
    err = clSetKernelArg(kernelPushGradient, 0, sizeof(cl_mem), &imageBuffer3);
    err |= clSetKernelArg(kernelPushGradient, 1, sizeof(cl_mem), &imageBuffer1);
    err |= clSetKernelArg(kernelPushGradient, 2, sizeof(cl_float), &pushGradientStrength);
    if (err != CL_SUCCESS)
        throw"pushGradient clSetKernelArg error";

    //enqueue
    clEnqueueWriteImage(commandQueue, imageBuffer0, CL_FALSE, orgin, orgRegion, orgImage.step, 0, orgImage.data, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelGetGray, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
    for (i = 0; i < ps && i < pcc; i++)//pcc for push color count
    {
        clEnqueueNDRangeKernel(commandQueue, kernelPushColor, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        clEnqueueNDRangeKernel(commandQueue, kernelGetGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        clEnqueueNDRangeKernel(commandQueue, kernelPushGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
    }
    if (i < ps)
    {
        //reset getGradient
        err = clSetKernelArg(kernelGetGradient, 0, sizeof(cl_mem), &imageBuffer1);
        err |= clSetKernelArg(kernelGetGradient, 1, sizeof(cl_mem), &imageBuffer2);
        if (err != CL_SUCCESS)
            throw"Reset getGradient clSetKernelArg error";
        //reset pushGradient
        err = clSetKernelArg(kernelPushGradient, 0, sizeof(cl_mem), &imageBuffer2);
        err |= clSetKernelArg(kernelPushGradient, 1, sizeof(cl_mem), &imageBuffer1);
        err |= clSetKernelArg(kernelPushGradient, 2, sizeof(cl_float), &pushGradientStrength);
        if (err != CL_SUCCESS)
            throw"Reset pushGradient clSetKernelArg error";

        while (i++ < ps)
        {
            clEnqueueNDRangeKernel(commandQueue, kernelGetGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
            clEnqueueNDRangeKernel(commandQueue, kernelPushGradient, 2, nullptr, size, nullptr, 0, nullptr, nullptr);
        }
    }
    //blocking read
    clEnqueueReadImage(commandQueue, imageBuffer1, CL_TRUE, orgin, dstRegion, dstImage.step, 0, dstImage.data, 0, nullptr, nullptr);

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

}

void Anime4KGPU::releaseOpenCL()
{
    if (program != nullptr)
        clReleaseProgram(program);
    if (commandQueue != nullptr)
        clReleaseCommandQueue(commandQueue);
    if (context != nullptr)
        clReleaseContext(context);
    if (device != nullptr)
        clReleaseDevice(device);
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
