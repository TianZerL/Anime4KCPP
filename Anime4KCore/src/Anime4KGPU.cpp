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
    unsigned int maxThreads,
    unsigned int platformID,
    unsigned int deviceID
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
program(nullptr), device(nullptr), frameGPUDoneCount(0),
pID(platformID), dID(deviceID)
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

    if (zf == 2.0)
    {
        nWidth = 1.0 / double(W);
        nHeight = 1.0 / double(H);
    }
    else
    {
        nWidth = double(orgW) / double(W);
        nHeight = double(orgH) / double(H);
    }

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

            pool.exec<std::function<void()>>([orgFrame = orgFrame.clone(), this, curFrame]()mutable
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

std::pair<std::pair<int, std::vector<int>>, std::string> Anime4KGPU::listGPUs()
{
    cl_int err = 0;
    cl_uint platforms = 0;
    cl_uint devices = 0;
    cl_platform_id* platform = nullptr;
    cl_device_id* device = nullptr;

    size_t platformNameLength = 0;
    size_t DeviceNameLength = 0;
    char* platformName = nullptr;
    char* DeviceName = nullptr;

    std::ostringstream GPUInfo;

    std::vector<int> devicesVector;

    err = clGetPlatformIDs(0, nullptr, &platforms);
    if (err != CL_SUCCESS || !platforms)
        return std::pair<std::pair<int, std::vector<int>>, std::string>({ 0,{0} }, "No suppoted platform");

    platform = new cl_platform_id[platforms];
    err = clGetPlatformIDs(platforms, platform, nullptr);
    if (err != CL_SUCCESS)
    {
        delete[] platform;
        return std::pair<std::pair<int, std::vector<int>>, std::string>({ 0,{0} }, "inital platform error");
    }

    for (cl_uint i = 0; i < platforms; i++)
    {
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, 0, nullptr, &platformNameLength);
        if (err != CL_SUCCESS)
        {
            delete[] platform;
            return std::pair<std::pair<int, std::vector<int>>, std::string>({ 0,{0} }, "Failed to get platform name length infomation");
        }


        platformName = new char[platformNameLength];
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, platformNameLength, platformName, nullptr);
        if (err != CL_SUCCESS)
        {
            delete[] platformName;
            delete[] platform;
            return std::pair<std::pair<int, std::vector<int>>, std::string>({ 0,{0} }, "Failed to get platform name infomation");
        }
        GPUInfo << "Platform " << i << ": " << platformName << std::endl;

        delete[] platformName;

        err = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &devices);
        if (err != CL_SUCCESS || !devices)
        {
            delete[] platform;
            return std::pair<std::pair<int, std::vector<int>>, std::string>({ 0,{0} }, "No supported GPU");
        }

        devicesVector.push_back(devices);

        device = new cl_device_id[devices];
        err = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_GPU, devices, device, nullptr);
        if (err != CL_SUCCESS)
        {
            delete[] device;
            delete[] platform;
            return std::pair<std::pair<int, std::vector<int>>, std::string>({ 0,{0} }, "inital GPU error");
        }

        for (cl_uint j = 0; j < devices; j++)
        {
            err = clGetDeviceInfo(device[j], CL_DEVICE_NAME, 0, nullptr, &DeviceNameLength);
            if (err != CL_SUCCESS)
            {
                clReleaseDevice(device[j]);
                delete[] device;
                delete[] platform;
                return std::pair<std::pair<int, std::vector<int>>, std::string>({ 0,{0} }, "Failed to get device name length infomation");
            }


            DeviceName = new char[DeviceNameLength];
            err = clGetDeviceInfo(device[j], CL_DEVICE_NAME, DeviceNameLength, DeviceName, nullptr);
            if (err != CL_SUCCESS)
            {
                clReleaseDevice(device[j]);
                delete[] DeviceName;
                delete[] device;
                delete[] platform;
                return std::pair<std::pair<int, std::vector<int>>, std::string>({ 0,{0} }, "Failed to get device name infomation");
            }
            GPUInfo << "Device " << j << ": " << DeviceName << std::endl;

            delete[] DeviceName;
            clReleaseDevice(device[j]);
        }
        delete[] device;
    }

    std::pair<std::pair<int, std::vector<int>>, std::string> ret({platforms, devicesVector}, std::string(GPUInfo.str()));

    delete[] platform;

    return ret;
}

std::pair<bool, std::string> Anime4KGPU::checkGPUSupport(unsigned int pID, unsigned int dID)
{
    cl_int err = 0;
    cl_uint platforms = 0;
    cl_uint devices = 0;
    cl_platform_id firstPlatform = nullptr;
    cl_device_id device = nullptr;

    size_t platformNameLength = 0;
    size_t DeviceNameLength = 0;
    char* platformName = nullptr;
    char* DeviceName = nullptr;

    err = clGetPlatformIDs(0, nullptr, &platforms);
    if (err != CL_SUCCESS || !platforms)
        return std::pair<bool, std::string>(false, "No suppoted platform");

    cl_platform_id* tmpPlatform = new cl_platform_id[platforms];
    err = clGetPlatformIDs(platforms, tmpPlatform, nullptr);
    if (err != CL_SUCCESS)
    {
        delete[] tmpPlatform;
        return std::pair<bool, std::string>(false, "inital platform error");
    }


    if (pID >= 0 && pID < platforms)
        firstPlatform = tmpPlatform[pID];
    else
        firstPlatform = tmpPlatform[0];

    delete[] tmpPlatform;

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


    err = clGetDeviceIDs(firstPlatform, CL_DEVICE_TYPE_GPU, 0, nullptr, &devices);
    if (err != CL_SUCCESS || !devices)
    {
        delete[] platformName;
        return std::pair<bool, std::string>(false, "No supported GPU");
    }

    cl_device_id* tmpDevice = new cl_device_id[devices];
    err = clGetDeviceIDs(firstPlatform, CL_DEVICE_TYPE_GPU, devices, tmpDevice, nullptr);
    if (err != CL_SUCCESS)
    {
        delete[] platformName;
        delete[] tmpDevice;
        return std::pair<bool, std::string>(false, "No supported GPU");
    }

    if (dID >= 0 && dID < devices)
        device = tmpDevice[dID];
    else
        device = tmpDevice[0];

    delete[] tmpDevice;

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
        std::string("Platform: ") + platformName + "\n" + "Device: " + DeviceName);

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
    cl_kernel kernelGetGray = nullptr;
    if (zf == 2.0)
        kernelGetGray = clCreateKernel(program, "getGray", &err);
    else
        kernelGetGray = clCreateKernel(program, "getGrayLanczos4", &err);
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
    cl_uint platforms = 0;
    cl_uint devices = 0;
    cl_platform_id currentplatform = nullptr;

    //init platform
    err = clGetPlatformIDs(0, nullptr, &platforms);
    if (err != CL_SUCCESS || !platforms)
    {
        std::cout << err << std::endl;
        throw"Failed to find OpenCL platform";
    }
        
    cl_platform_id *tmpPlatform = new cl_platform_id[platforms];
    err = clGetPlatformIDs(platforms, tmpPlatform, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cout << err << std::endl;
        delete[] tmpPlatform;
        throw"Failed to get OpenCL platform";
    }


    if (pID >= 0 && pID < platforms)
        currentplatform = tmpPlatform[pID];
    else
        currentplatform = tmpPlatform[0];

    delete[] tmpPlatform;

    //init device
    err = clGetDeviceIDs(currentplatform, CL_DEVICE_TYPE_GPU, 0, nullptr, &devices);
    if (err != CL_SUCCESS || !devices)
    {
        std::cout << err << std::endl;
        throw"Failed to find supported GPU";
    }

    cl_device_id* tmpDevice = new cl_device_id[devices];
    err = clGetDeviceIDs(currentplatform, CL_DEVICE_TYPE_GPU, devices, tmpDevice, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cout << err << std::endl;
        delete[] tmpDevice;
        throw"inital GPU error";
    }

    if (dID >= 0 && dID < devices)
        device = tmpDevice[dID];
    else
        device = tmpDevice[0];

    delete[] tmpDevice;

    //init context
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        std::cout << err << std::endl;
        releaseOpenCL();
        throw"Failed to create context";
    }

    //init command queue
    commandQueue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        if (err == CL_INVALID_DEVICE)
        {
#pragma warning(disable:4996)
            commandQueue = clCreateCommandQueue(context, device, 0, &err);
            if (err != CL_SUCCESS)
            {
                std::cout << err << std::endl;
                releaseOpenCL();
                throw"Failed to create command queue";
            }
        }
        else
        {
            std::cout << err << std::endl;
            releaseOpenCL();
            throw"Failed to create command queue";
        }
    }

    //read kernel files
    std::string Anime4KCPPKernelSourceString = readKernel("Anime4KCPPKernel.cl");
    const char* Anime4KCPPKernelSource = Anime4KCPPKernelSourceString.c_str();

    //create program
    program = clCreateProgramWithSource(context, 1, &Anime4KCPPKernelSource, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        std::cout << err << std::endl;
        releaseOpenCL();
        throw"Failed to create OpenCL program";
    }

    //build program
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        size_t buildErrorSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &buildErrorSize);
        char* buildError = new char[buildErrorSize];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, buildErrorSize, buildError, nullptr);
        releaseOpenCL();
        //print build info
        std::cout << buildError << std::endl;
        delete[] buildError;
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
