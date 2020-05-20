#define DLL

#include "Anime4KGPU.h"

Anime4KCPP::Anime4KGPU::Anime4KGPU(const Parameters& parameters) :
    Anime4K(parameters),
    format(), dstDesc(), orgDesc(),
    nWidth(0.0), nHeight(0.0) {}

void Anime4KCPP::Anime4KGPU::process()
{
    initProcess();

    if (zf == 2.0)
    {
        nWidth = 1.0 / static_cast<double>(W);
        nHeight = 1.0 / static_cast<double>(H);
    }
    else
    {
        nWidth = static_cast<double>(orgW) / static_cast<double>(W);
        nHeight = static_cast<double>(orgH) / static_cast<double>(H);
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
        frameCount = 0;
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
                while (frameCount == doneFrameCount)
                    std::this_thread::yield();
                count = frameCount - doneFrameCount;
                doneFrameCount = frameCount;
            }
        }
    }
}

void Anime4KCPP::Anime4KGPU::initGPU(unsigned int platformID, unsigned int deviceID)
{
    if (!isInitialized)
    {
        pID = platformID;
        dID = deviceID;
        initOpenCL();
        isInitialized = true;
    }
}

void Anime4KCPP::Anime4KGPU::releaseGPU()
{
    if (isInitialized)
    {
        releaseOpenCL();
        context = nullptr;
        commandQueue = nullptr;
        program = nullptr;
        device = nullptr;
        isInitialized = false;
    }
}

bool Anime4KCPP::Anime4KGPU::isInitializedGPU()
{
    return isInitialized;
}

std::pair<std::pair<int, std::vector<int>>, std::string> Anime4KCPP::Anime4KGPU::listGPUs()
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

std::pair<bool, std::string> Anime4KCPP::Anime4KGPU::checkGPUSupport(unsigned int pID, unsigned int dID)
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

inline void Anime4KCPP::Anime4KGPU::initProcess()
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
}

void Anime4KCPP::Anime4KGPU::runKernel(cv::InputArray orgImg, cv::OutputArray dstImg)
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

void Anime4KCPP::Anime4KGPU::initOpenCL()
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
        throw"GPU initialization error";
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
    
#ifndef CL_VERSION_2_0 //for OpenCL SDK older than v2.0 to build
    commandQueue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS)
    {
        std::cout << err << std::endl;
        releaseOpenCL();
        throw"Failed to create command queue";
    }
#else
    commandQueue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        if (err == CL_INVALID_DEVICE)//for GPUs that only support OpenCL1.2
        {
#ifdef _MSC_VER
#pragma warning (disable: 4996)// this is for building in MSVC
#endif // _MSCV_VER
            //do not worry about this warning, it is for compatibility
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
#endif // SPECIAL OPENCL VERSION

#ifndef BUILT_IN_KERNEL
    //read kernel files
    std::string Anime4KCPPKernelSourceString = readKernel("Anime4KCPPKernel.cl");
#endif // BUILT_IN_KERNEL
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

void Anime4KCPP::Anime4KGPU::releaseOpenCL()
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

std::string Anime4KCPP::Anime4KGPU::readKernel(const std::string& fileName)
{
    std::ifstream kernelFile(fileName);
    if (!kernelFile.is_open())
        throw"Read kernel error";
    std::ostringstream source;
    source << kernelFile.rdbuf();
    return std::string(source.str());
}

//init OpenCL arguments
bool Anime4KCPP::Anime4KGPU::isInitialized = false;
cl_context Anime4KCPP::Anime4KGPU::context = nullptr;
cl_command_queue Anime4KCPP::Anime4KGPU::commandQueue = nullptr;
cl_program Anime4KCPP::Anime4KGPU::program = nullptr;
cl_device_id Anime4KCPP::Anime4KGPU::device = nullptr;
unsigned int Anime4KCPP::Anime4KGPU::pID = 0U;
unsigned int Anime4KCPP::Anime4KGPU::dID = 0U;

#ifdef BUILT_IN_KERNEL
const std::string Anime4KCPP::Anime4KGPU::Anime4KCPPKernelSourceString =
R"(#define MAX3(a, b, c) fmax(fmax(a,b),c)
#define MIN3(a, b, c) fmin(fmin(a,b),c)

#define RANGE 12.56637061436f

__constant sampler_t samplerN = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
__constant sampler_t samplerL = CLK_NORMALIZED_COORDS_TRUE  | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

inline static void getLightest(float4 *mc, float4 *a, float4 *b, float4 *c, float strength)
{
    (*mc) = mad((native_divide((*a) + (*b) + (*c), 3.0f) - (*mc)), strength, (*mc));
}

inline static void getAVerage(float4 *mc, float4 *a, float4 *b, float4 *c, float strength)
{
    (*mc).xyz = mad((native_divide((*a).xyz + (*b).xyz + (*c).xyz, 3.0f) - (*mc).xyz), strength, (*mc).xyz);
    (*mc).w = 0.299f * (*mc).z + 0.587f * (*mc).y + 0.114f * (*mc).x;
}

inline static float Lanczos4(float x)
{
    if(x == 0.0f)
        return 1.0f;
    x *= M_PI_F;
    if(x >= -RANGE && x < RANGE)
        return native_divide(4.0f * native_sin(x) * native_sin(x * 0.25f), x * x);
    else
        return 0.0f;
}

__kernel void getGrayLanczos4(__read_only image2d_t srcImg, __write_only image2d_t dstImg, float nWidth, float nHeight) 
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(dstImg) || y >= get_image_height(dstImg))
        return;

    const int2 coord = (int2)(x, y);
    const float2 scale = (float2)(nWidth, nHeight);
    const float2 xy = ((convert_float2(coord) + 0.5f) * scale) - 0.5f;
    const float2 fxy = floor(xy);

    float4 mc = (0.0f);

    #pragma unroll 8
    for(float sx = fxy.x - 3.0f; sx <= fxy.x + 4.0f; sx += 1.0f)
    {
        float coeffX = Lanczos4(xy.x - sx);
        mc += 
        read_imagef(srcImg, samplerN, (float2)(sx, fxy.y - 3.0f)) * coeffX * Lanczos4(xy.y - fxy.y + 3.0f) +
        read_imagef(srcImg, samplerN, (float2)(sx, fxy.y - 2.0f)) * coeffX * Lanczos4(xy.y - fxy.y + 2.0f) +
        read_imagef(srcImg, samplerN, (float2)(sx, fxy.y - 1.0f)) * coeffX * Lanczos4(xy.y - fxy.y + 1.0f) +
        read_imagef(srcImg, samplerN, (float2)(sx, fxy.y - 0.0f)) * coeffX * Lanczos4(xy.y - fxy.y + 0.0f) +
        read_imagef(srcImg, samplerN, (float2)(sx, fxy.y + 1.0f)) * coeffX * Lanczos4(xy.y - fxy.y - 1.0f) +
        read_imagef(srcImg, samplerN, (float2)(sx, fxy.y + 2.0f)) * coeffX * Lanczos4(xy.y - fxy.y - 2.0f) +
        read_imagef(srcImg, samplerN, (float2)(sx, fxy.y + 3.0f)) * coeffX * Lanczos4(xy.y - fxy.y - 3.0f) +
        read_imagef(srcImg, samplerN, (float2)(sx, fxy.y + 4.0f)) * coeffX * Lanczos4(xy.y - fxy.y - 4.0f);
    }

    //gray
    mc.w = 0.299f * mc.z  + 0.587f * mc.y  + 0.114f * mc.x;

    write_imagef(dstImg, coord, mc);
}

__kernel void getGray(__read_only image2d_t srcImg, __write_only image2d_t dstImg, float nWidth, float nHeight) 
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(dstImg) || y >= get_image_height(dstImg))
        return;

    const int2 coord = (int2)(x, y);

    float4 mc = read_imagef(srcImg, samplerL, (convert_float2(coord) + 0.5f) * (float2)(nWidth, nHeight));

    //gray
    mc.w = 0.299f * mc.z  + 0.587f * mc.y  + 0.114f * mc.x;

    write_imagef(dstImg, coord, mc);
}

__kernel void pushColor(__read_only image2d_t srcImg, __write_only image2d_t dstImg, float strength)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(srcImg) || y >= get_image_height(srcImg))
        return;

    int2 coord = (int2)(x, y);

    float4 tl = read_imagef(srcImg, samplerN, (int2)(x-1,y-1));
    float4 tc = read_imagef(srcImg, samplerN, (int2)(x,y-1));
    float4 tr = read_imagef(srcImg, samplerN, (int2)(x+1,y-1));
    float4 ml = read_imagef(srcImg, samplerN, (int2)(x-1,y));
    float4 mc = read_imagef(srcImg, samplerN, coord);
    float4 mr = read_imagef(srcImg, samplerN, (int2)(x+1,y));
    float4 bl = read_imagef(srcImg, samplerN, (int2)(x-1,y+1));
    float4 bc = read_imagef(srcImg, samplerN, (int2)(x,y+1));
    float4 br = read_imagef(srcImg, samplerN, (int2)(x+1,y+1));

    float maxD,minL;

    //top and bottom
    maxD = MAX3(bl.w, bc.w, br.w);
    minL = MIN3(tl.w, tc.w, tr.w);
    if (minL > mc.w && mc.w > maxD)
        getLightest(&mc, &tl, &tc, &tr, strength);
    else
    {
        maxD = MAX3(tl.w, tc.w, tr.w);
        minL = MIN3(bl.w, bc.w, br.w);
        if (minL > mc.w && mc.w > maxD)
            getLightest(&mc, &bl, &bc, &br, strength);
    }

    //sundiagonal
    maxD = MAX3(ml.w, mc.w, bc.w);
    minL = MIN3(tc.w, tr.w, mr.w);
    if (minL > maxD)
        getLightest(&mc, &tc, &tr, &mr, strength);
    else
    {
        maxD = MAX3(tc.w, mc.w, mr.w);
        minL = MIN3(ml.w, bl.w, bc.w);
        if (minL > maxD)
            getLightest(&mc, &ml, &bl, &bc, strength);
    }

    //left and right
    maxD = MAX3(tl.w, ml.w, bl.w);
    minL = MIN3(tr.w, mr.w, br.w);
    if (minL > mc.w && mc.w > maxD)
        getLightest(&mc, &tr, &mr, &br, strength);
    else
    {
        maxD = MAX3(tr.w, mr.w, br.w);
        minL = MIN3(tl.w, ml.w, bl.w);
        if (minL > mc.w && mc.w > maxD)
            getLightest(&mc, &tl, &ml, &bl, strength);
    }

    //diagonal
    maxD = MAX3(tc.w, mc.w, ml.w);
    minL = MIN3(mr.w, br.w, bc.w);
    if (minL > maxD)
        getLightest(&mc, &mr, &br, &bc, strength);
    else
    {
        maxD = MAX3(bc.w, mc.w, mr.w);
        minL = MIN3(ml.w, tl.w, tc.w);
        if (minL > maxD)
            getLightest(&mc, &ml, &tl, &tc, strength);
    }
    
    write_imagef(dstImg, coord, mc);
}

__kernel void getGradient(__read_only image2d_t srcImg, __write_only image2d_t dstImg)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(srcImg) || y >= get_image_height(srcImg))
        return;

    int2 coord = (int2)(x, y);

    float4 tl = read_imagef(srcImg, samplerN, (int2)(x-1,y-1));
    float4 tc = read_imagef(srcImg, samplerN, (int2)(x,y-1));
    float4 tr = read_imagef(srcImg, samplerN, (int2)(x+1,y-1));
    float4 ml = read_imagef(srcImg, samplerN, (int2)(x-1,y));
    float4 mc = read_imagef(srcImg, samplerN, coord);
    float4 mr = read_imagef(srcImg, samplerN, (int2)(x+1,y));
    float4 bl = read_imagef(srcImg, samplerN, (int2)(x-1,y+1));
    float4 bc = read_imagef(srcImg, samplerN, (int2)(x,y+1));
    float4 br = read_imagef(srcImg, samplerN, (int2)(x+1,y+1));

    const float gradX = tr.w + mr.w + mr.w + br.w - tl.w - ml.w - ml.w - bl.w;
    const float gradY = tl.w + tc.w + tc.w + tr.w - bl.w - bc.w - bc.w - br.w;

    const float grad = clamp(native_sqrt(gradX * gradX + gradY * gradY), 0.0f, 1.0f);
    mc.w = 1.0f - grad;

    write_imagef(dstImg, coord, mc);
}

__kernel void pushGradient(__read_only image2d_t srcImg, __write_only image2d_t dstImg, float strength)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(srcImg) || y >= get_image_height(srcImg))
        return;

    int2 coord = (int2)(x, y);

    float4 tl = read_imagef(srcImg, samplerN, (int2)(x-1,y-1));
    float4 tc = read_imagef(srcImg, samplerN, (int2)(x,y-1));
    float4 tr = read_imagef(srcImg, samplerN, (int2)(x+1,y-1));
    float4 ml = read_imagef(srcImg, samplerN, (int2)(x-1,y));
    float4 mc = read_imagef(srcImg, samplerN, coord);
    float4 mr = read_imagef(srcImg, samplerN, (int2)(x+1,y));
    float4 bl = read_imagef(srcImg, samplerN, (int2)(x-1,y+1));
    float4 bc = read_imagef(srcImg, samplerN, (int2)(x,y+1));
    float4 br = read_imagef(srcImg, samplerN, (int2)(x+1,y+1));

    float maxD,minL;

    //top and bottom
    maxD = MAX3(bl.w, bc.w, br.w);
    minL = MIN3(tl.w, tc.w, tr.w);
    if (minL > mc.w && mc.w > maxD)
    {
        getAVerage(&mc, &tl, &tc, &tr, strength);
        write_imagef(dstImg, coord, mc);
        return;
    }   

    maxD = MAX3(tl.w, tc.w, tr.w);
    minL = MIN3(bl.w, bc.w, br.w);
    if (minL > mc.w && mc.w > maxD)
    {
        getAVerage(&mc, &bl, &bc, &br, strength);
        write_imagef(dstImg, coord, mc);
        return;
    }

    //sundiagonal
    maxD = MAX3(ml.w, mc.w, bc.w);
    minL = MIN3(tc.w, tr.w, mr.w);
    if (minL > maxD)
    {
        getAVerage(&mc, &tc, &tr, &mr, strength);
        write_imagef(dstImg, coord, mc);
        return;
    }

    maxD = MAX3(tc.w, mc.w, mr.w);
    minL = MIN3(ml.w, bl.w, bc.w);
    if (minL > maxD)
    {
        getAVerage(&mc, &ml, &bl, &bc, strength);
        write_imagef(dstImg, coord, mc);
        return;
    }

    //left and right
    maxD = MAX3(tl.w, ml.w, bl.w);
    minL = MIN3(tr.w, mr.w, br.w);
    if (minL > mc.w && mc.w > maxD)
    {
        getAVerage(&mc, &tr, &mr, &br, strength);
        write_imagef(dstImg, coord, mc);
        return;
    }

    maxD = MAX3(tr.w, mr.w, br.w);
    minL = MIN3(tl.w, ml.w, bl.w);
    if (minL > mc.w && mc.w > maxD)
    {
        getAVerage(&mc, &tl, &ml, &bl, strength);
        write_imagef(dstImg, coord, mc);
        return;
    }

    //diagonal
    maxD = MAX3(tc.w, mc.w, ml.w);
    minL = MIN3(mr.w, br.w, bc.w);
    if (minL > maxD)
    {
        getAVerage(&mc, &mr, &br, &bc, strength);
        write_imagef(dstImg, coord, mc);
        return;
    }
    maxD = MAX3(bc.w, mc.w, mr.w);
    minL = MIN3(ml.w, tl.w, tc.w);
    if (minL > maxD)
    {
        getAVerage(&mc, &ml, &tl, &tc, strength);
        write_imagef(dstImg, coord, mc);
        return;
    }

    mc.w = 0.299f * mc.z + 0.587f * mc.y + 0.114f * mc.x;
    write_imagef(dstImg, coord, mc);
})";
#endif // BUILT_IN_KERNEL