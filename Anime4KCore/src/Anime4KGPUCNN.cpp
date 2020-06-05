#include "Anime4KGPUCNN.h"

Anime4KCPP::Anime4KGPUCNN::Anime4KGPUCNN(const Parameters& parameters) :
    Anime4K(parameters) {}

void Anime4KCPP::Anime4KGPUCNN::process()
{
    double tmpZf = log2(zf);
    int tmpZfUp = ceil(tmpZf);
    if (!vm)
    {
        cv::Mat tmpImg = orgImg;
        cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2YUV);
        std::vector<cv::Mat> yuv(3);
        cv::split(tmpImg, yuv);
        tmpImg = yuv[Y];
        for (int i = 0; i < tmpZfUp; i++)
        {
            dstImg.create(tmpImg.rows * 2, tmpImg.cols * 2, CV_8UC1);
            runKernel(tmpImg, dstImg);
            cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
            cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
            tmpImg = dstImg;
        }
        cv::merge(std::vector{ dstImg,yuv[U],yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
        if (tmpZfUp - tmpZf > 0.00001)
        {
            cv::resize(dstImg, dstImg, cv::Size(W, H), 0, 0, cv::INTER_LANCZOS4);
        }
    }
    else
    {
        VideoIO::instance().init(
            [this, tmpZfUp, tmpZf]()
            {
                Frame frame = VideoIO::instance().read();
                cv::Mat orgFrame = frame.first;
                cv::Mat dstFrame;

                cv::Mat tmpFrame = orgFrame;
                cv::cvtColor(tmpFrame, tmpFrame, cv::COLOR_BGR2YUV);
                std::vector<cv::Mat> yuv(3);
                cv::split(tmpFrame, yuv);
                tmpFrame = yuv[Y];
                for (int i = 0; i < tmpZfUp; i++)
                {
                    dstFrame.create(tmpFrame.rows * 2, tmpFrame.cols * 2, CV_8UC1);
                    runKernel(tmpFrame, dstFrame);
                    cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
                    cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
                    tmpFrame = dstFrame;
                }
                cv::merge(std::vector{ dstFrame,yuv[U],yuv[V] }, dstFrame);
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
}

void Anime4KCPP::Anime4KGPUCNN::initGPU(unsigned int platformID, unsigned int deviceID)
{
    if (!isInitialized)
    {
        pID = platformID;
        dID = deviceID;
        initOpenCL();
        isInitialized = true;
    }
}

void Anime4KCPP::Anime4KGPUCNN::releaseGPU()
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

bool Anime4KCPP::Anime4KGPUCNN::isInitializedGPU()
{
    return isInitialized;
}

void Anime4KCPP::Anime4KGPUCNN::runKernel(cv::InputArray orgImg, cv::OutputArray dstImg)
{
    cl_int err;
    int i;

    cv::Mat orgImage = orgImg.getMat();
    cv::Mat dstImage = dstImg.getMat();

    cl_image_format format;
    cl_image_format tmpFormat;
    cl_image_desc dstDesc;
    cl_image_desc orgDesc;

    const size_t orgin[3] = { 0,0,0 };
    const size_t orgRegion[3] = { size_t(orgImage.cols),size_t(orgImage.rows),1 };
    const size_t dstRegion[3] = { size_t(dstImage.cols),size_t(dstImage.rows),1 };
    const size_t orgSize[2] = { size_t(orgImage.cols),size_t(orgImage.rows) };
    const size_t dstSize[2] = { size_t(dstImage.cols),size_t(dstImage.rows) };
    
    const cl_int L2 = 0, L3 = 1, L4 = 2, L5 = 3, L6 = 4, L7 = 5, L8 = 6, L9 = 7;

    //init frame
    format.image_channel_data_type = CL_UNORM_INT8;
    format.image_channel_order = CL_R;

    tmpFormat.image_channel_data_type = CL_FLOAT;
    tmpFormat.image_channel_order = CL_RGBA;

    orgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    orgDesc.image_height = orgImage.rows;
    orgDesc.image_width = orgImage.cols;
    orgDesc.image_row_pitch = 0;
    orgDesc.image_slice_pitch = 0;
    orgDesc.num_mip_levels = 0;
    orgDesc.num_samples = 0;
    orgDesc.buffer = nullptr;

    dstDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    dstDesc.image_height = dstImage.rows;
    dstDesc.image_width = dstImage.cols;
    dstDesc.image_row_pitch = 0;
    dstDesc.image_slice_pitch = 0;
    dstDesc.num_mip_levels = 0;
    dstDesc.num_samples = 0;
    dstDesc.buffer = nullptr;

    cl_kernel kernelConv1To8L1 = clCreateKernel(program, "conv1To8", &err);
    if (err != CL_SUCCESS)
    {
        throw"Failed to create OpenCL kernel L1";
    }
    cl_kernel kernelConv8To8L2 = clCreateKernel(program, "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        throw"Failed to create OpenCL kernel L2";
    }
    cl_kernel kernelConv8To8L3 = clCreateKernel(program, "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        throw"Failed to create OpenCL kernel L3";
    }
    cl_kernel kernelConv8To8L4 = clCreateKernel(program, "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        throw"Failed to create OpenCL kernel L4";
    }
    cl_kernel kernelConv8To8L5 = clCreateKernel(program, "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        throw"Failed to create OpenCL kernel L5";
    }
    cl_kernel kernelConv8To8L6 = clCreateKernel(program, "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        throw"Failed to create OpenCL kernel L6";
    }
    cl_kernel kernelConv8To8L7 = clCreateKernel(program, "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        throw"Failed to create OpenCL kernel L7";
    }
    cl_kernel kernelConv8To8L8 = clCreateKernel(program, "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        throw"Failed to create OpenCL kernel L8";
    }
    cl_kernel kernelConv8To8L9 = clCreateKernel(program, "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        throw"Failed to create OpenCL kernel L9";
    }
    cl_kernel kernelConvTranspose8To1L10 = clCreateKernel(program, "convTranspose8To1", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        clReleaseKernel(kernelConv8To8L9);
        throw"Failed to create OpenCL kernel L10";
    }


    cl_mem imageBufferOrg = clCreateImage(context, CL_MEM_READ_ONLY, &format, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        throw"imageBufferOrg error";
    }

    cl_mem imageBufferTmp11 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        throw"imageBufferTmp11 error";
    }

    cl_mem imageBufferTmp21 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp11);
        throw"imageBufferTmp21 error";
    }

    cl_mem imageBufferTmp12 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp11);
        clReleaseMemObject(imageBufferTmp21);
        throw"imageBufferTmp12 error";
    }

    cl_mem imageBufferTmp22 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp11);
        clReleaseMemObject(imageBufferTmp21);
        clReleaseMemObject(imageBufferTmp12);
        throw"imageBufferTmp22 error";
    }

    cl_mem imageBufferDst = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp11);
        clReleaseMemObject(imageBufferTmp21);
        clReleaseMemObject(imageBufferTmp12);
        clReleaseMemObject(imageBufferTmp22);
        throw"imageBufferDst error";
    }

    //L1
    err = clSetKernelArg(kernelConv1To8L1, 0, sizeof(cl_mem), &imageBufferOrg);
    err |= clSetKernelArg(kernelConv1To8L1, 1, sizeof(cl_mem), &imageBufferTmp11);
    err |= clSetKernelArg(kernelConv1To8L1, 2, sizeof(cl_mem), &imageBufferTmp21);
    if (err != CL_SUCCESS)
        throw"L1 clSetKernelArg error";
    //L2
    err = clSetKernelArg(kernelConv8To8L2, 0, sizeof(cl_mem), &imageBufferTmp11);
    err |= clSetKernelArg(kernelConv8To8L2, 1, sizeof(cl_mem), &imageBufferTmp21);
    err |= clSetKernelArg(kernelConv8To8L2, 2, sizeof(cl_mem), &imageBufferTmp12);
    err |= clSetKernelArg(kernelConv8To8L2, 3, sizeof(cl_mem), &imageBufferTmp22);
    err |= clSetKernelArg(kernelConv8To8L2, 4, sizeof(cl_int), &L2);
    if (err != CL_SUCCESS)
        throw"L2 clSetKernelArg error";
    //L3
    err = clSetKernelArg(kernelConv8To8L3, 0, sizeof(cl_mem), &imageBufferTmp12);
    err |= clSetKernelArg(kernelConv8To8L3, 1, sizeof(cl_mem), &imageBufferTmp22);
    err |= clSetKernelArg(kernelConv8To8L3, 2, sizeof(cl_mem), &imageBufferTmp11);
    err |= clSetKernelArg(kernelConv8To8L3, 3, sizeof(cl_mem), &imageBufferTmp21);
    err |= clSetKernelArg(kernelConv8To8L3, 4, sizeof(cl_int), &L3);
    if (err != CL_SUCCESS)
        throw"L3 clSetKernelArg error";
    //L4
    err = clSetKernelArg(kernelConv8To8L4, 0, sizeof(cl_mem), &imageBufferTmp11);
    err |= clSetKernelArg(kernelConv8To8L4, 1, sizeof(cl_mem), &imageBufferTmp21);
    err |= clSetKernelArg(kernelConv8To8L4, 2, sizeof(cl_mem), &imageBufferTmp12);
    err |= clSetKernelArg(kernelConv8To8L4, 3, sizeof(cl_mem), &imageBufferTmp22);
    err |= clSetKernelArg(kernelConv8To8L4, 4, sizeof(cl_int), &L4);
    if (err != CL_SUCCESS)
        throw"L4 clSetKernelArg error";
    //L5
    err = clSetKernelArg(kernelConv8To8L5, 0, sizeof(cl_mem), &imageBufferTmp12);
    err |= clSetKernelArg(kernelConv8To8L5, 1, sizeof(cl_mem), &imageBufferTmp22);
    err |= clSetKernelArg(kernelConv8To8L5, 2, sizeof(cl_mem), &imageBufferTmp11);
    err |= clSetKernelArg(kernelConv8To8L5, 3, sizeof(cl_mem), &imageBufferTmp21);
    err |= clSetKernelArg(kernelConv8To8L5, 4, sizeof(cl_int), &L5);
    if (err != CL_SUCCESS)
        throw"L5 clSetKernelArg error";
    //L6
    err = clSetKernelArg(kernelConv8To8L6, 0, sizeof(cl_mem), &imageBufferTmp11);
    err |= clSetKernelArg(kernelConv8To8L6, 1, sizeof(cl_mem), &imageBufferTmp21);
    err |= clSetKernelArg(kernelConv8To8L6, 2, sizeof(cl_mem), &imageBufferTmp12);
    err |= clSetKernelArg(kernelConv8To8L6, 3, sizeof(cl_mem), &imageBufferTmp22);
    err |= clSetKernelArg(kernelConv8To8L6, 4, sizeof(cl_int), &L6);
    if (err != CL_SUCCESS)
        throw"L6 clSetKernelArg error";
    //L7
    err = clSetKernelArg(kernelConv8To8L7, 0, sizeof(cl_mem), &imageBufferTmp12);
    err |= clSetKernelArg(kernelConv8To8L7, 1, sizeof(cl_mem), &imageBufferTmp22);
    err |= clSetKernelArg(kernelConv8To8L7, 2, sizeof(cl_mem), &imageBufferTmp11);
    err |= clSetKernelArg(kernelConv8To8L7, 3, sizeof(cl_mem), &imageBufferTmp21);
    err |= clSetKernelArg(kernelConv8To8L7, 4, sizeof(cl_int), &L7);
    if (err != CL_SUCCESS)
        throw"L7 clSetKernelArg error";
    //L8
    err = clSetKernelArg(kernelConv8To8L8, 0, sizeof(cl_mem), &imageBufferTmp11);
    err |= clSetKernelArg(kernelConv8To8L8, 1, sizeof(cl_mem), &imageBufferTmp21);
    err |= clSetKernelArg(kernelConv8To8L8, 2, sizeof(cl_mem), &imageBufferTmp12);
    err |= clSetKernelArg(kernelConv8To8L8, 3, sizeof(cl_mem), &imageBufferTmp22);
    err |= clSetKernelArg(kernelConv8To8L8, 4, sizeof(cl_int), &L8);
    if (err != CL_SUCCESS)
        throw"L8 clSetKernelArg error";
    //L9
    err = clSetKernelArg(kernelConv8To8L9, 0, sizeof(cl_mem), &imageBufferTmp12);
    err |= clSetKernelArg(kernelConv8To8L9, 1, sizeof(cl_mem), &imageBufferTmp22);
    err |= clSetKernelArg(kernelConv8To8L9, 2, sizeof(cl_mem), &imageBufferTmp11);
    err |= clSetKernelArg(kernelConv8To8L9, 3, sizeof(cl_mem), &imageBufferTmp21);
    err |= clSetKernelArg(kernelConv8To8L9, 4, sizeof(cl_int), &L9);
    if (err != CL_SUCCESS)
        throw"L9 clSetKernelArg error";
    //L10
    err = clSetKernelArg(kernelConvTranspose8To1L10, 0, sizeof(cl_mem), &imageBufferTmp11);
    err |= clSetKernelArg(kernelConvTranspose8To1L10, 1, sizeof(cl_mem), &imageBufferTmp21);
    err |= clSetKernelArg(kernelConvTranspose8To1L10, 2, sizeof(cl_mem), &imageBufferDst);
    if (err != CL_SUCCESS)
        throw"L10 clSetKernelArg error";

    clEnqueueWriteImage(commandQueue, imageBufferOrg, CL_FALSE, orgin, orgRegion, orgImage.step, 0, orgImage.data, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv1To8L1, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L2, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L3, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L4, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L5, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L6, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L7, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L8, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L9, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConvTranspose8To1L10, 2, nullptr, dstSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadImage(commandQueue, imageBufferDst, CL_TRUE, orgin, dstRegion, dstImage.step, 0, dstImage.data, 0, nullptr, nullptr);

    //clean
    clReleaseMemObject(imageBufferOrg);
    clReleaseMemObject(imageBufferTmp11);
    clReleaseMemObject(imageBufferTmp21);
    clReleaseMemObject(imageBufferTmp12);
    clReleaseMemObject(imageBufferTmp22);
    clReleaseMemObject(imageBufferDst);

    clReleaseKernel(kernelConv1To8L1);
    clReleaseKernel(kernelConv8To8L2);
    clReleaseKernel(kernelConv8To8L3);
    clReleaseKernel(kernelConv8To8L4);
    clReleaseKernel(kernelConv8To8L5);
    clReleaseKernel(kernelConv8To8L6);
    clReleaseKernel(kernelConv8To8L7);
    clReleaseKernel(kernelConv8To8L8);
    clReleaseKernel(kernelConv8To8L9);
    clReleaseKernel(kernelConvTranspose8To1L10);

    //std::cout << dstImage << std::endl;
}

void Anime4KCPP::Anime4KGPUCNN::initOpenCL()
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

    cl_platform_id* tmpPlatform = new cl_platform_id[platforms];
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
    std::string ACNetKernelSourceString = readKernel("ACNetKernel.cl");
#endif // BUILT_IN_KERNEL
    const char* ACNetKernelSource = ACNetKernelSourceString.c_str();

    //create program
    program = clCreateProgramWithSource(context, 1, &ACNetKernelSource, nullptr, &err);
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

void Anime4KCPP::Anime4KGPUCNN::releaseOpenCL()
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

std::string Anime4KCPP::Anime4KGPUCNN::readKernel(const std::string& fileName)
{
    std::ifstream kernelFile(fileName);
    if (!kernelFile.is_open())
        throw"Read kernel error";
    std::ostringstream source;
    source << kernelFile.rdbuf();
    return std::string(source.str());
}

//init OpenCL arguments
bool Anime4KCPP::Anime4KGPUCNN::isInitialized = false;
cl_context Anime4KCPP::Anime4KGPUCNN::context = nullptr;
cl_command_queue Anime4KCPP::Anime4KGPUCNN::commandQueue = nullptr;
cl_program Anime4KCPP::Anime4KGPUCNN::program = nullptr;
cl_device_id Anime4KCPP::Anime4KGPUCNN::device = nullptr;
unsigned int Anime4KCPP::Anime4KGPUCNN::pID = 0U;
unsigned int Anime4KCPP::Anime4KGPUCNN::dID = 0U;

#ifdef BUILT_IN_KERNEL
const std::string Anime4KCPP::Anime4KGPUCNN::ACNetKernelSourceString = 
R"(#define RELU(x) fmax(x, 0.0f)

__constant sampler_t samplerN = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__constant float kernelsL1[9 * 8] = 
{
9.2272e-02,  1.2570e-01, -1.5282e-02,
-1.3409e-01,  7.2190e-01,  1.9612e-01,
 1.0453e-01,  8.4511e-02,  3.2972e-02,
 1.2242e-02,  1.9478e-01, -8.5346e-01,
 2.2631e-02,  8.6564e-01, -2.7245e-01,
-2.0227e-02,  7.4729e-02, -4.7416e-03,
-1.6457e-01,  6.6215e-01,  7.6708e-03,
-8.2886e-01,  4.3532e-01,  2.0739e-02,
-5.1994e-02, -3.7025e-02,  1.6032e-02,
 2.2851e-02, -9.6207e-01,  9.5596e-02,
-4.6188e-02,  1.6230e-01,  7.3529e-01,
-1.0738e-02,  3.2278e-02, -7.8451e-03,
 2.0564e-01, -4.2684e-02,  1.4962e-01,
 3.0146e-01,  5.5630e-01,  2.5966e-01,
 1.6087e-01, -1.7521e-01, -3.3185e-02,
 1.1727e-01, -1.9920e-01, -1.7486e-01,
 3.1379e-01, -3.8220e-01, -2.1523e-01,
 2.0200e-01,  3.0992e-01,  5.1515e-02,
-2.7132e-02,  7.1935e-02, -7.5391e-03,
-1.6304e-02,  1.0667e+00, -7.1720e-02,
 1.6368e-04, -3.5761e-01, -6.0912e-01,
-5.3884e-03, -9.2047e-01,  3.0588e-02,
 4.6173e-02,  9.5072e-01, -7.3402e-02,
-2.1013e-02, -9.4680e-03,  2.9357e-02
};

__constant float biasL1[8] = 
{
-7.4477e-01, -2.6887e-02,  2.3039e-02, -3.1657e-02, -3.5426e-04, 2.6175e-02,  2.0173e-02, -2.7086e-02
};)"
R"(
__constant float kernelsL[8][9 * 8 * 8] = 
{
{
 1.8221e-01,  6.3212e-02, -1.2652e-01,
 1.2931e-01,  4.8768e-02, -7.4272e-02,
-1.5941e-01,  6.5098e-03, -1.6454e-02,
-1.5938e-01, -1.3463e-01, -9.1331e-02,
 6.8563e-02, -1.8822e-01, -9.0560e-02,
-1.3412e-01, -1.3309e-01,  1.0675e-01,
 3.8597e-02, -1.0191e-01, -2.6567e-01,
 4.6832e-02, -1.9165e-01,  4.3209e-02,
-1.0936e-01, -5.2018e-02,  4.1730e-02,
 3.8841e-01,  3.1029e-02,  4.5984e-02,
-7.5272e-02,  1.6197e-01,  1.6960e-01,
 2.2508e-01,  5.5097e-01,  2.3939e-02,
 2.9924e-03,  2.7444e-01, -1.0985e-01,
 4.9409e-02, -1.8732e-02, -1.2845e-01,
 7.6762e-02,  1.2238e-01, -2.6657e-01,
-1.7187e-01, -1.9681e-01,  3.5439e-02,
-1.3525e-01,  9.6793e-02, -7.3562e-02,
-9.0658e-02, -5.3841e-02, -6.1696e-02,
 3.8147e-01,  3.0448e-01, -2.1703e-01,
 1.3070e-01, -3.1028e-01,  2.8814e-02,
-4.1590e-02,  4.6872e-03,  3.9949e-02,
 9.5090e-02, -7.3559e-02,  3.4317e-02,
-2.7071e-02,  2.2387e-02,  1.4454e-01,
-2.1033e-01,  1.5248e-01, -1.8784e-01,
 5.9198e-02,  1.1955e-01,  3.1283e-02,
 1.4652e-01,  6.1238e-01,  8.8925e-02,
 1.0822e-02,  1.5275e-01,  9.8656e-03,
-4.1697e-02, -8.0667e-02, -4.5871e-02,
-9.1869e-02,  1.6633e-01, -4.5683e-02,
-1.6842e-01, -1.3736e-02,  3.6097e-04,
 2.7415e-02, -2.6970e-02,  3.2438e-02,
 6.8630e-02,  2.4091e-01, -1.7492e-01,
 8.6020e-02,  2.2617e-01, -2.8069e-02,
 6.8882e-03, -1.0537e-02,  1.1285e-02,
 9.8033e-02, -1.3391e-02, -2.5313e-02,
-1.2279e-02, -1.7841e-01,  1.2927e-01,
-8.4111e-03, -9.3943e-02, -7.8833e-02,
 1.1164e-01,  2.0516e-01, -1.7780e-02,
 7.2845e-02, -7.7627e-02,  1.3147e-02,
-1.2452e-02,  1.4986e-01,  8.2951e-02,
-1.0494e-01, -2.9412e-01,  2.0203e-02,
 3.9057e-02, -1.6880e-01, -2.3646e-02,
-1.7856e-01, -1.5576e-01, -3.1830e-03,
-8.1905e-02,  4.5968e-01,  2.7145e-02,
 2.9753e-03,  1.8426e-02,  4.3275e-02,
 2.1363e-02, -6.5214e-02,  7.4165e-02,
 5.2400e-02,  2.2971e-01,  1.6632e-03,
 1.8980e-01, -1.6758e-01,  1.6768e-01,
-1.0252e-01, -3.8394e-02,  7.6731e-02,
-3.6679e-03,  1.2777e-01, -5.9425e-02,
-4.6721e-02,  3.9304e-02,  9.1553e-02,
 3.8850e-02,  3.8389e-02,  6.6960e-02,
 3.6690e-02,  4.8267e-03, -7.7022e-02,
 6.0578e-02, -1.5036e-01,  8.1762e-02,
-3.1805e-02,  1.2762e-02, -1.0617e-01,
-2.3478e-02, -3.7310e-02, -1.5171e-01,
 1.9508e-02,  2.1338e-01, -2.1243e-01,
-2.6884e-02, -1.5979e-02, -1.3145e-01,
 1.7949e-01,  2.2106e-01, -8.9061e-03,
 1.7715e-01, -3.2444e-01, -9.2675e-02,
-1.0385e-01, -1.6574e-01, -7.5180e-02,
-3.9463e-02,  2.4838e-02,  1.9745e-01,
 1.5138e-01,  3.0927e-02,  1.1893e-01,
 1.4266e-02, -6.0976e-02, -1.3239e-01,
 6.7690e-02,  2.2158e-01,  8.4932e-02,
-1.3038e-01, -8.0158e-04,  1.1140e-02,
-3.0191e-02,  2.5621e-01,  3.5163e-02,
 1.3216e-01,  2.5819e-01, -6.0904e-02,
-2.3911e-02,  4.9311e-02, -7.8031e-02,
-3.6335e-02,  1.5025e-02, -9.7102e-03,
-1.5486e-01, -1.5130e-01, -7.2697e-02,
-2.4446e-02, -4.3771e-01, -2.9798e-02,
-7.0794e-02,  1.3646e-01, -9.2750e-04,
 9.2402e-03,  1.8070e-01,  2.5704e-02,
-8.3302e-03,  5.7733e-03,  3.5606e-02,
-2.3078e-02, -1.5188e-01, -9.6620e-03,
-6.4997e-02,  1.9456e-01, -1.2812e-01,
-3.0430e-02, -1.2938e-01,  1.0042e-01,
-2.5663e-02,  1.8931e-02,  4.9123e-02,
 3.0819e-02,  2.4494e-01,  1.2368e-01,
 9.0711e-03,  1.3700e-01,  1.4421e-01,
-1.1253e-01,  3.3308e-02,  9.2980e-02,
 1.2614e-01, -9.8001e-02, -9.5395e-02,
 1.4035e-01, -2.9892e-01,  8.5534e-02,
-3.1068e-02, -1.1503e-01, -6.5670e-02,
 6.2791e-02,  2.4409e-01,  4.8189e-02,
 4.1478e-02,  3.6473e-02,  1.3605e-01,
 2.9394e-03,  1.5456e-02,  1.4242e-01,
 2.8307e-02,  7.8412e-02, -5.6567e-02,
-6.1940e-02,  5.0871e-02, -1.0940e-02,
 6.7335e-02, -6.4785e-02, -9.3228e-02,
 2.3052e-02,  1.4078e-01,  1.5646e-01,
-9.7920e-03, -1.6338e-03,  4.7873e-03,
-7.6630e-03, -1.0126e-01,  1.0920e-01,
-1.7024e-01, -2.9386e-01, -1.4854e-01,
 9.6418e-02,  2.0663e-01, -1.0462e-01,
-2.4006e-02,  1.6165e-01, -8.5956e-03,
-3.0708e-02, -1.1191e-01,  2.1777e-01,
-7.7602e-02, -1.0961e-01, -6.8072e-03,
 6.7153e-02,  3.0147e-01,  5.1709e-02,
 2.2551e-01,  1.6840e-01,  2.4935e-01,
 3.7141e-01,  7.5778e-02,  3.0336e-02,
-1.1620e-01, -2.5449e-02, -2.9635e-01,
 4.2912e-02, -5.2279e-03,  7.9257e-01,
-9.3572e-02, -2.8331e-01,  2.9113e-01,
 1.0692e-02, -2.8691e-02, -6.2718e-02,
-9.0376e-03, -2.2143e-02, -5.9330e-02,
-2.1271e-02,  4.8916e-02, -1.1890e-01,
-2.4064e-03,  2.1288e-01,  1.2840e-01,
-1.6568e-01, -2.0505e-01,  2.0475e-01,
-1.1459e-01,  5.6298e-02, -9.7340e-02,
 6.6450e-02, -3.4334e-01, -4.2163e-01,
 6.7871e-03, -1.7081e-01, -2.0482e-01,
 9.3896e-02, -5.1579e-02, -1.3004e-01,
-2.2968e-02,  1.1605e-01,  4.7362e-02,
-7.0415e-02, -3.3688e-01,  6.6914e-02,
 1.0308e-01,  2.9382e-03, -8.7922e-02,
-3.5942e-02,  3.4054e-02,  5.3037e-02,
-4.2643e-02, -3.8332e-03,  2.5985e-01,
-8.1397e-02,  4.1825e-01,  7.1042e-02,
 1.7519e-01,  1.6750e-01,  9.3732e-02,
 2.5423e-01,  2.6490e-01,  1.1851e-01,
 1.0262e-01,  1.0488e-01,  8.3631e-02,
-2.6059e-02,  1.3575e-01,  1.3836e-02,
-1.4978e-01,  2.6084e-01,  1.0071e-01,
-1.9244e-01,  8.1539e-03,  5.7492e-03,
-6.6055e-03,  6.8705e-02, -1.6593e-01,
 5.4556e-02,  7.9820e-02, -3.7893e-01,
 2.8685e-02,  2.7593e-01, -3.5936e-02,
 1.0176e-01,  6.5954e-02, -4.1193e-02,
 1.8105e-01,  1.9445e-01, -1.7032e-02,
 1.9459e-03, -3.0590e-01, -8.8353e-03,
 7.0706e-02,  1.7193e-01,  1.6534e-01,
 5.2768e-02,  2.2471e-01,  2.7788e-01,
 2.4786e-02,  2.7045e-01,  1.4781e-01,
-4.5112e-02,  2.3585e-02,  3.3668e-02,
 3.5756e-03, -1.6293e-01,  5.4950e-02,
-5.3354e-02,  8.6662e-02,  7.6881e-02,
-3.3907e-01, -2.5661e-01,  4.4301e-02,
 8.8241e-02,  3.6995e-01,  6.0618e-02,
-1.0445e-02,  6.4614e-02, -6.8944e-02,
-3.0018e-02, -5.4119e-03, -4.6493e-02,
 1.2895e-02,  1.7162e-01,  6.3015e-02,
-2.0528e-02, -1.8090e-01,  1.0398e-01,
 9.4944e-02, -1.8184e-01, -2.2940e-03,
-6.6665e-02,  1.5764e-01,  1.4113e-01,
 4.6279e-02, -4.7605e-02, -3.4291e-02,
-5.9775e-02, -1.1464e-01, -9.7913e-03,
 1.6768e-01,  3.4831e-01,  5.2680e-02,
-1.7231e-01, -1.2593e-01,  9.7332e-03,
-2.0364e-02, -9.2346e-02,  1.7324e-01,
 9.7357e-02,  1.1141e-01, -2.9019e-02,
-1.3685e-01,  1.3477e-01,  1.0202e-01,
-4.5975e-02, -9.7563e-02, -1.0285e-01,
-3.3112e-03,  2.5169e-01, -9.9592e-02,
 7.0381e-02, -2.4390e-01, -2.5913e-02,
-1.9259e-02, -1.6505e-01, -4.4569e-02,
-1.7213e-01,  2.7285e-01,  1.2657e-01,
-1.0318e-01, -1.4538e-01,  1.9215e-01,
 2.7051e-01,  2.0523e-01, -1.0565e-01,
 7.4304e-02,  1.9057e-01, -3.0707e-02,
 2.2480e-02,  3.0812e-02,  3.7586e-02,
 1.4474e-01, -2.5314e-02,  3.8714e-02,
-1.6799e-01,  6.5341e-02,  6.5259e-02,
-7.9980e-03,  4.1393e-02, -3.9326e-02,
 2.3515e-02,  3.2509e-02,  1.2556e-01,
 6.4182e-02,  6.6594e-01, -4.6309e-02,
 2.6091e-01,  2.7119e-02,  3.3683e-02,
-4.9421e-02,  4.8037e-02, -4.7768e-02,
 2.4300e-02,  9.0839e-02,  1.4008e-02,
-6.3909e-02,  7.2057e-02, -1.0137e-01,
-4.5610e-02,  1.2559e-02,  2.6425e-02,
-5.5825e-02,  1.8745e-01, -4.0840e-02,
-1.9845e-02,  6.5562e-01,  1.0898e-02,
-2.4671e-02,  9.5408e-02, -2.2946e-02,
-1.7670e-02,  6.9855e-02, -7.2161e-03,
 1.1961e-02, -1.1696e-01, -1.3273e-01,
-1.7429e-02,  1.6502e-01, -4.1149e-03,
-1.7239e-02,  2.3999e-01,  4.9996e-02,
 4.0423e-02, -7.6489e-02,  8.2335e-02,
-8.7369e-03, -1.1526e-01, -7.8585e-02,
 3.1607e-02,  5.2380e-01, -3.5025e-03,
-2.8281e-02, -1.8841e-01, -1.2578e-01,
 8.0348e-03,  1.1105e-02, -5.6380e-02,
 4.3388e-02, -1.6254e-01,  4.1271e-02,
-4.2446e-02,  3.8191e-02,  4.0225e-02,
 6.8892e-02, -3.7291e-02, -5.2809e-02,
-3.4652e-02,  7.9726e-01, -5.5930e-02,
-2.8873e-02,  1.0525e-01, -1.7124e-02,
-9.0510e-03, -1.2725e-01, -8.2102e-02,
-1.6687e-01, -1.5154e-01, -7.5661e-02,
-5.2025e-02, -1.6993e-01,  1.0086e-01
}
,)"
R"(
{
-4.9077e-02, -6.4584e-02, -4.2960e-02,
 4.0659e-02, -7.0338e-02, -8.3809e-03,
-1.5734e-03, -5.2116e-02, -7.5534e-02,
 1.4785e-02, -4.0422e-02, -5.2390e-02,
 5.4408e-02,  5.2189e-02, -2.5028e-02,
 5.5778e-02,  3.9226e-02,  3.5280e-02,
 1.1164e-02, -5.0444e-02,  3.7947e-02,
 2.6973e-01,  3.2699e-01,  2.0832e-02,
 3.8362e-02, -8.5338e-02,  3.8716e-02,
-8.6233e-03, -4.4944e-02, -5.2736e-02,
-1.8374e-01,  1.2336e-01, -2.7175e-02,
 6.0527e-02,  8.7389e-03, -2.0910e-03,
-1.0196e-03,  1.6047e-01, -2.1893e-02,
 1.9370e-01,  4.4910e-01,  8.6093e-02,
 1.2876e-01,  1.0911e-01, -6.2863e-02,
-1.2637e-02, -1.0084e-01, -1.1627e-01,
 9.3179e-02,  1.9028e-01, -2.0397e-01,
-2.6408e-02,  1.3606e-01, -8.6076e-03,
-1.7791e-02, -2.4887e-02, -1.3328e-02,
-4.9230e-02,  2.0702e-01,  1.6966e-01,
-6.4848e-02, -8.7684e-02,  2.1326e-02,
-2.1238e-02,  3.1733e-01, -7.2057e-02,
-8.0414e-02,  4.5757e-01, -8.0889e-02,
 1.6809e-02, -4.1740e-02,  7.0189e-02,
-1.0842e-01,  1.7090e-01,  2.9161e-01,
 2.0309e-01, -2.5974e-01,  1.7305e-01,
 2.6443e-01,  1.3967e-01, -1.4253e-01,
-1.3431e-01, -4.3437e-01,  1.5140e-01,
 3.6370e-02,  2.1109e-01, -1.2709e-01,
 2.9852e-02,  1.7924e-02, -1.7528e-01,
-7.1489e-02, -7.6305e-01, -9.0099e-02,
-1.7235e-01, -2.2947e-02,  5.9149e-02,
 3.1945e-02,  1.3711e-01,  4.9728e-02,
-1.6711e-02, -5.2419e-02,  9.6215e-02,
 7.8171e-02,  1.9741e-01,  2.2115e-01,
-4.5362e-02,  3.2627e-02,  3.4850e-02,
 1.8324e-02, -1.5543e-01, -8.1372e-02,
 2.4103e-01, -1.6004e-01,  1.4880e-02,
-1.2197e-01, -6.4775e-02, -6.2188e-02,
-1.0815e-01, -2.4144e-01,  9.7464e-02,
 8.0140e-02,  2.6031e-01, -5.1319e-02,
-4.7376e-02, -2.3621e-02, -3.6893e-02,
-2.2354e-01,  2.9908e-01,  1.9756e-01,
 2.3740e-01, -6.8371e-02, -3.6678e-01,
 1.5958e-02, -1.7128e-01, -5.8263e-02,
 1.2398e-01,  1.7735e-01,  1.5660e-01,
-7.4479e-02,  3.2445e-02, -1.1865e-01,
-2.5939e-02, -1.1237e-02,  2.3873e-02,
-5.6482e-02,  8.4736e-02, -2.0225e-02,
 6.2314e-02, -1.0252e-01,  2.8450e-02,
-2.2691e-04,  3.1820e-02,  2.8976e-02,
-7.8789e-03, -4.1019e-02,  5.5655e-02,
-7.1532e-02,  1.8791e-01, -2.2134e-02,
-2.4663e-03,  7.7835e-02, -1.1930e-01,
 5.0934e-03,  2.5914e-02, -6.9006e-02,
 8.7591e-02,  1.0248e-01, -1.0506e-01,
-6.4577e-02,  5.3780e-03,  2.2648e-02,
-1.6925e-02,  1.3158e-02, -5.9132e-02,
-1.6256e-02,  1.7739e-01,  2.9871e-02,
-2.8738e-02,  5.6438e-02, -2.5228e-02,
-5.6795e-03, -2.0668e-02, -7.6091e-03,
 6.3457e-02, -1.9728e-01, -6.3770e-02,
 1.1712e-01,  1.2425e-01,  3.8227e-03,
-3.9693e-02,  1.3628e-01,  1.1901e-01,
 9.3149e-02,  3.9480e-01,  1.5146e-01,
 3.0612e-02,  2.4305e-01,  6.3542e-02,
-1.5277e-02,  9.5541e-02, -9.4211e-03,
 1.4093e-01,  1.5841e-01,  1.0791e-01,
-3.3514e-02, -3.6321e-01, -1.0025e-01,
 1.2038e-01, -1.6285e-01,  6.0452e-02,
-2.4989e-01,  3.1525e-01,  2.3659e-02,
-8.4040e-02, -2.8048e-02,  3.9397e-03,
 2.3479e-02, -7.0752e-02, -2.4111e-01,
-1.5076e-01, -4.2834e-02, -4.6177e-02,
-3.2031e-01,  1.2658e-01,  3.4348e-02,
-2.0306e-01, -1.8463e-01, -1.8145e-01,
 2.9000e-01,  5.8861e-01, -1.1362e-01,
-1.3151e-01,  8.3227e-02, -1.7782e-01,
-7.8488e-02, -1.6630e-01, -4.6899e-02,
-8.7425e-02,  2.9553e-01,  5.9401e-02,
-9.6194e-02, -4.4305e-02,  1.9495e-03,
 1.0846e-01, -1.7812e-01, -8.3809e-02,
 2.5939e-01,  1.6111e-01,  6.9297e-02,
 7.9238e-02,  4.2983e-01, -1.4574e-02,
 6.3641e-02,  1.1814e-01,  3.0058e-02,
-2.7747e-01,  3.4229e-03, -4.6782e-03,
-1.8443e-01, -6.2887e-02,  1.5402e-02,
-9.9913e-02, -5.2354e-02, -3.5345e-02,
 1.9193e-01,  1.3668e-01,  7.0823e-02,
-2.4238e-01, -9.3310e-02, -3.8631e-02,
 7.6262e-02, -1.6128e-02, -1.8312e-02,
-4.3358e-02, -2.3563e-01, -3.3862e-02,
-5.8146e-02, -6.2719e-02,  1.7218e-01,
-4.1666e-01, -7.5645e-02,  5.5110e-03,
-4.0900e-01,  2.0528e-01,  3.5457e-03,
-2.2532e-02, -5.5725e-02,  1.5779e-02,
-6.5336e-03, -2.8478e-02, -3.4736e-02,
 8.3000e-02, -1.8120e-02, -1.6989e-03,
-1.2303e-01, -2.5433e-01, -1.2678e-01,
-1.7882e-02, -2.5658e-02, -3.0640e-02,
-1.3181e-02,  7.2009e-02, -7.4684e-02,
 6.3070e-40, -9.9058e-02, -8.5923e-02,
 2.0390e-02,  4.4250e-02, -7.9733e-02,
-5.1912e-02,  1.5968e-01, -3.8990e-02,
-4.6844e-03, -8.1944e-02,  4.2654e-02,
 1.6592e-02,  5.9797e-02, -1.2204e-01,
-2.5359e-01,  1.3531e-01, -1.1049e-01,
-1.5170e-01,  1.3571e-01,  2.8438e-02,
 2.4160e-02,  3.5329e-02,  2.4462e-02,
-1.7603e-01, -1.2269e-01,  1.4909e-01,
-5.1905e-01, -2.1981e-01, -1.0348e-01,
-1.0335e-02,  6.5809e-02, -5.3626e-02,
-6.1654e-02,  3.0865e-01,  5.9228e-02,
-3.9104e-01, -2.2770e-01,  7.4561e-02,
-4.0219e-02,  7.6460e-02,  1.3817e-02,
-7.2185e-02,  3.6790e-01, -8.1726e-02,
-1.3745e-01,  4.6380e-02,  5.0863e-02,
 2.2871e-03, -7.2967e-02,  1.6622e-02,
 2.1177e-01,  4.6899e-01, -4.7199e-03,
-3.3511e-02, -1.3652e-01,  3.2397e-02,
-5.5046e-02,  4.4355e-02, -5.7487e-03,
-3.3443e-02, -1.5459e-01,  4.4264e-02,
-2.5154e-02,  4.7542e-02, -3.6066e-02,
 7.9373e-03,  1.0533e-01,  5.3082e-02,
 1.1827e-01, -1.5881e-01,  2.0321e-01,
-2.5883e-02, -1.1121e-01,  3.8086e-02,
 3.9343e-02, -1.4946e-01,  5.4741e-02,
 3.0156e-01, -2.1722e-01, -2.4833e-02,
 7.4271e-03,  1.9456e-02,  1.5183e-02,
-1.6684e-01,  1.7277e-01, -1.6829e-03,
-3.1971e-01,  3.2596e-01, -8.8923e-02,
 4.0296e-02, -1.4262e-01, -5.3337e-02,
 2.6221e-02,  1.2129e-03, -8.5688e-03,
 8.4837e-02, -3.7299e-02,  7.3421e-02,
 3.0066e-02,  3.2365e-02, -2.3040e-02,
-5.1678e-02,  2.5830e-01, -2.0536e-02,
 1.5001e-02,  1.7711e-01,  4.5448e-02,
-1.0756e-01, -1.8844e-01, -5.1357e-02,
-2.4867e-02,  1.1422e-01, -2.0449e-02,
-1.0003e-01,  3.8700e-01, -6.9790e-03,
 6.3972e-02,  3.1805e-01,  8.3606e-02,
 1.1860e-01, -1.6405e-01,  9.2468e-02,
 2.8021e-01,  2.5270e-01, -1.6418e-01,
 1.2705e-01, -5.1544e-02,  6.2713e-02,
-1.2765e-01, -1.1741e-01, -2.1646e-01,
 1.3192e-01, -2.6478e-01, -9.2520e-02,
 3.2786e-03, -2.0097e-01, -1.2329e-01,
-6.9569e-03, -3.4701e-02,  2.7486e-01,
-8.8802e-02, -3.5888e-01,  6.1823e-02,
 1.5706e-02,  2.0619e-02,  1.1470e-01,
 3.2890e-02, -3.1990e-01,  2.0465e-01,
 1.5796e-01, -4.0526e-01, -1.1442e-01,
 7.9465e-02,  1.3766e-01,  1.3628e-01,
-2.0921e-01, -1.2245e-01,  2.0633e-01,
-2.3013e-01, -7.2193e-01, -4.2358e-02,
-2.0441e-01, -4.8066e-01, -1.3939e-01,
 1.9203e-01,  1.5506e-01, -2.5617e-01,
 1.7178e-01,  4.5557e-01,  4.4588e-02,
 2.3239e-01,  9.9494e-02,  1.8406e-01,
-1.2044e-01,  2.0750e-01,  1.3087e-01,
 6.2688e-03,  1.7598e-01, -6.6133e-02,
 3.9866e-02,  6.1298e-02,  7.9176e-02,
-6.2564e-03,  1.8297e-02,  2.4288e-02,
-4.1978e-02,  7.5876e-02,  1.3336e-01,
-1.8504e-02, -1.5127e-02, -6.0184e-02,
 2.9427e-01,  3.0279e-01, -2.6182e-02,
 1.3650e-01, -1.9013e-01, -3.0584e-03,
 1.4814e-02,  3.6700e-02,  4.0690e-02,
-3.8285e-01, -3.2379e-01,  3.9664e-02,
-3.6651e-01,  2.3366e-02, -5.8116e-02,
-2.4894e-01, -1.8484e-01, -3.6219e-01,
 2.2578e-02, -9.4037e-02,  9.7560e-02,
 1.6641e-01,  1.1111e-01, -1.3857e-01,
-1.0062e-01,  3.0369e-02, -4.6619e-02,
-3.2468e-02, -1.7113e-01,  7.2862e-02,
 8.6801e-02, -1.1853e-01, -8.4216e-02,
-9.0982e-02,  6.5032e-02, -3.5422e-02,
-3.8019e-02, -1.5081e-01,  2.9963e-02,
 1.7886e-01, -4.1219e-02,  2.7479e-02,
 5.4055e-02, -1.5391e-01, -5.0515e-02,
 2.4880e-01,  2.6112e-01, -9.6236e-04,
 3.7670e-01,  6.0306e-02,  7.8365e-02,
 2.1743e-01,  2.4590e-01,  4.7737e-02,
 1.4158e-01, -4.4880e-02, -8.1449e-02,
 1.0121e-01,  2.4400e-01, -2.1322e-01,
-5.2574e-02,  3.5462e-02, -6.5620e-02,
-6.1942e-02, -1.3207e-02,  5.7262e-02,
-7.2846e-02,  2.4168e-01, -3.2189e-02,
-2.0800e-02, -9.2925e-03,  1.6286e-02,
-1.2944e-03, -2.6106e-01, -5.5997e-04,
-2.2060e-02, -8.5441e-02, -1.2905e-01,
 6.0723e-03, -6.8619e-02,  1.1620e-02
}
,)"
R"(
{
 2.2834e-02,  3.7433e-03, -2.1343e-02,
-1.9906e-01,  4.0537e-01, -2.4626e-01,
 2.6000e-02,  1.5449e-01,  5.2577e-02,
-5.5491e-02, -4.8605e-03, -5.2021e-02,
 1.1609e-02,  2.0908e-01,  2.2674e-01,
 1.0085e-01,  2.9484e-01,  2.6733e-01,
-8.9050e-02,  1.4612e-01, -8.8425e-02,
 1.9653e-01,  1.6370e-01, -2.2981e-01,
 1.6135e-02, -1.5149e-01,  5.0530e-02,
 1.1252e-01,  9.0199e-02,  6.4165e-02,
 2.8530e-01, -3.8369e-01, -5.7849e-01,
-1.4509e-02, -1.3489e-01, -2.1992e-01,
 4.8308e-02,  5.1834e-02, -5.8170e-02,
-2.5443e-02,  3.4541e-01, -3.7435e-01,
 2.2682e-03,  4.1949e-02,  9.4179e-03,
-1.5277e-02, -8.9774e-02,  1.7957e-01,
-3.0015e-02,  3.0565e-01, -2.9587e-01,
 7.8760e-03, -2.4449e-02, -2.8121e-03,
 9.1430e-03, -6.2735e-02,  5.9142e-03,
 7.4124e-03,  1.3001e-01,  1.7749e-01,
 2.0003e-01, -1.0617e-01,  4.9584e-03,
 4.1268e-02,  8.9135e-02, -4.9114e-02,
-7.3466e-02, -3.7044e-02, -3.5040e-01,
-2.9975e-02, -8.9946e-03, -3.7875e-02,
 2.9884e-02, -1.4808e-01,  4.7787e-02,
-1.2699e-01, -9.1423e-02, -5.3977e-02,
-1.7785e-02,  2.9891e-02, -5.6855e-03,
-7.0813e-02, -1.7200e-01, -3.8504e-03,
 4.9354e-02,  1.1601e-01,  2.5630e-01,
-1.7295e-02,  2.4338e-01, -6.5835e-02,
-3.5792e-02, -1.9034e-02, -9.5830e-02,
-6.8075e-03,  1.5473e-01, -4.5020e-02,
 2.9092e-02, -5.4506e-02,  7.7732e-02,
-2.0961e-03,  1.8119e-01, -2.4656e-02,
 6.8523e-02,  1.0634e-01, -2.6782e-01,
-3.4002e-02, -8.8813e-02,  4.2204e-02,
 4.8954e-02, -5.7349e-02, -2.0212e-02,
-9.5662e-02, -1.0526e-01, -5.6271e-02,
-5.6216e-02, -1.1130e-02, -1.4864e-03,
-3.0244e-02, -7.7690e-03,  9.6958e-02,
 2.9307e-02,  1.4343e-01,  3.1934e-01,
 6.4454e-02,  1.9724e-02, -2.3079e-02,
-1.2803e-01,  6.5423e-03, -1.5884e-01,
 7.6447e-02,  5.5207e-01,  6.3693e-02,
-1.7438e-01,  1.7310e-01,  4.6675e-02,
 4.6232e-02,  7.3360e-02, -1.5323e-01,
 1.0902e-02, -1.4923e-02,  9.5443e-02,
 1.8087e-02, -5.9946e-02, -1.1084e-01,
-5.9561e-02, -7.3729e-04,  7.8748e-02,
 3.5657e-01,  1.5770e-02,  1.7375e-01,
 9.3210e-02,  1.3231e-02,  4.4977e-02,
 1.0125e-01,  1.3872e-01, -4.3277e-02,
 2.0040e-01, -2.2672e-01, -7.1723e-02,
-1.6448e-01, -8.0318e-02, -1.1439e-01,
 1.0256e-01,  9.3240e-02,  8.2556e-02,
-2.3791e-01, -2.7823e-01,  2.3544e-01,
-2.2851e-02, -1.4107e-01,  1.2279e-01,
-8.7790e-03,  3.7408e-02, -3.6392e-02,
-1.6043e-01,  2.5635e-01,  1.8560e-01,
 5.4550e-03,  7.2405e-02, -4.7996e-02,
-4.3225e-02,  3.1403e-02,  2.2453e-02,
 5.3320e-02, -1.3215e-01,  3.3622e-02,
 4.3753e-02,  1.7009e-02,  1.7289e-02,
 6.8642e-02,  2.3051e-02,  4.8341e-02,
-6.2853e-02, -3.3710e-01,  5.3632e-02,
 2.7896e-02, -2.1915e-01,  9.3345e-02,
 2.4826e-02, -6.3672e-02,  1.2631e-02,
-1.3082e-01,  5.9828e-02, -6.6708e-02,
-6.0952e-02, -4.5006e-01, -9.0729e-03,
 2.1281e-02,  3.8911e-03,  1.3900e-01,
 2.7530e-02, -8.2553e-04,  9.0368e-02,
 6.8432e-02,  1.5377e-01, -2.8579e-02,
-5.9751e-02,  3.2439e-02, -5.1037e-02,
-1.4065e-02, -2.4179e-01, -1.5983e-01,
-2.3125e-01, -1.3940e-01,  2.9562e-02,
-1.7655e-02,  7.5401e-02, -1.1459e-02,
-1.5585e-01,  1.8354e-01,  1.0691e-01,
 4.0111e-01,  5.0514e-01,  3.6141e-03,
-1.2172e-01, -1.5731e-01, -1.3686e-01,
 1.7980e-01,  1.6324e-01, -1.3941e-01,
 1.4399e-01,  5.4963e-02, -8.5619e-03,
-1.5451e-01, -3.7165e-01, -1.5501e-01,
 1.8050e-02, -5.6398e-02,  1.4005e-02,
 4.6435e-02,  2.7728e-01,  2.4183e-01,
 2.8596e-02, -2.2416e-02, -8.9035e-03,
 2.8797e-02,  4.2167e-01,  1.8590e-01,
-3.8574e-02,  8.2193e-02, -3.1196e-02,
 3.5932e-02,  1.1091e-01,  6.3168e-02,
-1.4408e-01,  1.7124e-01, -1.0139e-01,
 1.4438e-01, -5.4504e-02, -2.9144e-02,
 5.4586e-02,  1.0701e-02,  5.7591e-02,
 2.5158e-01,  2.5278e-01,  8.8665e-02,
 3.8294e-01,  1.3365e-01,  1.0628e-01,
-1.0737e-01, -3.4831e-01,  1.0840e-01,
 1.2373e-01, -2.0438e-01, -4.0129e-01,
-3.0124e-01, -2.8592e-02,  2.4008e-01,
 3.6233e-03, -3.3343e-02, -4.7253e-03,
-5.8460e-02,  3.1610e-01,  1.0649e-02,
-2.3921e-02, -3.0216e-02, -3.7867e-03,
 9.6576e-02, -4.3473e-02,  1.9862e-02,
-1.5193e-01,  1.2558e-01,  1.4502e-01,
 3.2048e-03,  4.2826e-01, -1.5356e-01,
-1.5436e-02, -2.6327e-01, -4.0548e-02,
 8.7563e-02,  2.6970e-01,  2.0885e-01,
-4.1120e-02, -1.2623e-01, -7.8811e-02,
-2.1884e-01, -1.2907e-01,  2.0752e-01,
-1.6068e-01, -6.5775e-02,  4.4888e-03,
-3.7008e-02, -9.9657e-02,  1.9444e-01,
 3.7785e-02, -2.1598e-02,  2.4873e-03,
-2.6304e-04, -1.0797e-01, -1.0803e-01,
 7.5807e-02,  2.5729e-02,  9.5539e-02,
 2.9127e-02,  4.3045e-02, -5.2777e-02,
 2.2009e-02,  3.8096e-01, -3.5948e-01,
-8.7752e-02, -1.1576e-01, -8.1762e-02,
 1.3790e-01,  7.3423e-02,  2.0858e-02,
 1.3289e-01,  1.6113e-01, -1.7199e-01,
-5.8690e-02,  1.1465e-01, -1.4316e-01,
-8.3055e-02, -5.9369e-02, -2.3700e-01,
 3.1214e-02, -1.4499e-01,  3.3022e-02,
 1.2839e-01, -1.4206e-02,  2.5224e-01,
 1.6184e-02,  1.2295e-02, -2.8967e-02,
-1.5906e-01,  3.9451e-01, -6.6219e-02,
-1.4362e-01, -1.6011e-01,  5.3637e-02,
 1.1845e-02,  2.4933e-02,  2.1563e-02,
-8.0148e-02,  1.1610e-01, -3.6424e-02,
 9.5419e-02, -1.5850e-01,  2.7918e-02,
-4.3999e-02,  1.7063e-01, -7.8161e-02,
 1.7313e-01,  4.0542e-01,  9.2553e-02,
 1.1227e-01,  1.2596e-01, -7.8019e-02,
-4.5692e-02,  6.5875e-02, -1.2505e-02,
 3.8156e-02, -6.2078e-02,  2.7383e-04,
-3.9357e-02,  2.8625e-02,  3.3574e-02,
 3.2864e-02, -1.0425e-02, -4.7426e-03,
-3.3601e-02,  7.9290e-01,  1.2693e-02,
 4.9807e-02,  1.6826e-02, -6.2695e-03,
-7.6047e-02, -1.7007e-01, -7.1868e-02,
-1.1495e-02,  3.5118e-01, -1.9567e-01,
 1.1620e-01,  1.0302e-01, -5.9116e-02,
 9.8342e-03,  8.4895e-02,  3.4076e-02,
 1.9252e-02, -3.0012e-01,  1.0614e-01,
 2.0168e-01, -9.7745e-02, -1.4725e-02,
 2.2589e-02, -2.2530e-02,  8.0429e-03,
-6.2460e-02,  2.4878e-02,  3.3552e-02,
-1.5797e-02,  3.6242e-02, -9.9317e-03,
 6.0637e-03,  1.4210e-02, -3.6083e-02,
-9.3552e-02,  4.6970e-04, -2.0397e-01,
 2.6445e-01, -8.7635e-02,  4.4420e-02,
-3.1034e-02, -2.4809e-02, -7.9973e-02,
 3.9832e-02,  5.6320e-02, -1.6584e-01,
-1.0379e-01,  1.9124e-01, -1.5995e-01,
 1.5375e-02, -6.5487e-02, -2.2607e-01,
 4.0743e-02,  5.3877e-01, -1.8405e-01,
-2.0368e-01,  9.7637e-02, -8.3024e-03,
 5.8429e-02,  6.7743e-02, -3.1835e-02,
-9.1126e-02,  1.1041e-01, -1.5942e-01,
-9.9946e-03,  1.7703e-01, -4.3255e-02,
-4.3542e-02,  5.5290e-02,  7.2669e-03,
 1.0514e-01, -3.2062e-02,  8.5025e-02,
 4.7238e-02, -8.7266e-02,  1.5054e-01,
-4.5008e-02, -1.3546e-01, -4.5637e-02,
 6.7271e-02, -6.6028e-02, -1.0247e-01,
-1.5471e-01,  2.4660e-02,  2.0746e-02,
-2.2287e-02,  1.7142e-01,  5.7267e-02,
 1.3805e-01,  2.3145e-01,  1.8566e-01,
-1.6814e-01,  6.3401e-01,  2.3465e-01,
 6.9327e-02,  4.9812e-02, -3.4604e-02,
 1.1620e-01,  3.7884e-02,  9.6009e-02,
 1.5200e-01, -3.0406e-01,  3.3117e-01,
-5.5577e-40,  4.2861e-40,  4.2928e-40,
 2.9413e-40, -4.9091e-40,  6.1044e-40,
 5.1992e-40, -4.3213e-40, -6.4278e-42,
-1.2156e-41, -9.0106e-41, -5.6461e-41,
 6.7939e-40, -7.3206e-40, -3.9711e-41,
-7.4119e-40, -7.8616e-41, -1.5225e-41,
-2.4095e-10, -3.3202e-10, -3.3933e-10,
-3.1747e-10, -4.1122e-10, -4.1069e-10,
-2.7581e-10, -3.4352e-10, -3.1796e-10,
-3.9805e-40, -2.5352e-40, -5.8848e-40,
 2.3525e-40,  7.2841e-41, -2.8202e-40,
 6.2541e-40, -3.0049e-40, -5.5016e-40,
-4.3227e-41,  3.3051e-41,  2.1926e-39,
 3.6859e-40,  3.0199e-40,  5.1046e-40,
-3.4874e-40, -6.0714e-40,  2.6955e-41,
-3.8249e-40,  4.3046e-40, -6.3792e-40,
-3.8337e-41,  5.4799e-40, -1.1553e-40,
 3.6351e-40,  5.0462e-40,  1.6541e-40,
 5.9147e-40, -6.0144e-42,  6.8537e-40,
-5.7882e-40,  1.4350e-40,  6.5315e-40,
 5.7509e-40,  6.0992e-40,  2.2348e-40,
 5.4661e-40,  2.6654e-40, -1.6661e-40,
-5.2731e-40,  1.3934e-40,  5.1983e-41,
-5.2514e-40,  3.0844e-40, -3.4043e-40
}
,)"
R"(
{
 4.9797e-02,  1.7344e-02, -1.0028e-01,
-1.5218e-02,  3.2069e-02,  2.1757e-02,
-1.2627e-01, -1.8398e-01, -3.2554e-02,
-4.0863e-02, -1.4385e-01,  3.8687e-02,
 8.7972e-02,  2.0804e-01, -8.1046e-02,
 3.5250e-03, -1.9311e-01, -2.6860e-01,
-5.0595e-02,  1.0865e-01,  7.8966e-02,
-4.3057e-02, -1.4518e-01, -1.5197e-01,
-8.7345e-02, -6.9877e-03,  3.7020e-02,
 4.0063e-02, -7.0256e-03, -1.1711e-01,
-5.4260e-02,  6.8589e-02, -2.1553e-01,
-2.3390e-02, -1.2383e-01,  1.9448e-01,
-8.7979e-02,  1.0218e-01, -1.1069e-01,
-5.0038e-02, -1.7105e-01,  3.3148e-01,
 2.5134e-02, -1.0616e-01, -2.9208e-02,
-9.4628e-02, -5.0686e-02,  1.9851e-01,
 8.6313e-02, -2.7759e-01,  1.0726e-01,
 1.2736e-02,  2.5671e-01, -2.5944e-01,
 2.7605e-01,  2.1954e-01, -1.6458e-01,
 5.1504e-02,  9.2099e-03,  3.3447e-01,
-5.8482e-02,  2.2706e-01,  6.3022e-02,
 2.3200e-40, -4.0718e-40, -3.5282e-40,
-2.2962e-40, -5.7106e-40, -5.3744e-41,
-5.0681e-40, -4.8941e-40, -3.6821e-40,
 2.5909e-01,  1.6235e-02, -2.5420e-03,
 5.1874e-01,  1.6976e-01,  1.9238e-02,
 1.6555e-01,  3.8547e-02, -1.5985e-02,
-9.3822e-02,  1.0144e-01, -3.0740e-02,
-1.3598e-02, -4.6433e-02,  2.2108e-02,
-1.5328e-01,  2.3690e-03, -2.7983e-02,
-2.2820e-01,  1.1347e-02, -4.7096e-02,
-1.2724e-01,  6.2541e-02,  1.2533e-02,
-3.1650e-02, -6.2042e-03,  1.3698e-02,
-2.2762e-03, -1.6032e-01, -5.7491e-02,
-5.0303e-02,  1.3843e-01, -8.2110e-02,
-2.3059e-02, -8.0865e-02,  3.7068e-03,
 7.1424e-02, -3.1767e-02,  5.5948e-02,
-2.2705e-02,  2.1243e-03,  9.7844e-02,
-1.0080e-01,  2.7760e-02, -4.9373e-03,
-1.4756e-01, -1.2031e-01,  2.0025e-02,
-3.1595e-01,  5.1137e-01,  4.7956e-02,
-3.2422e-02,  9.7466e-03,  3.5242e-02,
-6.7666e-02,  1.1462e-01, -3.5587e-02,
 1.1959e-01, -2.7204e-03,  5.4378e-02,
-3.2878e-02,  5.2510e-02, -1.3153e-02,
-4.5539e-41,  3.9298e-40, -3.4940e-40,
 1.4600e-40, -3.2539e-40,  1.5473e-40,
-3.0523e-41, -2.0709e-40,  5.0203e-40,
-6.4715e-02, -9.4663e-02,  9.9256e-02,
 1.3537e-01,  5.3098e-01, -1.5170e-02,
-9.8711e-02,  4.0590e-02,  1.4176e-02,
-9.7166e-02, -1.0378e-01, -7.4776e-02,
 2.5714e-02,  2.9173e-02,  1.0741e-01,
-2.1824e-01, -1.1614e-01, -1.5774e-02,
 1.8924e-01, -9.4563e-03, -9.1887e-02,
-2.6068e-03, -3.6727e-01, -2.5262e-01,
 1.1392e-01,  8.5993e-02,  1.5428e-01,
 1.2855e-02, -9.3306e-02,  1.0816e-01,
 2.1955e-02, -2.1500e-01,  9.7376e-02,
 1.2347e-01,  1.4408e-01, -2.3344e-02,
 7.8354e-02, -2.8305e-02, -5.7389e-02,
 1.1374e-01,  1.0559e-01,  4.0298e-02,
 7.3669e-02, -1.0631e-01,  4.3469e-03,
 3.6728e-02, -1.0026e-02,  4.4353e-02,
-4.8244e-02,  3.4574e-02, -7.1158e-02,
 2.1913e-02, -1.6265e-01,  1.3327e-01,
-2.5156e-01,  2.4682e-01, -6.5907e-02,
-3.0124e-01,  2.3403e-01, -1.6368e-01,
-1.9269e-02,  1.5492e-01, -3.1138e-02,
-7.2503e-42, -3.0681e-40, -3.8003e-40,
 5.5141e-40,  6.0322e-40,  1.5700e-40,
-1.2225e-40,  2.1020e-40, -1.6536e-40,
 1.5666e-01,  1.0638e-01,  8.5878e-03,
 5.6267e-01,  3.6003e-01,  4.4611e-02,
 1.5994e-01,  1.6095e-01,  3.4463e-02,
 5.3049e-02, -1.5818e-01,  5.5151e-02,
-1.4333e-01, -1.6824e-01,  5.5075e-02,
-2.4284e-01, -1.0440e-01,  4.6043e-02,
-2.7966e-02, -1.2212e-01,  2.7314e-02,
-4.6396e-02, -2.7625e-01,  6.5133e-02,
 7.0825e-02, -1.0743e-01,  3.1158e-02,
-3.6488e-02, -2.3873e-01,  2.5322e-03,
 2.4735e-02,  2.7896e-01,  2.0450e-02,
 1.5495e-02, -4.9258e-02, -1.4110e-01,
 5.0034e-02,  4.4547e-02,  1.3854e-04,
-6.2222e-01,  2.5153e-02, -5.4604e-02,
 2.4900e-03,  2.2442e-02,  2.3571e-02,
-1.0009e-01, -2.2815e-01,  1.1901e-01,
-3.8198e-01,  2.8921e-01,  2.7092e-01,
-3.3170e-02, -1.8929e-02,  5.5622e-02,
-1.5950e-03,  7.8702e-02, -5.4303e-02,
-1.5732e-01,  1.5377e-01, -1.4777e-02,
-7.5107e-03,  6.4360e-02,  1.3060e-02,
 3.3767e-34,  1.3170e-33,  7.5236e-34,
 1.9245e-34,  8.7398e-34,  5.8794e-34,
 8.6313e-35,  5.1722e-34,  4.3812e-34,
-8.7121e-02, -5.9551e-03,  1.8358e-02,
-2.4580e-01,  1.2409e-01,  3.8271e-03,
 9.0117e-02,  1.3185e-01,  1.6438e-02,
 2.7841e-03, -5.8072e-02,  2.0691e-02,
 6.4194e-02, -1.9641e-02, -1.0075e-02,
 7.3093e-02, -6.9765e-02,  1.1877e-02,
 1.3719e-01, -6.7362e-02,  2.4206e-02,
-4.8552e-02, -1.2681e-01, -4.1612e-03,
 7.1373e-02, -3.3882e-02,  3.1716e-02,
-1.8076e-02,  8.1828e-02,  4.0457e-02,
-2.7750e-02, -7.2997e-03,  2.0906e-02,
 2.3370e-02, -6.8065e-02, -2.8267e-02,
-2.8215e-02, -1.0271e-02, -2.2844e-02,
 2.8285e-02,  6.0390e-02, -1.0696e-02,
 2.1448e-02, -7.5739e-02,  3.8789e-02,
 8.7354e-02,  1.3826e-01,  6.0291e-03,
 1.4070e-01,  3.5219e-01,  1.6039e-01,
-6.1694e-02,  4.0799e-02,  2.4946e-02,
 6.1788e-02, -1.0661e-01,  1.2872e-02,
-6.4082e-02,  1.7444e-01, -4.3407e-02,
-7.3389e-02, -1.1922e-02,  4.4684e-02,
 2.8042e-07,  2.7007e-07,  2.3625e-07,
 2.7807e-07,  2.6815e-07,  2.4389e-07,
 2.4925e-07,  2.5652e-07,  2.2932e-07,
 4.6266e-02, -1.1495e-01,  1.0927e-01,
-1.7005e-01,  3.2018e-01,  1.6980e-02,
 1.8423e-01,  1.7288e-01, -4.9140e-02,
 1.8204e-01, -1.7026e-01,  2.2275e-02,
-9.6129e-02, -7.5094e-02,  1.4372e-02,
-3.2290e-02, -5.8395e-02,  1.0507e-02,
 1.2931e-01, -1.1475e-01, -5.4053e-02,
-1.4634e-02, -2.8756e-02, -1.2519e-01,
-7.0642e-02, -9.8695e-02,  1.7477e-01,
-5.3602e-02, -1.1736e-01,  5.7743e-02,
-5.1917e-02, -4.4510e-02,  4.3679e-02,
-7.0202e-02,  1.6929e-01,  4.0830e-01,
 1.5246e-01,  2.3265e-02, -1.8212e-02,
-9.3065e-02, -8.7416e-02,  7.4110e-02,
-4.5377e-02, -1.7464e-01,  2.3478e-01,
 3.6250e-02, -1.9458e-01,  6.1893e-02,
-3.5872e-02,  3.1101e-01, -6.8570e-02,
-2.8722e-03, -1.9809e-01,  9.4725e-02,
-1.8117e-01,  1.0357e-02, -1.2142e-01,
-1.4964e-01, -7.2898e-03, -5.5310e-02,
-1.7302e-01, -2.2731e-01, -8.7447e-02,
-4.2673e-40,  6.0039e-41,  5.1222e-40,
 2.1874e-42,  4.6373e-41,  1.7525e-40,
-5.1202e-40, -6.0520e-40,  3.2346e-40,
-1.1972e-02,  1.8985e-03, -5.8161e-02,
-4.1253e-02, -8.6592e-02, -5.7520e-02,
-1.0705e-02, -2.3401e-02,  2.4563e-02,
 7.5016e-02, -3.0411e-02, -4.1009e-02,
-3.2805e-02, -3.8099e-02, -2.5788e-02,
-2.1201e-01,  1.8789e-02, -6.4905e-02,
-2.9190e-02, -1.8764e-02,  2.1533e-01,
-6.0753e-02, -3.1598e-01,  4.4685e-02,
 6.0285e-02, -2.4542e-01, -1.0096e-01,
 1.1690e-02, -1.8609e-01, -1.0523e-01,
 6.2053e-02,  5.4312e-01,  4.0942e-03,
 2.2555e-02, -1.7790e-01, -6.5769e-02,
 4.5532e-02, -7.6834e-03,  5.1407e-02,
-3.7387e-01, -2.0435e-01,  6.1209e-02,
 4.1898e-02,  2.7135e-02,  1.8511e-01,
-1.2081e-01, -1.1248e-01,  1.9980e-01,
 2.7368e-01, -9.9409e-03, -2.5312e-01,
-1.3883e-01,  1.6030e-01, -2.0875e-02,
 3.2367e-01, -3.4159e-02, -2.5767e-02,
 2.4911e-01, -2.2750e-02,  1.8410e-01,
-6.6850e-02,  1.1942e-01,  6.8278e-02,
 1.0245e-40, -2.4890e-40,  1.8493e-40,
 5.8169e-40,  1.4988e-40,  4.5191e-40,
 4.7262e-40,  2.0290e-40,  2.6495e-40,
-5.4179e-02, -8.1237e-02, -4.3848e-02,
-9.3465e-02,  3.0611e-01,  5.3353e-02,
 1.9971e-02, -8.0686e-03, -1.0444e-01,
-8.7531e-02,  7.3262e-02,  1.0250e-02,
-1.1317e-01,  5.8175e-02, -7.8831e-02,
-3.3146e-01,  5.2934e-02, -5.0444e-02,
 1.2288e-02, -3.2486e-02,  1.7554e-03,
 1.9227e-02, -3.5867e-01,  5.7271e-02,
-3.4622e-03,  6.0023e-02,  3.9096e-02,
-1.6992e-02,  1.1134e-01,  1.6535e-02,
 4.3413e-03,  3.0204e-01,  2.1745e-01,
 6.8594e-02,  6.1805e-01,  4.1312e-01,
 1.8283e-02, -1.1851e-01,  7.7802e-02,
-4.0553e-02,  1.1894e-01,  8.0760e-02,
-1.0084e-01,  1.0608e-01,  1.8108e-01,
 2.5648e-02,  1.9312e-01,  3.3512e-02,
 2.0461e-02,  2.8094e-01, -1.6280e-02,
-2.8162e-02, -3.9213e-01, -1.3324e-01,
 1.9464e-02, -1.1641e-01, -4.7202e-02,
 1.4384e-01,  2.5900e-02,  1.7846e-01,
 8.7267e-02, -3.2034e-01, -4.3373e-02,
 3.9698e-28,  2.4137e-28,  4.9542e-29,
 2.3430e-28,  1.5342e-28,  5.0144e-29,
 1.0642e-28,  9.0909e-29,  2.7066e-29
}
,)"
R"(
{
 1.2839e-01,  3.6808e-02, -6.7379e-02,
-4.5296e-02, -1.1937e-02, -2.8355e-02,
-4.2202e-02, -6.1179e-03,  3.1314e-02,
-9.0346e-02, -2.9228e-01,  1.4110e-01,
 7.5519e-02,  2.0201e-01,  2.2555e-01,
-2.9237e-02, -9.0555e-02, -1.1268e-01,
 5.4562e-02,  4.7987e-02, -1.0501e-01,
-1.1527e-01,  1.5322e-01,  4.1475e-02,
-8.3322e-02,  1.1301e-03,  1.6360e-02,
-3.9503e-02, -1.7087e-01,  5.9261e-02,
 3.7228e-03,  1.4178e-01, -3.8039e-01,
 2.4087e-02, -3.8799e-02,  1.7630e-02,
-1.2344e-01, -4.9896e-01, -4.2380e-02,
 1.8317e-02,  3.8166e-01,  1.7205e-01,
-2.7176e-02,  9.8688e-02,  1.4743e-02,
 1.3840e-01, -2.6926e-02, -1.3501e-01,
-2.8616e-02,  1.9102e-01, -1.3128e-01,
 3.2426e-02, -1.3791e-02, -1.6369e-02,
-7.3400e-02,  4.6461e-02, -5.6989e-02,
 9.0619e-03, -6.5641e-03, -3.5915e-02,
 3.3651e-02, -7.0888e-02,  9.3330e-03,
 1.4143e-01,  6.9674e-01,  1.2663e-01,
 7.4295e-02,  1.9085e-01,  9.3635e-02,
-7.0578e-03,  1.0442e-03,  1.1247e-02,
-3.8944e-02,  1.7475e-02, -2.9053e-02,
-2.4826e-01, -3.1872e-01, -6.4085e-02,
 6.5441e-02, -5.0554e-02,  7.9579e-02,
-4.6935e-02,  1.2233e-01, -8.3620e-02,
-4.2468e-02, -2.5246e-01,  1.9895e-01,
 1.1276e-02, -1.6523e-02,  1.5225e-02,
-3.7841e-01, -7.0257e-02, -1.5498e-02,
 1.3369e-01, -3.4227e-01, -1.5118e-01,
 8.1283e-02,  5.7263e-02,  9.8518e-03,
 7.0249e-02,  2.7798e-01,  1.7434e-01,
 5.5501e-02,  3.5310e-02,  2.4487e-02,
-1.7118e-03, -4.5231e-02, -1.4902e-01,
-2.5514e-02,  8.4862e-02, -1.2275e-01,
 8.1766e-02, -1.1157e-01, -1.0164e-02,
 1.1668e-02, -2.2220e-02,  1.0136e-01,
 1.1680e-01,  1.3337e-01, -1.7552e-01,
 7.8990e-02, -1.7695e-01,  5.1244e-02,
 3.3530e-02,  4.1828e-03, -2.5288e-02,
-6.4670e-02, -5.0272e-02, -1.1713e-01,
-4.6413e-02,  2.5228e-03, -4.2654e-02,
-9.8468e-02, -4.4285e-02, -1.2282e-02,
 2.1131e-01,  4.7841e-01,  1.5529e-01,
-1.8930e-02, -1.3098e-02,  8.6349e-02,
-3.8244e-03, -2.5567e-02,  1.0353e-02,
-4.6704e-02, -1.1502e-01, -2.2582e-02,
 1.9767e-01,  1.4746e-01, -7.9376e-03,
-1.0393e-01, -4.4472e-02, -1.0617e-01,
 7.0639e-02, -4.2210e-02,  2.8618e-02,
 7.2998e-02, -6.3543e-02,  2.6834e-01,
 3.0146e-04, -7.4324e-02,  2.7947e-01,
 1.3418e-01,  2.8807e-02, -9.8958e-02,
 3.3813e-01,  3.7600e-01,  4.9650e-02,
 6.2473e-02, -1.5477e-01, -3.4063e-02,
-4.6960e-02,  1.0348e-01,  1.1841e-01,
-8.4937e-02,  1.6224e-01,  8.9997e-02,
 3.5439e-02,  1.9146e-01, -1.9804e-01,
 6.5525e-02,  3.9822e-04,  7.2553e-03,
 9.2212e-02, -2.4144e-01,  9.7701e-02,
 6.4337e-02, -1.6986e-01,  3.7214e-02,
-4.1267e-01,  1.8074e-01,  1.4673e-01,
 3.5280e-01, -5.8927e-01, -8.2559e-02,
 1.3865e-01,  1.4856e-01, -1.1659e-01,
-4.3157e-02,  6.9689e-02, -8.1445e-02,
-1.8658e-01,  9.5778e-02,  8.6497e-02,
-3.9499e-02,  1.6366e-01, -4.4598e-02,
-1.6076e-01,  1.2075e-01,  4.1906e-02,
-1.0250e-01,  2.2249e-01,  6.0865e-02,
-5.7662e-02, -7.8243e-02,  7.8855e-02,
-1.5234e-02,  1.5172e-01, -9.6607e-02,
 2.0345e-01,  2.5715e-01, -1.3886e-01,
 3.7036e-02,  1.0451e-01,  2.8231e-03,
 2.1919e-03, -1.5836e-02,  1.0705e-02,
-2.7962e-02,  3.9029e-02, -5.6236e-02,
-2.2419e-02, -4.9026e-02,  8.6997e-02,
 6.9576e-02,  5.6413e-02, -2.0668e-03,
-2.7232e-01, -1.2212e-01,  1.7850e-02,
-1.5460e-02,  3.3619e-02, -1.5672e-02,
 2.7137e-02,  8.7948e-02,  6.5268e-03,
 1.2782e-01,  5.9768e-01,  4.9635e-01,
-1.3368e-02,  8.6282e-03,  3.3810e-01,
-9.3113e-03, -3.1972e-02, -6.6279e-04,
 1.7559e-01,  4.0021e-01, -3.7315e-01,
-3.9770e-02,  8.0975e-02, -2.0559e-01,
-1.0865e-01,  9.9715e-02,  4.1955e-02,
 2.0645e-02, -2.0768e-02, -2.0278e-02,
 4.7991e-02, -1.4758e-01, -3.2556e-02,
-4.3636e-02, -6.9939e-02, -5.9770e-02,
 1.5002e-01,  5.8615e-01, -8.9843e-02,
 1.7086e-02, -8.4164e-02,  1.5284e-01,
-4.3161e-03, -4.6843e-02, -2.6635e-02,
 3.3487e-02,  3.7943e-01, -1.2899e-01,
-3.9367e-02,  2.5167e-02,  7.3345e-03,
 2.2266e-02,  1.2257e-01, -2.5187e-02,
 3.1926e-01, -4.5270e-01,  3.2100e-02,
-4.9325e-02,  4.3159e-02,  5.6320e-04,
 7.6897e-03,  1.4560e-01, -8.1121e-02,
 3.6720e-02,  7.9012e-02,  2.3002e-03,
-1.8697e-02, -5.0966e-02, -1.2706e-01,
-5.2716e-02, -2.0045e-02,  1.7373e-02,
-3.7597e-02,  1.3066e-01, -3.2182e-02,
-3.5386e-02,  1.9937e-02,  7.3493e-03,
 6.5403e-02,  1.6946e-01, -7.3808e-03,
 5.0281e-02,  2.4233e-01, -4.6206e-01,
-5.2758e-02,  1.7568e-02,  4.9886e-03,
 3.1907e-02,  1.8096e-01, -4.5404e-02,
 1.2480e-01,  3.4263e-01,  1.7586e-01,
-6.6024e-02,  5.6522e-02,  6.7026e-02,
-2.3154e-02, -3.7671e-02,  2.4725e-02,
 4.5601e-02, -4.0626e-02,  2.2742e-02,
-4.1183e-02,  3.6681e-02,  1.9755e-02,
 7.3044e-02,  7.8996e-03, -2.6487e-02,
 9.1459e-02, -2.4713e-01,  5.4887e-02,
-3.6731e-03,  1.2533e-01, -3.3009e-02,
-2.7194e-02, -3.3904e-01,  5.2276e-03,
 8.3188e-02,  1.8996e-01,  8.3122e-02,
 3.2673e-03,  3.2862e-02, -2.4484e-02,
-1.8961e-01, -1.6063e-02, -5.5199e-03,
-6.2875e-03,  1.8888e-03,  2.4141e-02,
 1.1211e-01, -1.4961e-01, -8.7362e-02,
 6.7418e-02,  4.9817e-02, -2.0935e-01,
-3.4556e-02, -9.9122e-02, -2.7863e-01,
-9.0968e-03,  2.0878e-01,  1.6574e-01,
-4.1758e-02,  4.1033e-02, -5.4355e-04,
 1.4621e-01, -4.7341e-02,  1.4747e-02,
 5.1022e-02, -7.1310e-02,  2.9747e-03,
-3.7785e-02, -1.2894e-01, -1.2228e-02,
 2.8978e-02, -1.3409e-01, -1.5665e-02,
 8.3866e-03,  1.4079e-01,  9.1289e-02,
 5.6058e-02,  1.4837e-02,  9.9984e-02,
 6.0923e-02, -3.4811e-01, -1.1649e-01,
-8.5065e-03,  1.5664e-01,  3.1240e-03,
 2.5650e-02, -8.4517e-02, -2.7035e-04,
-1.4427e-01,  4.9479e-02,  1.8414e-03,
-4.2500e-02, -3.0061e-02, -1.9405e-02,
-3.7432e-02, -1.9614e-01, -8.1843e-02,
 8.8076e-02,  1.5599e-01, -2.5447e-01,
-3.0605e-03,  6.1968e-02,  5.3989e-02,
-8.1379e-02,  1.6173e-02, -2.3907e-02,
 6.8640e-02,  2.2666e-02, -8.3857e-03,
 1.9910e-02,  8.7107e-02,  2.7861e-02,
 2.0986e-02,  1.6138e-01, -5.2335e-02,
 2.0453e-01, -2.8827e-01, -7.2135e-02,
-1.2089e-01,  6.9849e-02, -1.9183e-02,
 9.1797e-02,  2.3139e-02, -6.6590e-02,
-1.4124e-01,  4.9936e-01, -1.4936e-01,
 1.0355e-01,  3.0107e-01,  5.1952e-03,
 1.0997e-01, -1.7216e-02, -4.3230e-02,
 2.4609e-02, -2.4258e-01,  1.0183e-02,
-1.2358e-01, -2.2534e-01, -2.3282e-02,
-5.7883e-03,  1.0484e-01,  3.1645e-02,
-1.0173e-01, -9.3762e-02, -2.9311e-02,
-3.7341e-02,  9.1380e-02, -8.6212e-02,
 1.1523e-01,  3.9193e-02, -3.1336e-02,
-4.4188e-01,  2.6500e-01, -5.4275e-02,
 4.2830e-02,  3.9504e-02,  1.6036e-02,
-6.3901e-02,  5.3358e-02,  2.8781e-02,
 1.5087e-01, -5.6757e-02, -4.8234e-02,
-1.0333e-01,  9.1837e-02,  7.1969e-03,
 1.0475e-01, -9.6164e-03, -6.0523e-02,
-6.4989e-02, -4.5471e-02, -8.1156e-02,
 2.2787e-01,  4.2122e-01,  4.3752e-02,
-8.4524e-02, -1.2824e-01, -3.2161e-02,
 2.2523e-01,  1.6167e-01, -5.5443e-03,
 8.4938e-02,  6.2894e-02,  1.2985e-03,
 1.4082e-02,  4.4413e-02, -1.0861e-02,
 4.2678e-02, -2.2227e-02, -1.1612e-02,
-2.7226e-02, -3.0670e-02,  3.0298e-02,
 1.7987e-02, -9.3078e-02,  7.4212e-02,
 4.7606e-02, -7.6087e-02,  3.2006e-02,
 1.2621e-03, -1.1247e-02,  3.3190e-02,
-1.8513e-02,  8.8079e-02, -4.1406e-02,
 7.6062e-02, -1.1562e-01,  4.1330e-02,
-7.8028e-02,  4.1622e-02, -2.3733e-03,
 3.9900e-03,  1.6476e-02, -5.3494e-02,
-7.7035e-03, -6.6342e-03,  7.0292e-02,
-1.3032e-02,  5.6734e-02,  3.9049e-02,
-7.5841e-03, -5.6070e-02,  1.6285e-02,
 4.8887e-02, -3.9818e-01, -1.0952e-01,
-4.3564e-02, -9.9183e-02, -1.9498e-02,
-2.1922e-03,  4.9724e-02, -5.5142e-03,
-1.0049e-03, -6.8525e-02, -2.1482e-03,
 2.1455e-03,  4.5904e-02, -3.6267e-03,
-3.5904e-02, -2.2641e-02,  1.2457e-02,
 6.6951e-02,  5.3222e-03,  1.0474e-02,
-2.0614e-02,  7.3937e-02, -3.0696e-02,
-3.3673e-02,  9.8776e-02, -9.4984e-03,
 2.8029e-02, -1.0659e-02,  1.1856e-02,
 6.4507e-03,  2.0490e-02, -1.4227e-03
}
,)"
R"(
{
 4.4862e-03, -5.5155e-02,  3.0312e-02,
 4.1294e-02,  6.9984e-02,  1.2511e-01,
 5.5141e-03, -1.4298e-01,  2.3232e-01,
-1.7155e-02,  7.6742e-03, -1.3772e-02,
-3.4855e-03,  4.1676e-01, -1.5366e-01,
 9.9035e-03, -7.2083e-02, -1.2158e-01,
 2.7119e-02,  6.9204e-04,  2.5744e-02,
 9.3203e-02,  2.3255e-02, -2.4483e-02,
 5.1335e-02,  8.0554e-02, -1.0602e-01,
-2.7149e-02, -2.3273e-02, -7.4590e-02,
-2.7896e-01,  6.4507e-01,  5.8778e-02,
 3.2651e-03,  6.5670e-02, -1.0128e-01,
 2.6349e-02,  7.9302e-03, -4.1331e-02,
 2.1876e-01,  5.8368e-03, -1.4418e-01,
 2.2312e-02, -7.4867e-03, -1.2602e-01,
 3.5748e-02,  2.6653e-01,  4.5545e-02,
-8.0317e-02, -3.0223e-01,  1.2873e-01,
-7.8588e-02, -5.7888e-02,  3.3439e-02,
 4.2851e-02,  3.5972e-01,  6.7142e-01,
-6.4109e-02,  3.4969e-01, -1.8474e-01,
 1.3421e-02, -4.1062e-02,  2.1152e-02,
-8.0892e-03,  5.9220e-02,  4.7748e-03,
-4.5731e-02, -4.0852e-02,  1.6566e-02,
 2.0628e-02, -7.2901e-02, -2.6147e-03,
-4.8597e-02,  1.0379e-01,  2.1971e-02,
 1.6023e-01,  5.8693e-01,  7.0162e-02,
-6.5360e-02, -7.0749e-01, -1.3525e-01,
 1.1306e-02,  3.2044e-02, -6.6158e-02,
 9.7311e-02, -3.6739e-01,  2.3054e-02,
 2.4817e-02,  1.7704e-01,  6.7592e-02,
 6.0719e-02,  1.0031e-01,  1.6505e-01,
 1.9650e-01, -5.9757e-01,  4.7844e-02,
 4.6389e-02,  5.3369e-02, -6.4895e-02,
 8.4262e-02,  4.9003e-03, -2.5576e-02,
-1.8267e-01, -4.7049e-02,  1.1214e-01,
-9.4736e-02,  1.2931e-01,  1.8470e-02,
-5.0342e-02,  8.5759e-02,  6.5877e-02,
 8.5118e-02,  1.4021e-01,  1.0771e-01,
 5.6879e-02,  2.4132e-01,  1.6770e-01,
 7.8506e-02, -1.3634e-01,  4.2381e-02,
-2.2479e-02,  9.3006e-02,  6.1137e-03,
-9.5164e-02,  4.9613e-02, -3.3716e-02,
 5.0584e-02,  1.5802e-01, -3.2241e-01,
-1.1030e-02,  4.2624e-01, -2.3570e-01,
-1.9777e-02,  4.4884e-02, -5.8820e-02,
 1.1718e-01, -2.1559e-01,  5.2122e-02,
-2.0436e-01, -4.8309e-01, -1.4330e-01,
 5.0900e-02, -1.3252e-01, -1.2908e-02,
-2.9484e-03,  1.1399e-02,  9.6124e-03,
-3.4635e-02,  6.9747e-03,  5.6921e-02,
-6.2121e-02, -3.6712e-03,  8.6796e-02,
 2.1525e-02, -1.1328e-02,  1.2112e-04,
 2.9774e-03, -1.2287e-01,  7.8299e-02,
 1.2135e-01, -2.4746e-01,  5.1003e-02,
-3.0374e-02, -9.1556e-02,  1.5843e-02,
 1.2198e-02, -3.8379e-01,  1.6659e-01,
-1.8820e-01, -3.4096e-01,  1.8438e-01,
 1.0640e-02,  8.6941e-02,  7.9967e-02,
-3.1412e-01,  4.2294e-01, -9.9180e-02,
-3.2423e-01,  9.5441e-02,  1.0459e-01,
-4.2336e-02,  3.2760e-02,  4.1807e-03,
 1.1974e-01, -3.2349e-01,  1.2068e-01,
 1.5200e-01,  7.8194e-02, -1.4755e-01,
 7.7038e-02,  1.1185e-01,  2.8725e-02,
 3.7990e-02,  2.0080e-01,  9.0222e-02,
-3.1634e-02, -1.5621e-01,  1.0401e-01,
 5.1685e-02,  7.0847e-02,  6.8406e-02,
-3.3171e-02, -1.4812e-01,  3.3392e-01,
 1.0290e-02, -2.5362e-02, -9.6456e-02,
-9.8877e-04,  1.7585e-02, -2.8255e-02,
-4.5111e-02, -5.1043e-02, -5.9618e-02,
 1.3813e-01,  1.0275e-01, -8.3379e-02,
-1.4536e-40, -1.7049e-40, -2.6781e-40,
 4.5792e-40, -8.0052e-40,  1.3645e-40,
-5.8774e-40, -6.6690e-40, -4.7571e-40,
 1.1836e-39,  1.5398e-40, -3.3819e-40,
 1.0031e-39,  1.5334e-39, -1.0368e-39,
-1.1428e-39, -2.8750e-40, -1.4830e-39,
 5.6364e-40, -9.6393e-40, -2.3064e-41,
 2.8909e-40, -5.8115e-40,  2.9852e-41,
 8.9843e-40, -7.5503e-41, -6.0335e-40,
 5.8073e-40,  2.9252e-40,  8.1503e-40,
-8.3606e-40,  3.8172e-40, -2.0389e-40,
-2.1905e-41,  7.0321e-40, -2.9226e-40,
 2.9957e-41,  2.6068e-40,  6.1324e-40,
-6.5178e-40,  5.1421e-40, -4.1157e-40,
 2.1416e-41, -1.6614e-40, -3.0843e-42,
 1.1010e-39,  2.8507e-40,  6.6821e-40,
-1.0739e-39, -3.0797e-40, -6.0685e-40,
 5.4170e-40, -6.1858e-40,  9.3049e-41,
-7.1340e-40, -7.1060e-40,  7.4842e-40,
 3.9906e-40,  1.2356e-40,  3.8682e-40,
 2.8630e-40,  6.2303e-40, -8.2832e-40,
-4.1904e-40,  4.8916e-40, -3.6125e-40,
-5.5393e-40, -2.4980e-40, -6.1877e-40,
 2.7289e-40, -1.8348e-40, -5.6663e-40,
 2.0753e-02, -5.0760e-02,  3.1196e-02,
 2.4381e-01, -4.2393e-03,  4.7721e-03,
-1.6330e-01, -6.7275e-02,  1.8902e-01,
-5.7002e-02, -2.5285e-02, -1.0609e-02,
-3.8253e-02, -5.4225e-01,  1.7773e-02,
-3.9082e-03,  4.6443e-01, -1.8976e-01,
-3.1337e-02, -1.0601e-01,  3.9304e-02,
 2.0866e-01,  7.3200e-01, -2.5141e-01,
-2.3475e-02, -7.1672e-02,  6.0524e-03,
-7.9610e-02,  2.7777e-01,  6.6222e-02,
-3.1812e-01,  6.6897e-01, -1.6115e-01,
-5.4502e-02,  1.2301e-01, -6.0121e-02,
-9.4793e-04,  6.1690e-02,  7.8304e-02,
 1.1127e-01, -2.2745e-01, -1.6914e-01,
 4.5696e-02,  2.2113e-01, -1.2915e-01,
 1.1402e-01,  1.1122e-01,  3.3348e-02,
 7.6369e-03, -4.7367e-01,  4.8115e-02,
-9.2239e-02, -9.1885e-02,  2.6641e-02,
 4.6204e-02, -1.1789e-01,  1.9332e-03,
 9.2452e-04,  7.3879e-02, -1.7173e-01,
-1.9696e-02, -4.5106e-02,  1.4349e-01,
-5.0189e-02,  1.0848e-01, -8.8565e-02,
 1.7436e-02, -2.6464e-01,  2.7054e-01,
-4.2207e-02,  9.2968e-02, -3.0513e-02,
 4.9114e-41,  1.9746e-40,  1.2233e-40,
-2.0670e-40, -4.2341e-40, -2.0134e-40,
-2.3361e-40, -4.3932e-40, -6.1880e-40,
-5.0380e-40, -2.9085e-40, -4.1818e-40,
-5.3070e-40, -2.2434e-40,  8.1341e-41,
-3.7355e-40,  3.5521e-40, -1.6547e-40,
 4.0925e-40, -3.0204e-40, -1.5836e-40,
-4.1249e-41,  6.4614e-40, -3.2914e-40,
 2.2549e-40,  1.7634e-41, -7.8249e-42,
-5.7023e-40,  3.1029e-40, -1.8546e-40,
 4.2800e-40, -1.0257e-40,  4.4205e-40,
-4.9791e-40, -3.0575e-40, -4.9614e-40,
-1.3952e-06, -1.5598e-06, -1.3928e-06,
-1.6075e-06, -1.7903e-06, -1.5767e-06,
-1.4846e-06, -1.6345e-06, -1.4283e-06,
-1.6111e-39,  1.9755e-39, -1.5992e-39,
 2.3436e-39, -1.6878e-39,  1.9576e-39,
 1.9301e-39,  1.9485e-39, -2.5164e-39,
 4.6954e-40,  4.3795e-40, -2.1897e-40,
-2.6361e-40, -3.4450e-40, -4.1820e-40,
 1.9397e-40, -5.0630e-40,  6.2587e-40,
-5.5178e-33, -6.8212e-33, -1.9703e-33,
-9.7254e-33, -1.2775e-32, -4.8490e-33,
-8.0945e-33, -1.3175e-32, -5.6330e-33,
 3.9556e-40, -7.9752e-41, -1.2253e-40,
-1.5353e-40, -6.1462e-40, -6.3068e-40,
-5.0724e-41, -3.3937e-40, -4.1864e-40,
 5.4633e-40, -1.9048e-40,  9.8151e-41,
 5.0253e-40,  6.8856e-40,  2.0574e-40,
 6.9116e-40, -4.7129e-40,  5.7111e-40,
 4.5038e-40, -1.7225e-40,  3.5263e-40,
-3.1182e-40, -5.3738e-40, -4.2105e-40,
-1.9672e-40, -6.2491e-40, -3.3242e-40,
 3.9168e-40,  4.8625e-40, -5.8588e-40,
-1.9080e-40,  1.7733e-40, -6.3480e-40,
-1.3768e-40, -4.6211e-40,  3.7028e-40,
-1.5374e-40,  5.5546e-40, -1.9342e-40,
 6.3251e-40, -6.1766e-41,  6.1642e-40,
 2.1262e-40,  3.3078e-40,  2.0088e-40,
-3.7053e-41,  4.7588e-40, -7.4817e-40,
 5.4387e-40, -2.2625e-39, -3.7592e-40,
-5.8768e-40,  5.7387e-40, -6.9880e-40,
 2.6834e-40, -6.4139e-40, -1.3363e-40,
-2.7595e-40,  4.2089e-40,  1.7535e-40,
 3.2535e-40, -5.9403e-40, -2.7694e-40,
 5.4065e-40, -4.1218e-40, -1.6712e-40,
-3.6472e-40,  4.3364e-40, -1.9790e-40,
 4.3337e-40,  3.1095e-42, -5.3438e-40,
-3.2346e-02,  7.4618e-02,  3.0748e-03,
 2.4184e-02,  5.7867e-01,  8.9846e-02,
 1.5224e-01,  2.3174e-01,  1.3998e-01,
 3.3207e-02,  2.3671e-02, -1.9599e-02,
 3.8106e-02, -2.5901e-01, -3.9227e-02,
 3.6996e-02, -7.2460e-02, -6.7212e-02,
 3.0244e-02,  1.0455e-01,  7.3787e-02,
 1.5678e-01,  1.1420e-02,  2.1930e-01,
 3.1050e-02,  1.4391e-02, -8.0939e-02,
 6.7096e-02,  3.7718e-02, -5.7282e-02,
 1.5450e-01,  2.2861e-01, -1.7659e-03,
 1.7962e-01, -3.5376e-03, -3.7800e-02,
-3.6771e-02,  1.2731e-01, -1.3643e-02,
-1.8994e-01,  8.0294e-01, -5.1181e-02,
-7.9692e-02, -4.0189e-01, -2.0857e-01,
 3.8272e-02, -3.8309e-03, -3.4311e-02,
-1.5035e-01,  1.3447e-01,  1.1817e-01,
 2.1948e-02, -3.6355e-02, -4.2483e-03,
 1.4962e-02, -5.1394e-03,  8.4858e-02,
-3.9637e-02, -1.1960e-02,  3.7374e-01,
 6.8093e-03, -5.6983e-02,  4.8428e-02,
 1.7013e-02, -4.5316e-02,  6.9025e-03,
 7.8210e-02, -1.3982e-01, -4.3760e-02,
-2.1668e-02,  4.6630e-02, -4.1925e-02
}
,)"+
std::string(
R"(
{
 2.2301e-40, -5.1577e-40,  5.5657e-40,
-1.2146e-40, -6.1648e-40, -2.4849e-41,
 4.7284e-40, -2.1640e-40,  4.5871e-40,
-9.3633e-09, -1.1534e-08, -9.0504e-09,
-1.1695e-08, -1.4555e-08, -1.1189e-08,
-9.1119e-09, -1.1013e-08, -8.1803e-09,
-4.4982e-40, -2.4567e-40, -2.9616e-40,
 2.9693e-40, -1.2375e-40, -5.8490e-40,
-2.4706e-40,  6.0205e-40, -8.1547e-41,
-4.8855e-40,  8.6999e-40,  1.4035e-39,
-1.0706e-40,  5.3827e-40, -1.6413e-40,
 1.4714e-40,  1.8114e-39, -4.4881e-40,
 6.0014e-40,  3.2035e-40,  1.6637e-40,
-1.4119e-41,  6.1914e-40,  6.0835e-40,
 4.6040e-40, -7.4277e-41,  6.4045e-40,
 3.6232e-40, -1.1303e-40,  5.9304e-40,
-5.1129e-40,  2.0621e-40, -6.1410e-40,
 2.5973e-40, -3.0913e-40,  3.7876e-40,
-8.3174e-40, -6.3449e-41,  1.2075e-40,
 6.9108e-40, -1.2133e-39, -1.2005e-39,
 1.0798e-39,  3.1821e-41,  2.2965e-39,
 5.9695e-41,  2.9583e-40,  4.4130e-40,
 4.7190e-40, -2.2070e-40,  5.5496e-41,
-3.2211e-40,  1.5302e-42, -7.4046e-41,
-4.3505e-02, -2.7903e-03, -2.2765e-02,
 6.7106e-01,  6.5908e-02,  1.7588e-02,
-8.2807e-02,  7.5865e-02, -2.0675e-02,
 1.9583e-02,  1.0029e-02,  1.6567e-02,
-3.9688e-01,  2.7688e-01, -6.1370e-03,
 6.3884e-02,  6.6924e-03, -3.0922e-04,
-8.1589e-01, -8.4710e-02,  5.4826e-03,
-1.2896e-01, -4.9859e-02, -1.8887e-02,
-2.5638e-02,  1.6394e-02, -8.2594e-03,
 6.0313e-40,  7.0252e-40,  7.3325e-40,
 6.0697e-40, -9.1199e-41,  5.8965e-40,
 5.4830e-40,  1.3014e-40,  1.5585e-41,
 2.5389e-02,  3.6854e-02,  2.1425e-02,
 5.6533e-01,  2.8130e-02, -1.2533e-02,
 6.0970e-02, -6.4916e-02,  6.7069e-03,
 7.9372e-40, -8.9257e-40,  3.4439e-40,
-6.1050e-40, -8.7012e-40,  2.0928e-41,
-1.0644e-40,  3.8209e-40,  1.9829e-40,
-6.5191e-40,  4.6290e-40, -1.1813e-40,
-5.9172e-40,  3.9819e-40,  6.2347e-40,
-6.4533e-40,  6.5302e-40,  4.9927e-40,
 2.2223e-02,  3.1888e-02, -4.8330e-03,
-4.1709e-01,  1.1851e-01,  2.0635e-02,
 2.8740e-02,  5.5908e-03,  1.9860e-03,
-6.7036e-02,  1.1728e-01,  3.6617e-02,
-1.6449e-01,  1.5272e-01,  1.5646e-01,
-1.7445e-01, -1.1890e-01,  5.5423e-02,
 1.1021e-01,  2.2729e-01,  3.7162e-02,
 1.7612e-01,  3.7873e-01,  6.4712e-02,
 7.3294e-02,  1.0694e-01, -7.8673e-02,
-4.5823e-02, -1.0430e-02,  6.1962e-02,
 1.6867e-02, -4.1500e-02,  1.0369e-02,
 2.3894e-03,  1.3534e-02, -1.8316e-02,
 2.4428e-40, -3.0160e-40,  2.3184e-40,
-4.9114e-40,  5.6685e-40, -3.6020e-40,
 2.2618e-40, -2.8145e-40,  2.1149e-40,
 6.4274e-03, -6.9188e-02, -3.4050e-02,
-2.4341e-01,  6.0807e-01, -8.0922e-02,
 1.5843e-02, -1.5813e-01, -3.9675e-02,
 4.1146e-18,  6.6838e-18,  4.4612e-18,
 7.2648e-18,  1.1760e-17,  7.5084e-18,
 5.1023e-18,  7.9991e-18,  4.9642e-18,
 6.1657e-44, -4.4745e-40, -2.5471e-41,
-5.5927e-40,  2.5144e-40, -4.9958e-40,
-2.7970e-41, -4.2162e-40, -9.4327e-41,
-1.3392e-01, -3.4372e-01, -1.0475e-01,
-3.9292e-02,  6.5609e-01, -1.4339e-01,
 1.9960e-02,  9.6727e-02, -1.3689e-02,
 6.2444e-40,  4.6230e-41,  6.6417e-40,
-1.2387e-41, -4.3021e-40, -3.8472e-40,
-6.6580e-40, -3.6573e-40, -5.1053e-40,
-8.3709e-41,  5.3468e-40,  5.1925e-41,
 2.6828e-40,  3.4806e-40,  4.6021e-40,
-6.3006e-40,  5.8436e-41, -5.0784e-40,
 1.4661e-39, -1.0295e-39, -1.8524e-39,
 1.5979e-39,  1.2282e-39,  1.1184e-39,
 1.5365e-39,  1.4818e-39,  5.7516e-40,
-1.6488e-39, -1.5046e-39, -1.6062e-39,
-2.4471e-39, -1.6041e-39, -1.5962e-39,
-1.6831e-39, -1.5538e-39, -1.4685e-39,
 4.6627e-40, -1.9240e-40,  7.1650e-41,
-9.1348e-40, -9.6926e-40, -1.8333e-40,
 9.9025e-40,  4.9738e-40,  8.6593e-40,
-8.4210e-40,  1.9883e-40,  6.8754e-40,
 4.3220e-40, -2.2847e-39,  8.6282e-40,
-1.1835e-39, -1.2010e-39,  4.9947e-40,
-8.7012e-40, -1.5751e-39, -1.5611e-39,
-1.6191e-39, -2.2652e-39, -1.6108e-39,
-1.6492e-39, -1.4710e-39, -1.4596e-39,
 4.4878e-40, -2.3044e-40, -1.2716e-40,
 3.4864e-40, -3.9670e-40, -3.9163e-40,
-5.3673e-40,  3.3124e-40,  4.3954e-40,
 6.5296e-02, -4.2459e-02, -7.5920e-03,
 8.5711e-02, -1.5231e-01,  9.4892e-02,
 1.7894e-01, -1.1282e-01, -1.2023e-01,
-1.9641e-02, -6.1318e-02,  8.7589e-03,
-1.5042e-01, -4.7054e-01, -9.2884e-03,
-3.0731e-01, -5.3044e-01, -6.1672e-02,
-5.5581e-02,  5.2476e-02,  5.0257e-04,
-4.0202e-02,  8.4304e-03, -2.8452e-02,
 1.6589e-01,  8.1536e-02, -5.7249e-02,
 8.7873e-41, -5.3947e-40,  1.0246e-39,
 6.9132e-40, -1.0261e-39,  8.1854e-40,
 1.1811e-39, -1.4913e-40, -9.8691e-40,
-4.4091e-02, -2.5690e-02,  2.5287e-03,
 3.9116e-02,  1.8780e-01,  3.3415e-02,
 2.1417e-01, -1.0638e-01,  1.1855e-03,
 1.4705e-40, -3.7559e-40, -1.8060e-40,
 3.7334e-40,  5.0580e-40, -1.3147e-40,
-7.5824e-40, -2.9414e-40, -8.4552e-41,
 8.2211e-40,  9.9963e-40,  2.5481e-40,
 9.6309e-40,  7.1566e-41, -7.8466e-41,
 9.1378e-40,  8.1715e-40,  4.0329e-40,
 8.1928e-02,  5.3856e-02, -1.6423e-02,
 8.7092e-02,  3.1193e-01,  1.7289e-01,
 1.0639e-01, -5.0525e-02,  1.1959e-01,
 2.0415e-02,  7.6764e-03,  3.9243e-03,
-3.7049e-02, -4.7649e-01,  1.2155e-01,
 4.0496e-01, -1.5621e-01,  1.6162e-01,
 7.9479e-03,  1.5419e-02,  1.4357e-02,
-8.6266e-03, -4.6262e-01,  1.9387e-01,
-1.2989e-01,  3.1591e-01,  4.6384e-02,
-3.3060e-01, -6.2639e-02, -4.1639e-02,
-1.8404e-01, -1.2679e-01, -4.2288e-02,
-1.3347e-01,  7.3955e-03,  7.1572e-03,
 1.0260e-40,  5.0443e-40,  7.5150e-41,
-8.0138e-40, -5.1952e-40, -5.3810e-40,
 6.2240e-40, -7.2089e-40, -8.2983e-41,
 7.6826e-02,  7.2747e-03, -2.1536e-02,
 4.6962e-01,  2.7892e-01,  1.1120e-02,
 2.9191e-01,  2.8061e-01, -2.5694e-02,
-2.2784e-41,  5.3347e-40, -2.2320e-40,
-6.5341e-40, -5.3078e-40,  5.3574e-40,
 4.3224e-40,  5.9639e-40,  5.3225e-40,
 7.3197e-40, -4.1074e-40, -6.4190e-40,
 1.9200e-40, -3.4686e-40, -3.4828e-40,
 1.3597e-40, -9.6111e-41,  2.8293e-40,
-9.1843e-04,  4.7658e-02, -3.4574e-02,
-7.9576e-02,  4.9680e-01, -8.2039e-02,
-1.9160e-01,  9.2349e-02,  7.7611e-03,
 4.5418e-01, -3.6049e-01,  6.9717e-02,
-4.2100e-01,  1.6569e-01,  1.7893e-01,
 6.4669e-02, -2.7643e-02,  3.6314e-02,
-5.5488e-02, -5.1523e-01,  2.1178e-01,
 1.1711e-01,  1.9290e-01,  1.2538e-01,
-7.2021e-02,  3.9554e-02, -5.9003e-02,
 6.4575e-02, -3.3115e-01, -1.0201e-01,
-4.6795e-02,  3.2588e-02,  1.3091e-02,
-2.2692e-02, -2.5946e-02, -7.6855e-03,
-7.8598e-40,  4.6894e-40,  7.4835e-40,
 3.8054e-40, -3.7914e-41,  8.0111e-40,
-6.9396e-40,  5.1566e-41, -6.9757e-40,
-8.3092e-02,  2.1895e-01,  5.6910e-02,
 7.8206e-02,  3.0028e-01, -4.6467e-02,
 3.6758e-02, -4.5057e-02, -2.9295e-02,
-5.0168e-41, -5.0527e-40,  1.2100e-40,
-4.6086e-40, -3.2207e-40,  4.8475e-40,
-5.4175e-40,  1.7829e-40, -3.6568e-40,
-7.0737e-40, -5.7576e-40, -7.5052e-40,
-7.0244e-41,  9.3036e-41,  3.7423e-40,
-6.4648e-41, -1.6230e-40, -4.3126e-40,
-1.4179e-01,  2.8184e-01, -3.0471e-02,
 3.5260e-03,  6.2338e-02, -3.3083e-02,
-2.0328e-02, -3.2591e-02, -2.6941e-03,
-4.0485e-02, -1.5335e-01, -6.3759e-02,
-1.4422e-02,  3.5298e-01, -1.0414e-01,
 1.4421e-01, -1.2106e-01,  2.1336e-01,
 3.5337e-02,  1.0067e-01, -5.0853e-02,
 7.4069e-02, -1.4488e-02, -5.8778e-02,
-6.2685e-02, -1.4876e-01,  1.1411e-01,
 1.1478e-01,  2.8992e-02,  1.0892e-01,
 5.4370e-02, -1.2378e-01, -1.9509e-01,
-9.3790e-02, -1.9047e-02,  3.1103e-02,
-9.0405e-40,  9.1758e-40, -8.0097e-40,
 9.3452e-40, -2.5643e-40, -6.3585e-40,
-6.9494e-40, -6.6890e-40,  7.6062e-40,
 4.9056e-02,  1.1592e-01,  7.2157e-02,
-1.6127e-01, -5.6038e-01, -8.3930e-02,
 7.4108e-04,  3.2458e-01, -3.6774e-02,
 3.8060e-40,  7.1096e-40, -2.4106e-40,
 2.0487e-41,  1.4397e-40, -4.3942e-40,
-7.6269e-40, -5.7632e-40,  6.8197e-40,
-1.9558e-40,  9.4105e-40, -7.5593e-40,
 5.1216e-40, -6.6741e-41, -8.8285e-40,
-7.9128e-40, -4.2296e-40,  4.9649e-40,
-6.5104e-02, -2.5021e-01,  7.7951e-02,
 1.2653e-02,  5.5758e-01, -1.5358e-01,
-8.3377e-02, -1.4604e-01,  6.5280e-02
}
,)"
R"(
{
-8.2898e-40,  8.3135e-40,  1.0309e-39,
-7.5171e-40,  3.0987e-40,  2.6176e-40,
-4.9376e-40, -4.7046e-40,  7.8094e-40,
-1.5657e-02, -1.3350e-01,  6.7348e-02,
 2.6494e-02, -9.9266e-02,  5.0726e-02,
-5.8945e-03,  1.8837e-01, -2.8899e-02,
-6.2717e-02, -5.9973e-03, -3.6643e-02,
 5.5996e-02,  6.9508e-02,  2.9479e-02,
-1.4034e-02, -5.4825e-02,  1.4337e-02,
 3.7908e-40, -2.3657e-39,  1.6637e-40,
 5.0818e-39,  6.3100e-39,  7.6280e-41,
 5.1634e-39,  2.7274e-39, -5.4976e-40,
-2.6263e-02, -2.1658e-02,  5.4952e-02,
-1.0501e-01, -3.6764e-01, -1.6755e-02,
-3.8708e-02,  4.6754e-03,  3.7346e-02,
-1.7650e-02,  4.5400e-01,  7.9489e-03,
 2.2808e-02, -3.8918e-01,  1.0474e-01,
-3.2150e-02, -6.6783e-03, -5.7799e-03,
-2.5905e-02, -1.1937e-02,  2.1708e-02,
 3.8924e-02,  3.8795e-01, -8.9885e-03,
 3.9501e-02, -6.1047e-01, -8.6775e-02,
-3.6670e-01,  8.2096e-03,  7.7474e-03,
-6.7986e-02, -9.1490e-02, -2.7807e-02,
 2.0358e-02,  1.9286e-03, -1.2897e-02,
 1.2525e-40, -6.2202e-40,  5.4044e-40,
-7.8405e-41, -4.2936e-40,  9.8608e-41,
-5.0886e-40,  5.3584e-40,  2.6031e-40,
-4.1952e-03, -3.3664e-02, -1.1999e-02,
-9.8589e-03,  1.3417e-01,  9.7898e-01,
-4.5813e-03,  1.7191e-01, -1.3318e-01,
-2.3440e-02, -5.9664e-02, -1.8485e-02,
 7.3281e-02,  4.0403e-02,  8.6244e-03,
 5.4678e-02,  9.2945e-02,  1.9789e-02,
-9.6595e-21, -5.8398e-21, -5.5514e-24,
-2.0348e-20, -1.5149e-20, -5.6179e-22,
-4.2092e-21, -4.2615e-21,  9.7220e-25,
-3.6579e-02, -1.2132e-01,  7.9614e-02,
-2.7256e-02, -8.2712e-02,  5.7647e-03,
-7.0076e-03,  1.1902e-02, -1.3719e-03,
 1.2083e-02,  8.8617e-02,  5.5834e-02,
 4.3136e-03, -3.7655e-01, -6.0504e-03,
 1.1504e-02,  7.6550e-03,  1.0419e-02,
-2.4832e-03, -7.8295e-03,  9.9402e-03,
-1.6722e-02,  1.0036e-02, -5.4916e-02,
 6.7936e-02,  7.3061e-02, -2.7377e-02,
 9.6065e-02,  1.2107e-01, -8.1992e-03,
 2.5939e-02,  2.9583e-01,  1.2316e-02,
-2.2854e-02, -2.0364e-02, -5.9617e-03,
 3.8680e-40, -2.9783e-40,  4.4142e-40,
 2.4395e-40, -4.7682e-40,  2.8953e-40,
 5.7262e-40,  5.9434e-40,  2.5222e-40,
 2.7904e-03, -2.7069e-01,  2.3093e-02,
 3.9904e-02, -2.6427e-01,  1.0334e-01,
-2.9937e-03, -7.8890e-02, -8.6462e-02,
-1.2483e-01, -6.1148e-03, -3.3632e-02,
-8.5970e-02,  1.6548e-01,  8.9641e-02,
 6.2409e-03,  9.3431e-02,  9.0627e-03,
 7.3069e-41,  1.1883e-40, -3.1731e-40,
 3.7769e-40,  1.6246e-40, -4.5360e-40,
-7.0485e-42, -1.0542e-40, -3.8139e-41,
 1.2848e-01, -2.9901e-01, -7.0971e-02,
 1.3919e-02,  7.0554e-02, -2.0614e-03,
 3.0065e-02, -7.8691e-03,  3.9933e-03,
-5.8318e-02,  6.3462e-01,  2.9796e-02,
-1.2705e-03,  8.3589e-02, -1.6245e-01,
-4.3536e-03,  2.2795e-02,  1.2369e-02,
 4.3751e-03, -1.7238e-02,  6.9507e-03,
-6.1948e-02,  3.0229e-01,  7.6892e-03,
-5.9933e-02,  4.9741e-01,  9.7694e-02,
 2.0260e-02,  4.2249e-01, -8.1227e-02,
 4.5865e-02,  1.0715e-01, -2.8852e-06,
 5.6380e-03, -3.6884e-02, -8.7038e-04,
 1.7224e-16,  2.6130e-16,  1.7024e-16,
 2.8635e-16,  4.2144e-16,  2.6052e-16,
 1.8573e-16,  2.6001e-16,  1.5488e-16,
-4.2236e-03, -6.5587e-03, -3.5398e-02,
-6.1444e-03,  1.3917e-01, -1.5245e-01,
-2.7029e-03,  5.8043e-02, -6.5551e-03,
 3.5543e-02,  2.1427e-02,  6.3287e-03,
 7.3171e-02,  7.5999e-01,  1.3281e-01,
 1.8049e-02,  1.7023e-01,  2.4160e-02,
 1.6663e-05,  1.8981e-05,  1.7523e-05,
 1.6804e-05,  1.9127e-05,  1.7500e-05,
 1.4023e-05,  1.5938e-05,  1.4690e-05,
-1.6227e-01, -3.8794e-01, -1.7112e-01,
 8.9500e-03, -2.0286e-01,  5.1104e-03,
-1.6970e-03, -1.6603e-02, -7.0210e-03,
 3.8030e-02, -1.4732e-02,  9.0098e-02,
 3.6393e-04, -1.0499e-01, -5.5580e-02,
 1.7354e-03,  2.5134e-02, -1.2490e-03,
 3.1317e-03, -5.4939e-03, -4.5612e-03,
 2.9235e-02,  1.2555e-01, -2.5599e-02,
 9.6237e-03, -2.4120e-01,  2.2954e-02,
-3.2960e-02, -1.4256e-01,  2.2131e-02,
-2.3046e-03,  1.8023e-01,  2.0203e-03,
-1.5012e-02, -2.0165e-02, -1.0267e-02,
 5.2856e-40, -2.2257e-41,  4.5131e-40,
-1.4965e-40, -3.0416e-40,  6.6962e-40,
-2.2372e-40,  2.8067e-40, -5.2444e-40,
-5.9000e-03,  9.4155e-02, -1.5616e-02,
-1.5816e-04, -5.8065e-02,  3.9118e-01,
-6.7560e-03,  3.7164e-02, -5.4173e-03,
 1.8661e-02,  1.3765e-02, -1.9254e-04,
 2.8180e-02,  1.0456e-01, -4.5574e-02,
 9.9989e-03, -1.8481e-01,  4.9995e-02,
-2.7433e-40, -6.2360e-40,  3.6480e-40,
 1.9375e-40, -4.1877e-40, -5.8133e-40,
 4.3435e-40,  1.3591e-40, -4.9498e-40,
-7.0876e-02, -6.4387e-02,  3.1325e-02,
-5.4476e-02,  1.5148e-01, -3.5147e-02,
-5.0205e-03,  2.8259e-02, -6.5614e-03,
-1.9345e-04, -1.3521e-01, -8.2248e-02,
 3.0590e-02, -5.7365e-02,  4.7332e-01,
-1.0835e-02, -3.2795e-02, -7.4119e-02,
-2.7713e-02,  9.0443e-03, -2.8045e-02,
 1.1858e-02, -3.7487e-02,  4.2855e-02,
 1.0109e-02,  4.8828e-01, -3.4848e-01,
-5.5123e-02,  9.8343e-02, -1.7121e-02,
 4.3563e-02, -4.8440e-01,  3.8818e-02,
 1.0290e-02,  2.3850e-02, -7.1304e-03,
 8.0256e-40, -8.5564e-40,  1.1754e-39,
-1.3248e-39,  1.9088e-39, -9.5759e-40,
 5.1018e-40, -1.9621e-40, -1.1577e-39,
 3.7657e-03,  2.2415e-01,  8.9807e-03,
 1.1960e-03, -3.5972e-01, -2.2885e-01,
 1.1061e-02,  1.2194e-01,  5.2062e-02,
 5.5375e-02, -5.3896e-02,  2.6862e-02,
-1.1650e-01,  1.3926e-01, -5.5173e-02,
 5.3331e-02, -7.6999e-02,  6.4696e-04,
-5.9882e-40, -3.6667e-40,  3.3672e-40,
-3.8433e-40, -5.0150e-40, -6.3284e-40,
-3.6197e-40, -5.4138e-40,  1.7685e-40,
-2.6306e-03,  4.3950e-02, -1.1801e-01,
-7.5702e-02, -1.1015e-01, -1.1476e-01,
-3.1454e-02,  3.7433e-02, -7.3504e-03,
 2.4517e-02, -2.6038e-01, -4.8733e-02,
 2.2086e-03, -2.4738e-01,  3.6815e-01,
-5.1651e-03, -1.3402e-02, -1.6622e-02,
-1.5334e-02,  2.7074e-02, -4.6296e-03,
 4.8939e-02, -3.6936e-02,  3.3896e-02,
-1.5571e-03, -2.2458e-02, -2.0968e-01,
-2.4934e-01, -1.7596e-01,  1.0758e-01,
 1.4924e-01, -9.9015e-02,  3.4601e-02,
-2.6568e-02,  3.0482e-02, -6.9371e-03,
 1.1089e-40,  5.4700e-40, -4.3619e-40,
 2.9438e-40,  5.5714e-40, -6.3129e-40,
-4.4335e-40, -5.2929e-40,  5.2266e-40,
-1.6793e-02,  6.3459e-02, -9.8753e-03,
 1.4488e-02, -1.3738e-01,  4.3923e-01,
 1.6930e-02, -2.0722e-01, -8.9974e-04,
 3.2808e-02,  6.1693e-03, -3.1921e-03,
-7.2795e-02,  1.3619e-01,  3.3430e-02,
-3.4121e-02, -8.3406e-02, -1.8517e-02,
 4.5702e-33, -1.7326e-34, -3.5785e-33,
 4.2743e-32,  9.4178e-41,  1.7973e-33,
 2.0710e-32, -1.2911e-32,  7.7017e-33,
 3.2100e-02, -1.4806e-01, -1.8026e-02,
 4.2991e-02,  8.2282e-02, -9.3798e-02,
 5.7656e-02, -1.6077e-02,  2.4357e-03,
 1.7558e-02, -2.0692e-01,  9.7487e-02,
-8.0222e-02,  6.1438e-01, -2.7071e-02,
 1.9872e-02, -6.7999e-02, -2.2345e-02,
-2.0132e-02, -5.7227e-04, -2.5003e-02,
 2.6356e-02,  1.3916e-01, -8.1832e-02,
-8.5562e-02,  2.3727e-01,  1.1697e-02,
 4.2532e-02,  2.8059e-01, -1.2267e-02,
-1.1727e-01, -2.7114e-01,  1.6031e-02,
-2.2579e-03,  1.5922e-02,  7.8421e-03,
 1.5655e-40,  5.5273e-40,  5.3953e-40,
 2.1470e-40,  6.8239e-40, -7.6975e-41,
 3.6196e-40,  6.6385e-40,  4.7168e-40,
-1.1468e-03, -6.8809e-02,  6.1474e-02,
 4.1578e-03, -2.2328e-02, -4.4041e-01,
-2.0100e-03,  6.0063e-02,  6.5339e-02,
-2.5969e-02,  2.7496e-02,  7.8974e-03,
-4.6316e-03,  1.7598e-01,  1.9086e-02,
 2.6720e-02, -8.9809e-02, -9.2466e-03,
-4.4547e-42, -1.4439e-40,  4.9048e-40,
 2.2847e-40, -3.1970e-40, -4.5422e-40,
 1.7565e-40,  4.9223e-40, -2.5743e-41,
 7.1777e-02, -1.8382e-01, -4.0780e-02,
-5.2036e-02,  5.2353e-02, -2.9000e-02,
 4.9161e-03, -6.6188e-03,  1.1204e-02,
-3.2626e-03,  1.8573e-01, -4.4906e-02,
 2.1157e-03, -1.1803e-01,  8.3365e-02,
-4.3547e-04, -6.4845e-03, -9.0939e-03,
 8.2440e-04, -5.7086e-03,  1.6389e-03,
-1.1330e-02,  2.2332e-01,  3.3363e-02,
-7.0924e-03,  1.6767e-01, -5.8209e-02,
 8.1546e-02,  4.5576e-01,  2.6518e-03,
 3.9733e-02, -2.5861e-01,  1.5825e-02,
-1.4269e-03,  1.7546e-02,  7.9110e-03
}
};)"
R"(

__constant float biasL[8][8] = 
{
{
 0.0161, -0.6390, -0.0353, -0.0220, -0.0025, -0.0664, -0.0093, -0.0452
}
,
{
-0.0228, -0.0620,  0.0047, -0.0028, -0.0352,  0.0115,  0.0453, -0.0033
}
,
{
-7.8185e-03, -1.6162e-01,  6.2861e-02, -1.1911e-02, -7.8992e-04, 3.3446e-03, -1.0735e-02, -2.5413e-17
}
,
{
-0.0140,  0.0178, -0.0135,  0.0084, -0.0013, -0.0365, -0.0030,  0.0015
}
,
{
-0.0051, -0.0170, -0.0016, -0.0030,  0.1187,  0.0215, -0.0151,  0.7308
}
,
{
 7.5837e-02,  5.9498e-01, -1.0933e-02, -2.6867e-39, -8.5424e-03, -1.5711e-07, -4.9395e-39,  1.0379e-01
}
,
{
-2.5878e-11, -1.8090e-02, -1.2424e-02, -3.7977e-12,  6.0766e-02, -1.1337e-02, -9.4532e-03, -1.2313e-02
}
,
{
 0.0002,  0.0223,  0.0130,  0.1437, -0.0021, -0.0048, -0.0037,  0.0147
}
};

__constant float kernelsL10[4 * 8] = 
{
 0.3337, -0.0603,
-0.0803, -0.1182,
-0.0485, -0.2738,
 0.3074,  0.0302,
-0.2632,  0.0726,
-0.0010,  0.2033,
 0.3165,  0.2832,
 0.2232,  0.2722,
 0.2154,  0.0280,
 0.2199, -0.4580,
-0.1922,  0.0238,
 0.0128,  0.2118,
 0.1852,  0.1089,
-0.4328,  0.1301,
-0.1029,  0.2610,
 0.0579, -0.1900
};)"
R"(


__kernel void conv1To8(
    __read_only image2d_t srcImg, 
    __write_only image2d_t tmpImgOut1, 
    __write_only image2d_t tmpImgOut2)
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

    float4 c1234 = RELU((float4)(
        tl.x * kernelsL1[0*9+0] + tc.x * kernelsL1[0*9+1] + tr.x * kernelsL1[0*9+2] +
        ml.x * kernelsL1[0*9+3] + mc.x * kernelsL1[0*9+4] + mr.x * kernelsL1[0*9+5] +
        bl.x * kernelsL1[0*9+6] + bc.x * kernelsL1[0*9+7] + br.x * kernelsL1[0*9+8] + biasL1[0],

        tl.x * kernelsL1[1*9+0] + tc.x * kernelsL1[1*9+1] + tr.x * kernelsL1[1*9+2] +
        ml.x * kernelsL1[1*9+3] + mc.x * kernelsL1[1*9+4] + mr.x * kernelsL1[1*9+5] +
        bl.x * kernelsL1[1*9+6] + bc.x * kernelsL1[1*9+7] + br.x * kernelsL1[1*9+8] + biasL1[1],

        tl.x * kernelsL1[2*9+0] + tc.x * kernelsL1[2*9+1] + tr.x * kernelsL1[2*9+2] +
        ml.x * kernelsL1[2*9+3] + mc.x * kernelsL1[2*9+4] + mr.x * kernelsL1[2*9+5] +
        bl.x * kernelsL1[2*9+6] + bc.x * kernelsL1[2*9+7] + br.x * kernelsL1[2*9+8] + biasL1[2],

        tl.x * kernelsL1[3*9+0] + tc.x * kernelsL1[3*9+1] + tr.x * kernelsL1[3*9+2] +
        ml.x * kernelsL1[3*9+3] + mc.x * kernelsL1[3*9+4] + mr.x * kernelsL1[3*9+5] +
        bl.x * kernelsL1[3*9+6] + bc.x * kernelsL1[3*9+7] + br.x * kernelsL1[3*9+8] + biasL1[3]
    ));
    float4 c5678 = RELU((float4)(
        tl.x * kernelsL1[4*9+0] + tc.x * kernelsL1[4*9+1] + tr.x * kernelsL1[4*9+2] +
        ml.x * kernelsL1[4*9+3] + mc.x * kernelsL1[4*9+4] + mr.x * kernelsL1[4*9+5] +
        bl.x * kernelsL1[4*9+6] + bc.x * kernelsL1[4*9+7] + br.x * kernelsL1[4*9+8] + biasL1[4],

        tl.x * kernelsL1[5*9+0] + tc.x * kernelsL1[5*9+1] + tr.x * kernelsL1[5*9+2] +
        ml.x * kernelsL1[5*9+3] + mc.x * kernelsL1[5*9+4] + mr.x * kernelsL1[5*9+5] +
        bl.x * kernelsL1[5*9+6] + bc.x * kernelsL1[5*9+7] + br.x * kernelsL1[5*9+8] + biasL1[5],

        tl.x * kernelsL1[6*9+0] + tc.x * kernelsL1[6*9+1] + tr.x * kernelsL1[6*9+2] +
        ml.x * kernelsL1[6*9+3] + mc.x * kernelsL1[6*9+4] + mr.x * kernelsL1[6*9+5] +
        bl.x * kernelsL1[6*9+6] + bc.x * kernelsL1[6*9+7] + br.x * kernelsL1[6*9+8] + biasL1[6],

        tl.x * kernelsL1[7*9+0] + tc.x * kernelsL1[7*9+1] + tr.x * kernelsL1[7*9+2] +
        ml.x * kernelsL1[7*9+3] + mc.x * kernelsL1[7*9+4] + mr.x * kernelsL1[7*9+5] +
        bl.x * kernelsL1[7*9+6] + bc.x * kernelsL1[7*9+7] + br.x * kernelsL1[7*9+8] + biasL1[7]
    ));

    write_imagef(tmpImgOut1, coord, c1234);
    write_imagef(tmpImgOut2, coord, c5678);
})"
R"(

__kernel void conv8To8(
    __read_only image2d_t tmpImgIn1,
    __read_only image2d_t tmpImgIn2, 
    __write_only image2d_t tmpImgOut1, 
    __write_only image2d_t tmpImgOut2,
    int l)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(tmpImgIn1) || y >= get_image_height(tmpImgIn1))
        return;

    int2 coord = (int2)(x, y);

    float4 tl1 = read_imagef(tmpImgIn1, samplerN, (int2)(x-1,y-1));
    float4 tc1 = read_imagef(tmpImgIn1, samplerN, (int2)(x,y-1));
    float4 tr1 = read_imagef(tmpImgIn1, samplerN, (int2)(x+1,y-1));
    float4 ml1 = read_imagef(tmpImgIn1, samplerN, (int2)(x-1,y));
    float4 mc1 = read_imagef(tmpImgIn1, samplerN, coord);
    float4 mr1 = read_imagef(tmpImgIn1, samplerN, (int2)(x+1,y));
    float4 bl1 = read_imagef(tmpImgIn1, samplerN, (int2)(x-1,y+1));
    float4 bc1 = read_imagef(tmpImgIn1, samplerN, (int2)(x,y+1));
    float4 br1 = read_imagef(tmpImgIn1, samplerN, (int2)(x+1,y+1));

    float4 tl2 = read_imagef(tmpImgIn2, samplerN, (int2)(x-1,y-1));
    float4 tc2 = read_imagef(tmpImgIn2, samplerN, (int2)(x,y-1));
    float4 tr2 = read_imagef(tmpImgIn2, samplerN, (int2)(x+1,y-1));
    float4 ml2 = read_imagef(tmpImgIn2, samplerN, (int2)(x-1,y));
    float4 mc2 = read_imagef(tmpImgIn2, samplerN, coord);
    float4 mr2 = read_imagef(tmpImgIn2, samplerN, (int2)(x+1,y));
    float4 bl2 = read_imagef(tmpImgIn2, samplerN, (int2)(x-1,y+1));
    float4 bc2 = read_imagef(tmpImgIn2, samplerN, (int2)(x,y+1));
    float4 br2 = read_imagef(tmpImgIn2, samplerN, (int2)(x+1,y+1));
    
    float4 c1234 = RELU((float4)(
        tl1.x * kernelsL[l][0*72+0*9+0] + tc1.x * kernelsL[l][0*72+0*9+1] + tr1.x * kernelsL[l][0*72+0*9+2] +
        ml1.x * kernelsL[l][0*72+0*9+3] + mc1.x * kernelsL[l][0*72+0*9+4] + mr1.x * kernelsL[l][0*72+0*9+5] +
        bl1.x * kernelsL[l][0*72+0*9+6] + bc1.x * kernelsL[l][0*72+0*9+7] + br1.x * kernelsL[l][0*72+0*9+8] + 

        tl1.y * kernelsL[l][0*72+1*9+0] + tc1.y * kernelsL[l][0*72+1*9+1] + tr1.y * kernelsL[l][0*72+1*9+2] +
        ml1.y * kernelsL[l][0*72+1*9+3] + mc1.y * kernelsL[l][0*72+1*9+4] + mr1.y * kernelsL[l][0*72+1*9+5] +
        bl1.y * kernelsL[l][0*72+1*9+6] + bc1.y * kernelsL[l][0*72+1*9+7] + br1.y * kernelsL[l][0*72+1*9+8] + 

        tl1.z * kernelsL[l][0*72+2*9+0] + tc1.z * kernelsL[l][0*72+2*9+1] + tr1.z * kernelsL[l][0*72+2*9+2] +
        ml1.z * kernelsL[l][0*72+2*9+3] + mc1.z * kernelsL[l][0*72+2*9+4] + mr1.z * kernelsL[l][0*72+2*9+5] +
        bl1.z * kernelsL[l][0*72+2*9+6] + bc1.z * kernelsL[l][0*72+2*9+7] + br1.z * kernelsL[l][0*72+2*9+8] + 

        tl1.w * kernelsL[l][0*72+3*9+0] + tc1.w * kernelsL[l][0*72+3*9+1] + tr1.w * kernelsL[l][0*72+3*9+2] +
        ml1.w * kernelsL[l][0*72+3*9+3] + mc1.w * kernelsL[l][0*72+3*9+4] + mr1.w * kernelsL[l][0*72+3*9+5] +
        bl1.w * kernelsL[l][0*72+3*9+6] + bc1.w * kernelsL[l][0*72+3*9+7] + br1.w * kernelsL[l][0*72+3*9+8] +

        tl2.x * kernelsL[l][0*72+4*9+0] + tc2.x * kernelsL[l][0*72+4*9+1] + tr2.x * kernelsL[l][0*72+4*9+2] +
        ml2.x * kernelsL[l][0*72+4*9+3] + mc2.x * kernelsL[l][0*72+4*9+4] + mr2.x * kernelsL[l][0*72+4*9+5] +
        bl2.x * kernelsL[l][0*72+4*9+6] + bc2.x * kernelsL[l][0*72+4*9+7] + br2.x * kernelsL[l][0*72+4*9+8] + 

        tl2.y * kernelsL[l][0*72+5*9+0] + tc2.y * kernelsL[l][0*72+5*9+1] + tr2.y * kernelsL[l][0*72+5*9+2] +
        ml2.y * kernelsL[l][0*72+5*9+3] + mc2.y * kernelsL[l][0*72+5*9+4] + mr2.y * kernelsL[l][0*72+5*9+5] +
        bl2.y * kernelsL[l][0*72+5*9+6] + bc2.y * kernelsL[l][0*72+5*9+7] + br2.y * kernelsL[l][0*72+5*9+8] + 

        tl2.z * kernelsL[l][0*72+6*9+0] + tc2.z * kernelsL[l][0*72+6*9+1] + tr2.z * kernelsL[l][0*72+6*9+2] +
        ml2.z * kernelsL[l][0*72+6*9+3] + mc2.z * kernelsL[l][0*72+6*9+4] + mr2.z * kernelsL[l][0*72+6*9+5] +
        bl2.z * kernelsL[l][0*72+6*9+6] + bc2.z * kernelsL[l][0*72+6*9+7] + br2.z * kernelsL[l][0*72+6*9+8] + 

        tl2.w * kernelsL[l][0*72+7*9+0] + tc2.w * kernelsL[l][0*72+7*9+1] + tr2.w * kernelsL[l][0*72+7*9+2] +
        ml2.w * kernelsL[l][0*72+7*9+3] + mc2.w * kernelsL[l][0*72+7*9+4] + mr2.w * kernelsL[l][0*72+7*9+5] +
        bl2.w * kernelsL[l][0*72+7*9+6] + bc2.w * kernelsL[l][0*72+7*9+7] + br2.w * kernelsL[l][0*72+7*9+8] + biasL[l][0]
        ,
        tl1.x * kernelsL[l][1*72+0*9+0] + tc1.x * kernelsL[l][1*72+0*9+1] + tr1.x * kernelsL[l][1*72+0*9+2] +
        ml1.x * kernelsL[l][1*72+0*9+3] + mc1.x * kernelsL[l][1*72+0*9+4] + mr1.x * kernelsL[l][1*72+0*9+5] +
        bl1.x * kernelsL[l][1*72+0*9+6] + bc1.x * kernelsL[l][1*72+0*9+7] + br1.x * kernelsL[l][1*72+0*9+8] + 

        tl1.y * kernelsL[l][1*72+1*9+0] + tc1.y * kernelsL[l][1*72+1*9+1] + tr1.y * kernelsL[l][1*72+1*9+2] +
        ml1.y * kernelsL[l][1*72+1*9+3] + mc1.y * kernelsL[l][1*72+1*9+4] + mr1.y * kernelsL[l][1*72+1*9+5] +
        bl1.y * kernelsL[l][1*72+1*9+6] + bc1.y * kernelsL[l][1*72+1*9+7] + br1.y * kernelsL[l][1*72+1*9+8] + 

        tl1.z * kernelsL[l][1*72+2*9+0] + tc1.z * kernelsL[l][1*72+2*9+1] + tr1.z * kernelsL[l][1*72+2*9+2] +
        ml1.z * kernelsL[l][1*72+2*9+3] + mc1.z * kernelsL[l][1*72+2*9+4] + mr1.z * kernelsL[l][1*72+2*9+5] +
        bl1.z * kernelsL[l][1*72+2*9+6] + bc1.z * kernelsL[l][1*72+2*9+7] + br1.z * kernelsL[l][1*72+2*9+8] + 

        tl1.w * kernelsL[l][1*72+3*9+0] + tc1.w * kernelsL[l][1*72+3*9+1] + tr1.w * kernelsL[l][1*72+3*9+2] +
        ml1.w * kernelsL[l][1*72+3*9+3] + mc1.w * kernelsL[l][1*72+3*9+4] + mr1.w * kernelsL[l][1*72+3*9+5] +
        bl1.w * kernelsL[l][1*72+3*9+6] + bc1.w * kernelsL[l][1*72+3*9+7] + br1.w * kernelsL[l][1*72+3*9+8] +

        tl2.x * kernelsL[l][1*72+4*9+0] + tc2.x * kernelsL[l][1*72+4*9+1] + tr2.x * kernelsL[l][1*72+4*9+2] +
        ml2.x * kernelsL[l][1*72+4*9+3] + mc2.x * kernelsL[l][1*72+4*9+4] + mr2.x * kernelsL[l][1*72+4*9+5] +
        bl2.x * kernelsL[l][1*72+4*9+6] + bc2.x * kernelsL[l][1*72+4*9+7] + br2.x * kernelsL[l][1*72+4*9+8] + 

        tl2.y * kernelsL[l][1*72+5*9+0] + tc2.y * kernelsL[l][1*72+5*9+1] + tr2.y * kernelsL[l][1*72+5*9+2] +
        ml2.y * kernelsL[l][1*72+5*9+3] + mc2.y * kernelsL[l][1*72+5*9+4] + mr2.y * kernelsL[l][1*72+5*9+5] +
        bl2.y * kernelsL[l][1*72+5*9+6] + bc2.y * kernelsL[l][1*72+5*9+7] + br2.y * kernelsL[l][1*72+5*9+8] + 

        tl2.z * kernelsL[l][1*72+6*9+0] + tc2.z * kernelsL[l][1*72+6*9+1] + tr2.z * kernelsL[l][1*72+6*9+2] +
        ml2.z * kernelsL[l][1*72+6*9+3] + mc2.z * kernelsL[l][1*72+6*9+4] + mr2.z * kernelsL[l][1*72+6*9+5] +
        bl2.z * kernelsL[l][1*72+6*9+6] + bc2.z * kernelsL[l][1*72+6*9+7] + br2.z * kernelsL[l][1*72+6*9+8] + 

        tl2.w * kernelsL[l][1*72+7*9+0] + tc2.w * kernelsL[l][1*72+7*9+1] + tr2.w * kernelsL[l][1*72+7*9+2] +
        ml2.w * kernelsL[l][1*72+7*9+3] + mc2.w * kernelsL[l][1*72+7*9+4] + mr2.w * kernelsL[l][1*72+7*9+5] +
        bl2.w * kernelsL[l][1*72+7*9+6] + bc2.w * kernelsL[l][1*72+7*9+7] + br2.w * kernelsL[l][1*72+7*9+8] + biasL[l][1]
        ,
        tl1.x * kernelsL[l][2*72+0*9+0] + tc1.x * kernelsL[l][2*72+0*9+1] + tr1.x * kernelsL[l][2*72+0*9+2] +
        ml1.x * kernelsL[l][2*72+0*9+3] + mc1.x * kernelsL[l][2*72+0*9+4] + mr1.x * kernelsL[l][2*72+0*9+5] +
        bl1.x * kernelsL[l][2*72+0*9+6] + bc1.x * kernelsL[l][2*72+0*9+7] + br1.x * kernelsL[l][2*72+0*9+8] + 

        tl1.y * kernelsL[l][2*72+1*9+0] + tc1.y * kernelsL[l][2*72+1*9+1] + tr1.y * kernelsL[l][2*72+1*9+2] +
        ml1.y * kernelsL[l][2*72+1*9+3] + mc1.y * kernelsL[l][2*72+1*9+4] + mr1.y * kernelsL[l][2*72+1*9+5] +
        bl1.y * kernelsL[l][2*72+1*9+6] + bc1.y * kernelsL[l][2*72+1*9+7] + br1.y * kernelsL[l][2*72+1*9+8] + 

        tl1.z * kernelsL[l][2*72+2*9+0] + tc1.z * kernelsL[l][2*72+2*9+1] + tr1.z * kernelsL[l][2*72+2*9+2] +
        ml1.z * kernelsL[l][2*72+2*9+3] + mc1.z * kernelsL[l][2*72+2*9+4] + mr1.z * kernelsL[l][2*72+2*9+5] +
        bl1.z * kernelsL[l][2*72+2*9+6] + bc1.z * kernelsL[l][2*72+2*9+7] + br1.z * kernelsL[l][2*72+2*9+8] + 

        tl1.w * kernelsL[l][2*72+3*9+0] + tc1.w * kernelsL[l][2*72+3*9+1] + tr1.w * kernelsL[l][2*72+3*9+2] +
        ml1.w * kernelsL[l][2*72+3*9+3] + mc1.w * kernelsL[l][2*72+3*9+4] + mr1.w * kernelsL[l][2*72+3*9+5] +
        bl1.w * kernelsL[l][2*72+3*9+6] + bc1.w * kernelsL[l][2*72+3*9+7] + br1.w * kernelsL[l][2*72+3*9+8] +

        tl2.x * kernelsL[l][2*72+4*9+0] + tc2.x * kernelsL[l][2*72+4*9+1] + tr2.x * kernelsL[l][2*72+4*9+2] +
        ml2.x * kernelsL[l][2*72+4*9+3] + mc2.x * kernelsL[l][2*72+4*9+4] + mr2.x * kernelsL[l][2*72+4*9+5] +
        bl2.x * kernelsL[l][2*72+4*9+6] + bc2.x * kernelsL[l][2*72+4*9+7] + br2.x * kernelsL[l][2*72+4*9+8] + 

        tl2.y * kernelsL[l][2*72+5*9+0] + tc2.y * kernelsL[l][2*72+5*9+1] + tr2.y * kernelsL[l][2*72+5*9+2] +
        ml2.y * kernelsL[l][2*72+5*9+3] + mc2.y * kernelsL[l][2*72+5*9+4] + mr2.y * kernelsL[l][2*72+5*9+5] +
        bl2.y * kernelsL[l][2*72+5*9+6] + bc2.y * kernelsL[l][2*72+5*9+7] + br2.y * kernelsL[l][2*72+5*9+8] + 

        tl2.z * kernelsL[l][2*72+6*9+0] + tc2.z * kernelsL[l][2*72+6*9+1] + tr2.z * kernelsL[l][2*72+6*9+2] +
        ml2.z * kernelsL[l][2*72+6*9+3] + mc2.z * kernelsL[l][2*72+6*9+4] + mr2.z * kernelsL[l][2*72+6*9+5] +
        bl2.z * kernelsL[l][2*72+6*9+6] + bc2.z * kernelsL[l][2*72+6*9+7] + br2.z * kernelsL[l][2*72+6*9+8] + 

        tl2.w * kernelsL[l][2*72+7*9+0] + tc2.w * kernelsL[l][2*72+7*9+1] + tr2.w * kernelsL[l][2*72+7*9+2] +
        ml2.w * kernelsL[l][2*72+7*9+3] + mc2.w * kernelsL[l][2*72+7*9+4] + mr2.w * kernelsL[l][2*72+7*9+5] +
        bl2.w * kernelsL[l][2*72+7*9+6] + bc2.w * kernelsL[l][2*72+7*9+7] + br2.w * kernelsL[l][2*72+7*9+8] + biasL[l][2]
        ,
        tl1.x * kernelsL[l][3*72+0*9+0] + tc1.x * kernelsL[l][3*72+0*9+1] + tr1.x * kernelsL[l][3*72+0*9+2] +
        ml1.x * kernelsL[l][3*72+0*9+3] + mc1.x * kernelsL[l][3*72+0*9+4] + mr1.x * kernelsL[l][3*72+0*9+5] +
        bl1.x * kernelsL[l][3*72+0*9+6] + bc1.x * kernelsL[l][3*72+0*9+7] + br1.x * kernelsL[l][3*72+0*9+8] + 

        tl1.y * kernelsL[l][3*72+1*9+0] + tc1.y * kernelsL[l][3*72+1*9+1] + tr1.y * kernelsL[l][3*72+1*9+2] +
        ml1.y * kernelsL[l][3*72+1*9+3] + mc1.y * kernelsL[l][3*72+1*9+4] + mr1.y * kernelsL[l][3*72+1*9+5] +
        bl1.y * kernelsL[l][3*72+1*9+6] + bc1.y * kernelsL[l][3*72+1*9+7] + br1.y * kernelsL[l][3*72+1*9+8] + 

        tl1.z * kernelsL[l][3*72+2*9+0] + tc1.z * kernelsL[l][3*72+2*9+1] + tr1.z * kernelsL[l][3*72+2*9+2] +
        ml1.z * kernelsL[l][3*72+2*9+3] + mc1.z * kernelsL[l][3*72+2*9+4] + mr1.z * kernelsL[l][3*72+2*9+5] +
        bl1.z * kernelsL[l][3*72+2*9+6] + bc1.z * kernelsL[l][3*72+2*9+7] + br1.z * kernelsL[l][3*72+2*9+8] + 

        tl1.w * kernelsL[l][3*72+3*9+0] + tc1.w * kernelsL[l][3*72+3*9+1] + tr1.w * kernelsL[l][3*72+3*9+2] +
        ml1.w * kernelsL[l][3*72+3*9+3] + mc1.w * kernelsL[l][3*72+3*9+4] + mr1.w * kernelsL[l][3*72+3*9+5] +
        bl1.w * kernelsL[l][3*72+3*9+6] + bc1.w * kernelsL[l][3*72+3*9+7] + br1.w * kernelsL[l][3*72+3*9+8] +

        tl2.x * kernelsL[l][3*72+4*9+0] + tc2.x * kernelsL[l][3*72+4*9+1] + tr2.x * kernelsL[l][3*72+4*9+2] +
        ml2.x * kernelsL[l][3*72+4*9+3] + mc2.x * kernelsL[l][3*72+4*9+4] + mr2.x * kernelsL[l][3*72+4*9+5] +
        bl2.x * kernelsL[l][3*72+4*9+6] + bc2.x * kernelsL[l][3*72+4*9+7] + br2.x * kernelsL[l][3*72+4*9+8] + 

        tl2.y * kernelsL[l][3*72+5*9+0] + tc2.y * kernelsL[l][3*72+5*9+1] + tr2.y * kernelsL[l][3*72+5*9+2] +
        ml2.y * kernelsL[l][3*72+5*9+3] + mc2.y * kernelsL[l][3*72+5*9+4] + mr2.y * kernelsL[l][3*72+5*9+5] +
        bl2.y * kernelsL[l][3*72+5*9+6] + bc2.y * kernelsL[l][3*72+5*9+7] + br2.y * kernelsL[l][3*72+5*9+8] + 

        tl2.z * kernelsL[l][3*72+6*9+0] + tc2.z * kernelsL[l][3*72+6*9+1] + tr2.z * kernelsL[l][3*72+6*9+2] +
        ml2.z * kernelsL[l][3*72+6*9+3] + mc2.z * kernelsL[l][3*72+6*9+4] + mr2.z * kernelsL[l][3*72+6*9+5] +
        bl2.z * kernelsL[l][3*72+6*9+6] + bc2.z * kernelsL[l][3*72+6*9+7] + br2.z * kernelsL[l][3*72+6*9+8] + 

        tl2.w * kernelsL[l][3*72+7*9+0] + tc2.w * kernelsL[l][3*72+7*9+1] + tr2.w * kernelsL[l][3*72+7*9+2] +
        ml2.w * kernelsL[l][3*72+7*9+3] + mc2.w * kernelsL[l][3*72+7*9+4] + mr2.w * kernelsL[l][3*72+7*9+5] +
        bl2.w * kernelsL[l][3*72+7*9+6] + bc2.w * kernelsL[l][3*72+7*9+7] + br2.w * kernelsL[l][3*72+7*9+8] + biasL[l][3]
    ));)"
    R"(
    float4 c5678 = RELU((float4)(
        tl1.x * kernelsL[l][4*72+0*9+0] + tc1.x * kernelsL[l][4*72+0*9+1] + tr1.x * kernelsL[l][4*72+0*9+2] +
        ml1.x * kernelsL[l][4*72+0*9+3] + mc1.x * kernelsL[l][4*72+0*9+4] + mr1.x * kernelsL[l][4*72+0*9+5] +
        bl1.x * kernelsL[l][4*72+0*9+6] + bc1.x * kernelsL[l][4*72+0*9+7] + br1.x * kernelsL[l][4*72+0*9+8] + 

        tl1.y * kernelsL[l][4*72+1*9+0] + tc1.y * kernelsL[l][4*72+1*9+1] + tr1.y * kernelsL[l][4*72+1*9+2] +
        ml1.y * kernelsL[l][4*72+1*9+3] + mc1.y * kernelsL[l][4*72+1*9+4] + mr1.y * kernelsL[l][4*72+1*9+5] +
        bl1.y * kernelsL[l][4*72+1*9+6] + bc1.y * kernelsL[l][4*72+1*9+7] + br1.y * kernelsL[l][4*72+1*9+8] + 

        tl1.z * kernelsL[l][4*72+2*9+0] + tc1.z * kernelsL[l][4*72+2*9+1] + tr1.z * kernelsL[l][4*72+2*9+2] +
        ml1.z * kernelsL[l][4*72+2*9+3] + mc1.z * kernelsL[l][4*72+2*9+4] + mr1.z * kernelsL[l][4*72+2*9+5] +
        bl1.z * kernelsL[l][4*72+2*9+6] + bc1.z * kernelsL[l][4*72+2*9+7] + br1.z * kernelsL[l][4*72+2*9+8] + 

        tl1.w * kernelsL[l][4*72+3*9+0] + tc1.w * kernelsL[l][4*72+3*9+1] + tr1.w * kernelsL[l][4*72+3*9+2] +
        ml1.w * kernelsL[l][4*72+3*9+3] + mc1.w * kernelsL[l][4*72+3*9+4] + mr1.w * kernelsL[l][4*72+3*9+5] +
        bl1.w * kernelsL[l][4*72+3*9+6] + bc1.w * kernelsL[l][4*72+3*9+7] + br1.w * kernelsL[l][4*72+3*9+8] +

        tl2.x * kernelsL[l][4*72+4*9+0] + tc2.x * kernelsL[l][4*72+4*9+1] + tr2.x * kernelsL[l][4*72+4*9+2] +
        ml2.x * kernelsL[l][4*72+4*9+3] + mc2.x * kernelsL[l][4*72+4*9+4] + mr2.x * kernelsL[l][4*72+4*9+5] +
        bl2.x * kernelsL[l][4*72+4*9+6] + bc2.x * kernelsL[l][4*72+4*9+7] + br2.x * kernelsL[l][4*72+4*9+8] + 

        tl2.y * kernelsL[l][4*72+5*9+0] + tc2.y * kernelsL[l][4*72+5*9+1] + tr2.y * kernelsL[l][4*72+5*9+2] +
        ml2.y * kernelsL[l][4*72+5*9+3] + mc2.y * kernelsL[l][4*72+5*9+4] + mr2.y * kernelsL[l][4*72+5*9+5] +
        bl2.y * kernelsL[l][4*72+5*9+6] + bc2.y * kernelsL[l][4*72+5*9+7] + br2.y * kernelsL[l][4*72+5*9+8] + 

        tl2.z * kernelsL[l][4*72+6*9+0] + tc2.z * kernelsL[l][4*72+6*9+1] + tr2.z * kernelsL[l][4*72+6*9+2] +
        ml2.z * kernelsL[l][4*72+6*9+3] + mc2.z * kernelsL[l][4*72+6*9+4] + mr2.z * kernelsL[l][4*72+6*9+5] +
        bl2.z * kernelsL[l][4*72+6*9+6] + bc2.z * kernelsL[l][4*72+6*9+7] + br2.z * kernelsL[l][4*72+6*9+8] + 

        tl2.w * kernelsL[l][4*72+7*9+0] + tc2.w * kernelsL[l][4*72+7*9+1] + tr2.w * kernelsL[l][4*72+7*9+2] +
        ml2.w * kernelsL[l][4*72+7*9+3] + mc2.w * kernelsL[l][4*72+7*9+4] + mr2.w * kernelsL[l][4*72+7*9+5] +
        bl2.w * kernelsL[l][4*72+7*9+6] + bc2.w * kernelsL[l][4*72+7*9+7] + br2.w * kernelsL[l][4*72+7*9+8] + biasL[l][4]
        ,
        tl1.x * kernelsL[l][5*72+0*9+0] + tc1.x * kernelsL[l][5*72+0*9+1] + tr1.x * kernelsL[l][5*72+0*9+2] +
        ml1.x * kernelsL[l][5*72+0*9+3] + mc1.x * kernelsL[l][5*72+0*9+4] + mr1.x * kernelsL[l][5*72+0*9+5] +
        bl1.x * kernelsL[l][5*72+0*9+6] + bc1.x * kernelsL[l][5*72+0*9+7] + br1.x * kernelsL[l][5*72+0*9+8] + 

        tl1.y * kernelsL[l][5*72+1*9+0] + tc1.y * kernelsL[l][5*72+1*9+1] + tr1.y * kernelsL[l][5*72+1*9+2] +
        ml1.y * kernelsL[l][5*72+1*9+3] + mc1.y * kernelsL[l][5*72+1*9+4] + mr1.y * kernelsL[l][5*72+1*9+5] +
        bl1.y * kernelsL[l][5*72+1*9+6] + bc1.y * kernelsL[l][5*72+1*9+7] + br1.y * kernelsL[l][5*72+1*9+8] + 

        tl1.z * kernelsL[l][5*72+2*9+0] + tc1.z * kernelsL[l][5*72+2*9+1] + tr1.z * kernelsL[l][5*72+2*9+2] +
        ml1.z * kernelsL[l][5*72+2*9+3] + mc1.z * kernelsL[l][5*72+2*9+4] + mr1.z * kernelsL[l][5*72+2*9+5] +
        bl1.z * kernelsL[l][5*72+2*9+6] + bc1.z * kernelsL[l][5*72+2*9+7] + br1.z * kernelsL[l][5*72+2*9+8] + 

        tl1.w * kernelsL[l][5*72+3*9+0] + tc1.w * kernelsL[l][5*72+3*9+1] + tr1.w * kernelsL[l][5*72+3*9+2] +
        ml1.w * kernelsL[l][5*72+3*9+3] + mc1.w * kernelsL[l][5*72+3*9+4] + mr1.w * kernelsL[l][5*72+3*9+5] +
        bl1.w * kernelsL[l][5*72+3*9+6] + bc1.w * kernelsL[l][5*72+3*9+7] + br1.w * kernelsL[l][5*72+3*9+8] +

        tl2.x * kernelsL[l][5*72+4*9+0] + tc2.x * kernelsL[l][5*72+4*9+1] + tr2.x * kernelsL[l][5*72+4*9+2] +
        ml2.x * kernelsL[l][5*72+4*9+3] + mc2.x * kernelsL[l][5*72+4*9+4] + mr2.x * kernelsL[l][5*72+4*9+5] +
        bl2.x * kernelsL[l][5*72+4*9+6] + bc2.x * kernelsL[l][5*72+4*9+7] + br2.x * kernelsL[l][5*72+4*9+8] + 

        tl2.y * kernelsL[l][5*72+5*9+0] + tc2.y * kernelsL[l][5*72+5*9+1] + tr2.y * kernelsL[l][5*72+5*9+2] +
        ml2.y * kernelsL[l][5*72+5*9+3] + mc2.y * kernelsL[l][5*72+5*9+4] + mr2.y * kernelsL[l][5*72+5*9+5] +
        bl2.y * kernelsL[l][5*72+5*9+6] + bc2.y * kernelsL[l][5*72+5*9+7] + br2.y * kernelsL[l][5*72+5*9+8] + 

        tl2.z * kernelsL[l][5*72+6*9+0] + tc2.z * kernelsL[l][5*72+6*9+1] + tr2.z * kernelsL[l][5*72+6*9+2] +
        ml2.z * kernelsL[l][5*72+6*9+3] + mc2.z * kernelsL[l][5*72+6*9+4] + mr2.z * kernelsL[l][5*72+6*9+5] +
        bl2.z * kernelsL[l][5*72+6*9+6] + bc2.z * kernelsL[l][5*72+6*9+7] + br2.z * kernelsL[l][5*72+6*9+8] + 

        tl2.w * kernelsL[l][5*72+7*9+0] + tc2.w * kernelsL[l][5*72+7*9+1] + tr2.w * kernelsL[l][5*72+7*9+2] +
        ml2.w * kernelsL[l][5*72+7*9+3] + mc2.w * kernelsL[l][5*72+7*9+4] + mr2.w * kernelsL[l][5*72+7*9+5] +
        bl2.w * kernelsL[l][5*72+7*9+6] + bc2.w * kernelsL[l][5*72+7*9+7] + br2.w * kernelsL[l][5*72+7*9+8] + biasL[l][5]
        ,
        tl1.x * kernelsL[l][6*72+0*9+0] + tc1.x * kernelsL[l][6*72+0*9+1] + tr1.x * kernelsL[l][6*72+0*9+2] +
        ml1.x * kernelsL[l][6*72+0*9+3] + mc1.x * kernelsL[l][6*72+0*9+4] + mr1.x * kernelsL[l][6*72+0*9+5] +
        bl1.x * kernelsL[l][6*72+0*9+6] + bc1.x * kernelsL[l][6*72+0*9+7] + br1.x * kernelsL[l][6*72+0*9+8] + 

        tl1.y * kernelsL[l][6*72+1*9+0] + tc1.y * kernelsL[l][6*72+1*9+1] + tr1.y * kernelsL[l][6*72+1*9+2] +
        ml1.y * kernelsL[l][6*72+1*9+3] + mc1.y * kernelsL[l][6*72+1*9+4] + mr1.y * kernelsL[l][6*72+1*9+5] +
        bl1.y * kernelsL[l][6*72+1*9+6] + bc1.y * kernelsL[l][6*72+1*9+7] + br1.y * kernelsL[l][6*72+1*9+8] + 

        tl1.z * kernelsL[l][6*72+2*9+0] + tc1.z * kernelsL[l][6*72+2*9+1] + tr1.z * kernelsL[l][6*72+2*9+2] +
        ml1.z * kernelsL[l][6*72+2*9+3] + mc1.z * kernelsL[l][6*72+2*9+4] + mr1.z * kernelsL[l][6*72+2*9+5] +
        bl1.z * kernelsL[l][6*72+2*9+6] + bc1.z * kernelsL[l][6*72+2*9+7] + br1.z * kernelsL[l][6*72+2*9+8] + 

        tl1.w * kernelsL[l][6*72+3*9+0] + tc1.w * kernelsL[l][6*72+3*9+1] + tr1.w * kernelsL[l][6*72+3*9+2] +
        ml1.w * kernelsL[l][6*72+3*9+3] + mc1.w * kernelsL[l][6*72+3*9+4] + mr1.w * kernelsL[l][6*72+3*9+5] +
        bl1.w * kernelsL[l][6*72+3*9+6] + bc1.w * kernelsL[l][6*72+3*9+7] + br1.w * kernelsL[l][6*72+3*9+8] +

        tl2.x * kernelsL[l][6*72+4*9+0] + tc2.x * kernelsL[l][6*72+4*9+1] + tr2.x * kernelsL[l][6*72+4*9+2] +
        ml2.x * kernelsL[l][6*72+4*9+3] + mc2.x * kernelsL[l][6*72+4*9+4] + mr2.x * kernelsL[l][6*72+4*9+5] +
        bl2.x * kernelsL[l][6*72+4*9+6] + bc2.x * kernelsL[l][6*72+4*9+7] + br2.x * kernelsL[l][6*72+4*9+8] + 

        tl2.y * kernelsL[l][6*72+5*9+0] + tc2.y * kernelsL[l][6*72+5*9+1] + tr2.y * kernelsL[l][6*72+5*9+2] +
        ml2.y * kernelsL[l][6*72+5*9+3] + mc2.y * kernelsL[l][6*72+5*9+4] + mr2.y * kernelsL[l][6*72+5*9+5] +
        bl2.y * kernelsL[l][6*72+5*9+6] + bc2.y * kernelsL[l][6*72+5*9+7] + br2.y * kernelsL[l][6*72+5*9+8] + 

        tl2.z * kernelsL[l][6*72+6*9+0] + tc2.z * kernelsL[l][6*72+6*9+1] + tr2.z * kernelsL[l][6*72+6*9+2] +
        ml2.z * kernelsL[l][6*72+6*9+3] + mc2.z * kernelsL[l][6*72+6*9+4] + mr2.z * kernelsL[l][6*72+6*9+5] +
        bl2.z * kernelsL[l][6*72+6*9+6] + bc2.z * kernelsL[l][6*72+6*9+7] + br2.z * kernelsL[l][6*72+6*9+8] + 

        tl2.w * kernelsL[l][6*72+7*9+0] + tc2.w * kernelsL[l][6*72+7*9+1] + tr2.w * kernelsL[l][6*72+7*9+2] +
        ml2.w * kernelsL[l][6*72+7*9+3] + mc2.w * kernelsL[l][6*72+7*9+4] + mr2.w * kernelsL[l][6*72+7*9+5] +
        bl2.w * kernelsL[l][6*72+7*9+6] + bc2.w * kernelsL[l][6*72+7*9+7] + br2.w * kernelsL[l][6*72+7*9+8] + biasL[l][6]
        ,
        tl1.x * kernelsL[l][7*72+0*9+0] + tc1.x * kernelsL[l][7*72+0*9+1] + tr1.x * kernelsL[l][7*72+0*9+2] +
        ml1.x * kernelsL[l][7*72+0*9+3] + mc1.x * kernelsL[l][7*72+0*9+4] + mr1.x * kernelsL[l][7*72+0*9+5] +
        bl1.x * kernelsL[l][7*72+0*9+6] + bc1.x * kernelsL[l][7*72+0*9+7] + br1.x * kernelsL[l][7*72+0*9+8] + 

        tl1.y * kernelsL[l][7*72+1*9+0] + tc1.y * kernelsL[l][7*72+1*9+1] + tr1.y * kernelsL[l][7*72+1*9+2] +
        ml1.y * kernelsL[l][7*72+1*9+3] + mc1.y * kernelsL[l][7*72+1*9+4] + mr1.y * kernelsL[l][7*72+1*9+5] +
        bl1.y * kernelsL[l][7*72+1*9+6] + bc1.y * kernelsL[l][7*72+1*9+7] + br1.y * kernelsL[l][7*72+1*9+8] + 

        tl1.z * kernelsL[l][7*72+2*9+0] + tc1.z * kernelsL[l][7*72+2*9+1] + tr1.z * kernelsL[l][7*72+2*9+2] +
        ml1.z * kernelsL[l][7*72+2*9+3] + mc1.z * kernelsL[l][7*72+2*9+4] + mr1.z * kernelsL[l][7*72+2*9+5] +
        bl1.z * kernelsL[l][7*72+2*9+6] + bc1.z * kernelsL[l][7*72+2*9+7] + br1.z * kernelsL[l][7*72+2*9+8] + 

        tl1.w * kernelsL[l][7*72+3*9+0] + tc1.w * kernelsL[l][7*72+3*9+1] + tr1.w * kernelsL[l][7*72+3*9+2] +
        ml1.w * kernelsL[l][7*72+3*9+3] + mc1.w * kernelsL[l][7*72+3*9+4] + mr1.w * kernelsL[l][7*72+3*9+5] +
        bl1.w * kernelsL[l][7*72+3*9+6] + bc1.w * kernelsL[l][7*72+3*9+7] + br1.w * kernelsL[l][7*72+3*9+8] +

        tl2.x * kernelsL[l][7*72+4*9+0] + tc2.x * kernelsL[l][7*72+4*9+1] + tr2.x * kernelsL[l][7*72+4*9+2] +
        ml2.x * kernelsL[l][7*72+4*9+3] + mc2.x * kernelsL[l][7*72+4*9+4] + mr2.x * kernelsL[l][7*72+4*9+5] +
        bl2.x * kernelsL[l][7*72+4*9+6] + bc2.x * kernelsL[l][7*72+4*9+7] + br2.x * kernelsL[l][7*72+4*9+8] + 

        tl2.y * kernelsL[l][7*72+5*9+0] + tc2.y * kernelsL[l][7*72+5*9+1] + tr2.y * kernelsL[l][7*72+5*9+2] +
        ml2.y * kernelsL[l][7*72+5*9+3] + mc2.y * kernelsL[l][7*72+5*9+4] + mr2.y * kernelsL[l][7*72+5*9+5] +
        bl2.y * kernelsL[l][7*72+5*9+6] + bc2.y * kernelsL[l][7*72+5*9+7] + br2.y * kernelsL[l][7*72+5*9+8] + 

        tl2.z * kernelsL[l][7*72+6*9+0] + tc2.z * kernelsL[l][7*72+6*9+1] + tr2.z * kernelsL[l][7*72+6*9+2] +
        ml2.z * kernelsL[l][7*72+6*9+3] + mc2.z * kernelsL[l][7*72+6*9+4] + mr2.z * kernelsL[l][7*72+6*9+5] +
        bl2.z * kernelsL[l][7*72+6*9+6] + bc2.z * kernelsL[l][7*72+6*9+7] + br2.z * kernelsL[l][7*72+6*9+8] + 

        tl2.w * kernelsL[l][7*72+7*9+0] + tc2.w * kernelsL[l][7*72+7*9+1] + tr2.w * kernelsL[l][7*72+7*9+2] +
        ml2.w * kernelsL[l][7*72+7*9+3] + mc2.w * kernelsL[l][7*72+7*9+4] + mr2.w * kernelsL[l][7*72+7*9+5] +
        bl2.w * kernelsL[l][7*72+7*9+6] + bc2.w * kernelsL[l][7*72+7*9+7] + br2.w * kernelsL[l][7*72+7*9+8] + biasL[l][7]
    ));

    write_imagef(tmpImgOut1, coord, c1234);
    write_imagef(tmpImgOut2, coord, c5678);
})"
R"(

__kernel void convTranspose8To1(
    __read_only image2d_t tmpImgIn1,
    __read_only image2d_t tmpImgIn2, 
    __write_only image2d_t dstImg)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(dstImg) || y >= get_image_height(dstImg))
        return;

    int2 coord = (int2)(x, y);

    float4 mc1 = read_imagef(tmpImgIn1, samplerN, (int2)(x / 2, y / 2));
    float4 mc2 = read_imagef(tmpImgIn2, samplerN, (int2)(x / 2, y / 2));

    int2 pos = (int2)(x & 1, y & 1);
    int flag = 0;

    if (pos.x == 0 && pos.y != 0)
        flag = 0;
        //0 x
        //0 0
    else if (pos.x == 0 && pos.y == 0)
        flag = 1;
        //0 0
        //0 x
    else if (pos.x != 0 && pos.y == 0)
        flag = 2;
        //0 0
        //x 0
    else if (pos.x != 0 && pos.y != 0)
        flag = 3;
        //x 0
        //0 0

        //180 degree rotation for kernel
        //0 1  to  3 2
        //2 3      1 0
    float4 c;
    float tmp;
    switch(flag)
    {
    case 0:
        tmp = clamp(
            mc1.x * kernelsL10[0*4+2] +
            mc1.y * kernelsL10[1*4+2] +
            mc1.z * kernelsL10[2*4+2] +
            mc1.w * kernelsL10[3*4+2] +
            mc2.x * kernelsL10[4*4+2] +
            mc2.y * kernelsL10[5*4+2] +
            mc2.z * kernelsL10[6*4+2] +
            mc2.w * kernelsL10[7*4+2], 0.0f, 1.0f);
        
        c = (float4)(tmp, tmp, tmp, 1.0f);
        break;
    case 1:
        tmp = clamp(
            mc1.x * kernelsL10[0*4+0] +
            mc1.y * kernelsL10[1*4+0] +
            mc1.z * kernelsL10[2*4+0] +
            mc1.w * kernelsL10[3*4+0] +
            mc2.x * kernelsL10[4*4+0] +
            mc2.y * kernelsL10[5*4+0] +
            mc2.z * kernelsL10[6*4+0] +
            mc2.w * kernelsL10[7*4+0], 0.0f, 1.0f);

        c = (float4)(tmp, tmp, tmp, 1.0f);
        break;
    case 2:
        tmp = clamp(
            mc1.x * kernelsL10[0*4+1] +
            mc1.y * kernelsL10[1*4+1] +
            mc1.z * kernelsL10[2*4+1] +
            mc1.w * kernelsL10[3*4+1] +
            mc2.x * kernelsL10[4*4+1] +
            mc2.y * kernelsL10[5*4+1] +
            mc2.z * kernelsL10[6*4+1] +
            mc2.w * kernelsL10[7*4+1], 0.0f, 1.0f);
            
        c = (float4)(tmp, tmp, tmp, 1.0f);
        break;
    case 3:
        tmp = clamp(
            mc1.x * kernelsL10[0*4+3] +
            mc1.y * kernelsL10[1*4+3] +
            mc1.z * kernelsL10[2*4+3] +
            mc1.w * kernelsL10[3*4+3] +
            mc2.x * kernelsL10[4*4+3] +
            mc2.y * kernelsL10[5*4+3] +
            mc2.z * kernelsL10[6*4+3] +
            mc2.w * kernelsL10[7*4+3], 0.0f, 1.0f);
            
        c = (float4)(tmp, tmp, tmp, 1.0f);
        break;
    }

    write_imagef(dstImg, coord, c);
})");
#endif // BUILT_IN_KERNEL
