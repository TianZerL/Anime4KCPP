#define DLL

#include "Anime4KGPUCNN.h"

Anime4KCPP::Anime4KGPUCNN::Anime4KGPUCNN(const Parameters& parameters) :
    Anime4K(parameters) {}

void Anime4KCPP::Anime4KGPUCNN::process()
{
    double tmpZf = log2(zf);
    if (tmpZf < 0.0001)
        tmpZf = 1.0 - 0.0002;
    int tmpZfUp = ceil(tmpZf);
    std::function<void(cv::InputArray, cv::OutputArray)> runKernel;
    if (HDN)
    {
        runKernel =
            [this](cv::InputArray orgImg, cv::OutputArray dstImg)
        {
            runKernelACNetHDN(orgImg, dstImg);
        };
    }
    else
    {
        runKernel =
            [this](cv::InputArray orgImg, cv::OutputArray dstImg)
        {
            runKernelACNet(orgImg, dstImg);
        };
    }

    if (!vm)
    {
        if (!inputYUV)
        {
            cv::Mat tmpImg = orgImg;
            cv::Mat uv;
            cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2YUV);
            std::vector<cv::Mat> yuv(3);
            cv::split(tmpImg, yuv);
            tmpImg = yuv[Y];
            cv::merge(std::vector{ yuv[U],yuv[V] }, uv);
            for (int i = 0; i < tmpZfUp; i++)
            {
                dstImg.create(tmpImg.rows * 2, tmpImg.cols * 2, CV_8UC1);
                runKernel(tmpImg, dstImg);
                cv::resize(uv, uv, cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
                tmpImg = dstImg;
            }
            cv::merge(std::vector{ dstImg,uv }, dstImg);
            cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
            if (tmpZfUp - tmpZf > 0.00001)
            {
                cv::resize(dstImg, dstImg, cv::Size(W, H), 0, 0, cv::INTER_LANCZOS4);
            }
        }
        else
        {
            cv::Mat tmpImg = orgImg;
            cv::Mat uv;
            std::vector<cv::Mat> yuv(3);
            cv::split(tmpImg, yuv);
            tmpImg = yuv[Y];
            cv::merge(std::vector{ yuv[U],yuv[V] }, uv);
            for (int i = 0; i < tmpZfUp; i++)
            {
                dstImg.create(tmpImg.rows * 2, tmpImg.cols * 2, CV_8UC1);
                runKernel(tmpImg, dstImg);
                cv::resize(uv, uv, cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
                tmpImg = dstImg;
            }
            cv::merge(std::vector{ dstImg,uv }, dstImg);
            if (tmpZfUp - tmpZf > 0.00001)
            {
                cv::resize(dstImg, dstImg, cv::Size(W, H), 0, 0, cv::INTER_LANCZOS4);
            }
        }
    }
    else
    {
        VideoIO::instance().init(
            [this, tmpZfUp, tmpZf, &runKernel]()
            {
                Frame frame = VideoIO::instance().read();
                cv::Mat orgFrame = frame.first;
                cv::Mat dstFrame;

                cv::Mat tmpFrame = orgFrame;
                cv::Mat uv;
                cv::cvtColor(tmpFrame, tmpFrame, cv::COLOR_BGR2YUV);
                std::vector<cv::Mat> yuv(3);
                cv::split(tmpFrame, yuv);
                tmpFrame = yuv[Y];
                cv::merge(std::vector{ yuv[U],yuv[V] }, uv);
                for (int i = 0; i < tmpZfUp; i++)
                {
                    dstFrame.create(tmpFrame.rows * 2, tmpFrame.cols * 2, CV_8UC1);
                    runKernel(tmpFrame, dstFrame);
                    cv::resize(uv, uv, cv::Size(0, 0), 2.0, 2.0, cv::INTER_LANCZOS4);
                    tmpFrame = dstFrame;
                }
                cv::merge(std::vector{ dstFrame,uv }, dstFrame);
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

void Anime4KCPP::Anime4KGPUCNN::initGPU(unsigned int platformID, unsigned int deviceID, const CNNType type)
{
    if (!isInitialized)
    {
        pID = platformID;
        dID = deviceID;
        initOpenCL(type);
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
        programACNet = nullptr;
        programACNetHDN = nullptr;
        device = nullptr;
        isInitialized = false;
    }
}

bool Anime4KCPP::Anime4KGPUCNN::isInitializedGPU()
{
    return isInitialized;
}

void Anime4KCPP::Anime4KGPUCNN::runKernelACNet(cv::InputArray orgImg, cv::OutputArray dstImg)
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

    cl_kernel kernelConv1To8L1 = clCreateKernel(programACNet, "conv1To8", &err);
    if (err != CL_SUCCESS)
    {
        throw"Failed to create OpenCL kernel L1";
    }
    cl_kernel kernelConv8To8L2 = clCreateKernel(programACNet, "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        throw"Failed to create OpenCL kernel L2";
    }
    cl_kernel kernelConv8To8L3 = clCreateKernel(programACNet, "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        throw"Failed to create OpenCL kernel L3";
    }
    cl_kernel kernelConv8To8L4 = clCreateKernel(programACNet, "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        throw"Failed to create OpenCL kernel L4";
    }
    cl_kernel kernelConv8To8L5 = clCreateKernel(programACNet, "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        throw"Failed to create OpenCL kernel L5";
    }
    cl_kernel kernelConv8To8L6 = clCreateKernel(programACNet, "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        throw"Failed to create OpenCL kernel L6";
    }
    cl_kernel kernelConv8To8L7 = clCreateKernel(programACNet, "conv8To8", &err);
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
    cl_kernel kernelConv8To8L8 = clCreateKernel(programACNet, "conv8To8", &err);
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
    cl_kernel kernelConv8To8L9 = clCreateKernel(programACNet, "conv8To8", &err);
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
    cl_kernel kernelConvTranspose8To1L10 = clCreateKernel(programACNet, "convTranspose8To1", &err);
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

}

void Anime4KCPP::Anime4KGPUCNN::runKernelACNetHDN(cv::InputArray orgImg, cv::OutputArray dstImg)
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

    cl_kernel kernelConv1To8L1 = clCreateKernel(programACNetHDN, "conv1To8", &err);
    if (err != CL_SUCCESS)
    {
        throw"Failed to create OpenCL kernel L1";
    }
    cl_kernel kernelConv8To8L2 = clCreateKernel(programACNetHDN, "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        throw"Failed to create OpenCL kernel L2";
    }
    cl_kernel kernelConv8To8L3 = clCreateKernel(programACNetHDN, "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        throw"Failed to create OpenCL kernel L3";
    }
    cl_kernel kernelConv8To8L4 = clCreateKernel(programACNetHDN, "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        throw"Failed to create OpenCL kernel L4";
    }
    cl_kernel kernelConv8To8L5 = clCreateKernel(programACNetHDN, "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        throw"Failed to create OpenCL kernel L5";
    }
    cl_kernel kernelConv8To8L6 = clCreateKernel(programACNetHDN, "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        throw"Failed to create OpenCL kernel L6";
    }
    cl_kernel kernelConv8To8L7 = clCreateKernel(programACNetHDN, "conv8To8", &err);
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
    cl_kernel kernelConv8To8L8 = clCreateKernel(programACNetHDN, "conv8To8", &err);
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
    cl_kernel kernelConv8To8L9 = clCreateKernel(programACNetHDN, "conv8To8", &err);
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
    cl_kernel kernelConvTranspose8To1L10 = clCreateKernel(programACNetHDN, "convTranspose8To1", &err);
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

}

void Anime4KCPP::Anime4KGPUCNN::initOpenCL(const CNNType type)
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
    std::string ACNetKernelSourceString;
    std::string ACNetHDNKernelSourceString;
#endif // BUILT_IN_KERNEL
    const char* ACNetKernelSource;
    const char* ACNetHDNKernelSource;

    switch (type)
    {
    case CNNType::ACNet:
#ifndef BUILT_IN_KERNEL
        //read kernel files
        ACNetKernelSourceString = readKernel("ACNetKernel.cl");
#endif // BUILT_IN_KERNEL
        ACNetKernelSource = ACNetKernelSourceString.c_str();

        //create program
        programACNet = clCreateProgramWithSource(context, 1, &ACNetKernelSource, nullptr, &err);
        if (err != CL_SUCCESS)
        {
            std::cout << err << std::endl;
            releaseOpenCL();
            throw"Failed to create OpenCL program";
        }

        //build program
        err = clBuildProgram(programACNet, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS)
        {
            size_t buildErrorSize = 0;
            clGetProgramBuildInfo(programACNet, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &buildErrorSize);
            char* buildError = new char[buildErrorSize];
            clGetProgramBuildInfo(programACNet, device, CL_PROGRAM_BUILD_LOG, buildErrorSize, buildError, nullptr);
            releaseOpenCL();
            //print build info
            std::cout << buildError << std::endl;
            delete[] buildError;
            throw"Kernel build error";
        }
        break;
    case CNNType::ACNetHDN:
#ifndef BUILT_IN_KERNEL
        //read kernel files
        ACNetHDNKernelSourceString = readKernel("ACNetHDNKernel.cl");
#endif // BUILT_IN_KERNEL
        ACNetHDNKernelSource = ACNetHDNKernelSourceString.c_str();

        //create program
        programACNetHDN = clCreateProgramWithSource(context, 1, &ACNetHDNKernelSource, nullptr, &err);
        if (err != CL_SUCCESS)
        {
            std::cout << err << std::endl;
            releaseOpenCL();
            throw"Failed to create OpenCL program";
        }

        //build program
        err = clBuildProgram(programACNetHDN, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS)
        {
            size_t buildErrorSize = 0;
            clGetProgramBuildInfo(programACNetHDN, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &buildErrorSize);
            char* buildError = new char[buildErrorSize];
            clGetProgramBuildInfo(programACNetHDN, device, CL_PROGRAM_BUILD_LOG, buildErrorSize, buildError, nullptr);
            releaseOpenCL();
            //print build info
            std::cout << buildError << std::endl;
            delete[] buildError;
            throw"Kernel build error";
        }
        break;
    case CNNType::Default:
#ifndef BUILT_IN_KERNEL
        //read kernel files
        ACNetKernelSourceString = readKernel("ACNetKernel.cl");
        ACNetHDNKernelSourceString = readKernel("ACNetHDNKernel.cl");
#endif // BUILT_IN_KERNEL
        ACNetKernelSource = ACNetKernelSourceString.c_str();
        ACNetHDNKernelSource = ACNetHDNKernelSourceString.c_str();

        //create programACNet
        programACNet = clCreateProgramWithSource(context, 1, &ACNetKernelSource, nullptr, &err);
        if (err != CL_SUCCESS)
        {
            std::cout << err << std::endl;
            releaseOpenCL();
            throw"Failed to create OpenCL program";
        }

        //build programACNet
        err = clBuildProgram(programACNet, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS)
        {
            size_t buildErrorSize = 0;
            clGetProgramBuildInfo(programACNet, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &buildErrorSize);
            char* buildError = new char[buildErrorSize];
            clGetProgramBuildInfo(programACNet, device, CL_PROGRAM_BUILD_LOG, buildErrorSize, buildError, nullptr);
            releaseOpenCL();
            //print build info
            std::cout << buildError << std::endl;
            delete[] buildError;
            throw"Kernel build error";
        }

        //create programACNetHDN
        programACNetHDN = clCreateProgramWithSource(context, 1, &ACNetHDNKernelSource, nullptr, &err);
        if (err != CL_SUCCESS)
        {
            std::cout << err << std::endl;
            releaseOpenCL();
            throw"Failed to create OpenCL program";
        }

        //build programACNetHDN
        err = clBuildProgram(programACNetHDN, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS)
        {
            size_t buildErrorSize = 0;
            clGetProgramBuildInfo(programACNetHDN, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &buildErrorSize);
            char* buildError = new char[buildErrorSize];
            clGetProgramBuildInfo(programACNetHDN, device, CL_PROGRAM_BUILD_LOG, buildErrorSize, buildError, nullptr);
            releaseOpenCL();
            //print build info
            std::cout << buildError << std::endl;
            delete[] buildError;
            throw"Kernel build error";
        }
        break;
    }
}

void Anime4KCPP::Anime4KGPUCNN::releaseOpenCL()
{
    if (programACNet != nullptr)
        clReleaseProgram(programACNet);
    if (programACNetHDN != nullptr)
        clReleaseProgram(programACNetHDN);
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
cl_program Anime4KCPP::Anime4KGPUCNN::programACNet = nullptr;
cl_program Anime4KCPP::Anime4KGPUCNN::programACNetHDN = nullptr;
cl_device_id Anime4KCPP::Anime4KGPUCNN::device = nullptr;
unsigned int Anime4KCPP::Anime4KGPUCNN::pID = 0U;
unsigned int Anime4KCPP::Anime4KGPUCNN::dID = 0U;

#ifdef BUILT_IN_KERNEL
const std::string Anime4KCPP::Anime4KGPUCNN::ACNetKernelSourceString = 
R"(#define RELU(x) fmax(x, 0.0f)

__constant sampler_t samplerN = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__constant float kernelsL1[9 * 8] = 
{
 9.1525e-02,  1.2987e-01, -1.2165e-02,
-1.2689e-01,  7.3165e-01,  2.0307e-01,
 1.0727e-01,  9.0878e-02,  3.7344e-02,
 9.7309e-04,  1.6503e-01, -8.5758e-01,
 4.7297e-02,  8.5396e-01, -2.4973e-01,
-1.8050e-02,  6.9560e-02, -1.4136e-03,
-1.6474e-01,  6.3990e-01,  6.5482e-03,
-7.9771e-01,  4.1846e-01,  3.2384e-02,
-6.0763e-02, -1.9540e-02, -1.8302e-02,
-1.0385e-02, -9.2983e-01,  9.8546e-02,
-4.0637e-02,  1.5750e-01,  7.1096e-01,
-1.8174e-02,  2.7848e-02,  1.4766e-02,
 2.0535e-01, -4.1992e-02,  1.4843e-01,
 3.0405e-01,  5.6066e-01,  2.5997e-01,
 1.6505e-01, -1.7049e-01, -3.0099e-02,
 8.4347e-02, -2.5118e-01, -2.0671e-01,
 3.0869e-01, -3.0934e-01, -2.2030e-01,
 2.5821e-01,  3.0186e-01,  5.5492e-02,
-4.8591e-03,  4.5442e-02, -2.7406e-02,
-3.6979e-03,  1.0266e+00, -8.2313e-02,
 2.1112e-02, -3.0772e-01, -6.2881e-01,
-2.1588e-02, -8.8731e-01, -3.0740e-02,
 8.1086e-02,  9.0321e-01, -3.9606e-02,
 5.8978e-03,  9.8026e-04, -5.2215e-03
};

__constant float biasL1[8] = 
{
-0.7444, -0.0235,  0.0270, -0.0205,  0.0126,  0.0243,  0.0165, -0.0089
};
)"
R"(
__constant float kernelsL[8][9 * 8 * 8] = 
{
{
 1.6309e-01,  3.4023e-02, -1.0644e-01,
 1.3544e-01, -3.4155e-02, -7.4781e-02,
-3.9547e-02, -2.1064e-02, -4.4772e-02,
-1.3930e-01, -6.1934e-02, -5.3794e-02,
-2.0723e-02, -1.3269e-01, -1.2214e-01,
-1.5925e-01,  4.5216e-02,  3.7524e-02,
 5.1659e-02, -7.6368e-02, -3.0998e-01,
 3.8821e-02, -1.2660e-01,  7.2032e-02,
-9.4096e-02, -1.6277e-01,  1.0775e-01,
 3.3463e-01,  7.3684e-02,  2.5252e-02,
 7.8584e-02,  3.4059e-02,  1.8694e-01,
 2.1807e-01,  4.5848e-01,  9.5805e-03,
 2.6816e-02,  2.7793e-01, -1.2906e-01,
 2.6475e-02, -3.8952e-02, -1.2232e-01,
 3.1018e-02,  1.2465e-01, -2.1049e-01,
-6.0398e-02, -7.2330e-02,  5.4857e-02,
-1.2239e-01, -3.6885e-03, -2.4250e-02,
-1.2981e-01,  1.3868e-02, -6.6184e-02,
 4.1047e-01,  2.8986e-01, -4.8715e-02,
 1.2602e-01, -8.3619e-02, -7.2921e-02,
-4.4857e-02, -2.6865e-02,  6.4199e-02,
 7.2907e-02, -8.2036e-02, -3.4225e-04,
-5.6952e-02, -7.7810e-03,  7.1701e-02,
-2.1782e-01,  8.1458e-02, -1.1999e-01,
 4.5565e-02,  1.5311e-01,  3.5267e-02,
 1.3634e-01,  5.4400e-01,  7.0090e-02,
-8.5654e-03,  1.4545e-01, -6.2863e-03,
-7.1474e-02, -1.9124e-02, -5.0147e-02,
-3.6432e-02,  1.2984e-01,  1.7426e-02,
-3.8385e-02,  8.8372e-02,  2.0229e-02,
 2.0608e-02,  8.5619e-03, -4.2199e-02,
 7.9916e-02,  1.3692e-01, -1.3246e-01,
 1.2340e-01,  1.8653e-01, -4.9366e-02,
 1.9616e-02,  7.2897e-03,  7.2521e-03,
 1.0813e-01, -2.0087e-02, -2.7743e-02,
-5.2871e-02, -6.8949e-02,  9.3131e-02,
-3.3771e-02, -8.7242e-02, -5.1316e-02,
 8.5239e-02,  1.9902e-01,  3.5253e-03,
 4.6975e-02, -7.7727e-02,  2.0624e-02,
-1.0401e-01,  2.5474e-02,  5.5150e-02,
-9.7941e-02, -9.8904e-02,  4.2504e-02,
-5.3224e-02, -1.0917e-01, -3.2888e-02,
-1.4051e-01, -1.4500e-01, -7.2245e-02,
-5.0164e-02,  3.0433e-01,  5.2746e-02,
 1.0930e-02,  6.0765e-02, -1.0215e-02,
 2.1769e-02, -2.4786e-03,  6.8703e-02,
 7.4485e-02,  1.7673e-01,  4.2216e-02,
 1.5808e-02, -3.8343e-02,  1.4138e-02,
-9.8947e-02, -2.1655e-02,  2.1519e-02,
 8.9935e-02,  2.1302e-01, -4.7684e-02,
-1.0824e-01,  1.3952e-02,  7.8060e-02,
-2.0363e-03,  2.8165e-03,  1.1806e-02,
-8.9505e-02,  3.8269e-03, -4.9635e-02,
-2.6675e-03, -1.1953e-01,  2.8445e-02,
-1.8424e-02,  1.0195e-02, -1.5736e-01,
-6.1390e-02, -1.1363e-02, -7.6867e-02,
 5.8812e-02,  1.4854e-01, -2.7585e-01,
-2.9248e-02, -5.5216e-03, -9.7823e-02,
 1.0578e-01,  1.4817e-01, -6.8020e-02,
 1.5200e-01, -1.7457e-01, -3.5923e-02,
-8.4576e-02, -1.4026e-01, -7.6933e-02,
-3.1721e-02,  3.6490e-02,  1.7413e-01,
 1.1586e-01,  5.7193e-02,  9.0626e-02,
 3.0931e-02,  4.0230e-02, -6.6769e-02,
 7.9180e-02,  3.0398e-01,  1.6868e-01,
-1.0894e-01, -1.2762e-01, -5.7816e-03,
-5.2775e-02,  1.8684e-01,  2.0462e-02,
 1.2639e-01,  2.6609e-01, -5.3110e-03,
-4.4305e-02,  1.5906e-02, -3.5725e-02,
-8.4823e-02,  8.6374e-03, -5.5090e-02,
-1.2836e-01, -2.1579e-01, -1.8008e-01,
-7.6224e-02, -3.0877e-01, -1.5891e-01,
-1.0946e-01,  1.0119e-01, -1.0009e-02,
-1.9025e-02,  1.1753e-01,  1.1732e-02,
-2.5564e-02,  2.9803e-03,  2.0568e-02,
-4.7708e-02, -2.0758e-01, -1.8242e-02,
-8.5467e-02,  2.1979e-01, -1.2908e-01,
-2.0529e-02, -7.4663e-02,  9.3366e-02,
-2.2268e-02,  8.4558e-02,  1.2014e-01,
 5.1386e-03,  2.7058e-01,  2.2316e-01,
 1.5556e-02,  1.3433e-01,  2.1588e-01,
-1.0867e-01,  4.7044e-02,  3.5898e-03,
 9.0893e-03, -1.1928e-01, -1.7057e-02,
 7.1007e-02, -2.1654e-01,  7.2172e-02,
-4.6778e-02, -1.3996e-01, -1.0592e-01,
 3.7136e-02,  2.0642e-01,  4.9995e-03,
 1.4909e-02,  2.7320e-03,  1.0128e-01,
 1.1318e-02,  2.0564e-02,  1.2616e-01,
 4.6025e-02,  1.0520e-01, -9.5458e-02,
-2.1315e-02,  4.6180e-02, -2.8551e-02,
 1.5504e-01, -1.0777e-02, -1.0742e-01,
 6.3375e-02,  1.5946e-02,  1.2388e-01,
-2.4317e-02,  1.9888e-02,  3.2870e-03,
-1.5091e-02, -5.4944e-02,  8.4084e-02,
-1.1141e-01, -2.0755e-01, -7.9829e-02,
 6.4521e-02,  2.2505e-01, -1.3396e-01,
-4.4621e-02,  5.0667e-02,  5.9454e-02,
-3.0299e-02, -6.3112e-02,  1.8483e-01,
-6.8640e-02, -5.7660e-02, -1.0391e-02,
 7.5915e-02,  3.2944e-01,  1.1235e-01,
 2.4987e-01,  3.2734e-02,  1.8120e-01,
 3.3713e-01,  1.8123e-01,  5.3235e-02,
-9.7193e-02,  4.5638e-02, -8.9397e-02,
 3.0153e-02,  2.4615e-03,  7.3330e-01,
-6.2854e-02, -1.9801e-01,  2.7244e-01,
-3.7100e-02, -6.8288e-03, -4.7730e-02,
-8.9903e-02,  5.6748e-02, -1.4740e-01,
-8.8403e-02,  1.6297e-01, -9.4453e-02,
 2.1056e-02,  1.8706e-01,  1.2057e-01,
-1.4276e-01, -2.2631e-01,  1.8046e-01,
-8.0078e-02,  4.8505e-02, -1.0355e-01,
 4.2962e-02, -2.8774e-01, -5.2552e-01,
 3.0619e-04, -1.7129e-01, -1.1756e-01,
 8.1629e-02, -4.2066e-02, -1.0642e-01,
-9.9381e-02,  1.8124e-01,  6.5231e-02,
-5.6718e-02, -2.4679e-01, -2.8641e-02,
 9.8096e-02, -3.5493e-02, -8.2769e-02,
 2.0096e-02,  1.5187e-02,  4.9524e-02,
-7.4919e-03,  1.4722e-02,  2.4456e-01,
-1.0047e-01,  2.2566e-01,  4.5218e-02,
 1.6938e-01,  1.6919e-01,  9.3585e-02,
 2.5456e-01,  2.6972e-01,  1.1719e-01,
 9.8375e-02,  1.0038e-01,  6.9345e-02,
-2.9667e-02,  1.2724e-01,  2.1468e-02,
-1.6529e-01,  2.4435e-01,  7.8158e-02,
-1.7724e-01, -1.4400e-02,  2.0975e-02,
 4.0921e-03,  7.1360e-02, -1.5502e-01,
 6.9491e-02,  8.5696e-02, -3.6883e-01,
 3.6088e-02,  2.7398e-01, -3.9716e-02,
 6.2210e-02,  6.7503e-02, -2.6578e-02,
 1.9520e-01,  1.6376e-01, -5.9952e-02,
 2.0297e-02, -3.2806e-01, -1.0079e-03,
 7.2485e-02,  1.7281e-01,  1.6376e-01,
 5.7144e-02,  2.2883e-01,  2.7823e-01,
 3.0753e-02,  2.7461e-01,  1.4941e-01,
-3.7559e-02,  2.6351e-02,  3.0868e-02,
 1.4599e-02, -1.5070e-01,  4.8130e-02,
-6.5058e-02,  8.1646e-02,  6.1219e-02,
-3.3468e-01, -2.5825e-01,  4.5093e-02,
 9.2813e-02,  3.7744e-01,  7.2829e-02,
-1.4749e-02,  6.6809e-02, -6.2536e-02,
-2.7537e-02,  1.6111e-02, -6.6691e-03,
-5.8757e-03,  1.3962e-01,  3.6707e-02,
-3.6268e-02, -2.0297e-01,  1.0023e-01,
 9.0534e-02, -1.6367e-01,  2.5921e-02,
-6.7906e-02,  7.0495e-02,  1.5111e-01,
 7.2133e-02, -9.9682e-02, -6.7363e-03,
-2.6878e-02, -9.7501e-02, -3.8800e-02,
 1.3456e-01,  3.7210e-01,  8.2362e-02,
-3.0078e-01, -8.8079e-02,  3.0197e-02,
 1.5333e-02, -3.7870e-02,  9.6758e-02,
-1.4945e-03,  8.3713e-02,  2.1000e-02,
-9.0892e-02,  8.2174e-02, -1.8649e-02,
-9.1292e-02, -7.0315e-02, -1.2339e-01,
 1.9703e-02,  2.8892e-01, -8.0176e-02,
-3.6099e-02, -2.0788e-01, -5.4340e-02,
-1.1412e-02, -1.4606e-01, -2.2464e-02,
-1.6289e-01,  2.6988e-01,  1.1389e-01,
-1.1400e-01, -1.2370e-01,  1.6619e-01,
 1.5999e-01,  6.0306e-02, -1.1184e-01,
 1.3734e-01,  1.6022e-01, -7.6023e-03,
 2.8858e-02,  4.4842e-02,  2.5424e-02,
 2.2040e-01, -7.3930e-02, -1.5102e-02,
-9.4118e-02, -4.0745e-03,  5.1706e-02,
 7.4597e-03,  2.9031e-02, -1.4606e-02,
 6.9099e-02,  6.6094e-02,  1.2646e-01,
 1.2643e-01,  6.5075e-01, -4.3285e-03,
 1.9970e-01,  5.4293e-02,  4.0401e-02,
-4.6923e-02,  6.2494e-02, -1.7039e-02,
 3.7409e-02,  6.5111e-02, -4.3097e-02,
-4.6026e-02,  1.3961e-02, -1.8598e-02,
-6.7093e-02,  1.6921e-02,  3.6485e-02,
-1.2109e-01,  2.5291e-01, -4.9230e-02,
-1.1335e-01,  5.5626e-01, -4.3746e-03,
-3.1912e-02,  1.2722e-01, -9.9254e-03,
 2.6112e-04,  6.5248e-02, -3.1488e-02,
 1.6853e-02, -6.1447e-02, -1.6328e-01,
-2.2121e-02,  1.5277e-01,  2.5843e-02,
-2.1661e-02,  2.3095e-01,  1.1211e-01,
 3.4014e-02, -1.6510e-02,  1.4900e-01,
-1.7120e-02, -1.1956e-01, -7.4910e-02,
 4.6907e-02,  5.1095e-01, -1.2627e-02,
-1.2948e-02, -1.7091e-01, -1.6410e-01,
 1.0566e-02,  1.1434e-02, -2.9993e-02,
 1.0074e-02, -1.2135e-01, -2.2957e-02,
-2.0095e-02,  5.5913e-02,  1.6931e-02,
 5.0783e-02,  2.6236e-02, -3.5124e-02,
-2.9547e-02,  7.3516e-01,  1.2204e-02,
-1.2740e-02,  1.1552e-01, -5.6759e-03,
-1.2931e-02, -1.3538e-01, -6.4899e-02,
-1.4338e-01, -1.6226e-01, -9.3140e-02,
 1.0225e-02, -1.2771e-01,  1.0984e-01
}
,)"
R"(
{
-1.6304e-02, -1.5259e-02, -3.4533e-02,
 7.4577e-02,  5.2887e-02, -5.8468e-02,
-1.7804e-02, -7.1103e-02, -1.0957e-01,
 8.8385e-02,  6.5036e-02,  4.0093e-02,
 1.9346e-03, -3.4680e-02, -6.1286e-02,
 2.2625e-02,  2.2828e-02, -2.4536e-03,
 8.7822e-02,  1.3399e-01,  2.7128e-02,
 1.7581e-01,  3.1304e-01, -2.7751e-02,
 5.1399e-02, -1.8838e-02,  4.3625e-02,
 5.4746e-02, -1.7915e-02, -2.4884e-02,
-1.0906e-01,  7.7627e-02, -5.7094e-03,
 1.3469e-02, -1.0353e-02, -7.0626e-02,
 3.9188e-02,  1.1244e-01, -4.0835e-02,
 1.1216e-01,  3.3274e-01,  8.6200e-02,
 1.1605e-01,  1.0997e-01,  4.1749e-02,
-2.3950e-02, -1.4379e-01, -1.4632e-01,
 6.3082e-02,  1.4401e-01, -1.5967e-01,
 3.0769e-02,  1.0723e-01,  3.6826e-02,
-5.8066e-02, -5.0653e-02,  7.0175e-02,
-6.6808e-02,  2.9722e-01,  1.6889e-01,
-4.4368e-02, -3.7799e-02, -1.8119e-02,
-5.1212e-02,  2.9746e-01,  1.9968e-02,
-6.4673e-02,  4.4536e-01, -4.4624e-02,
 3.6075e-02, -2.4715e-02,  6.4896e-02,
-7.7666e-03,  1.6829e-01,  4.2689e-02,
 2.0900e-01, -1.7027e-01,  6.9642e-02,
 1.9921e-01,  7.6841e-02, -1.1614e-01,
-1.1936e-01, -2.6975e-01,  1.2306e-01,
 4.2531e-02,  7.7155e-02, -7.2232e-02,
 3.9764e-03, -5.0019e-02, -1.0286e-01,
-1.7974e-01, -6.1443e-01, -2.6852e-02,
-1.5173e-01,  3.8542e-02,  7.2637e-02,
-3.5979e-02,  1.1616e-01,  8.8519e-02,
 1.4012e-02,  7.6097e-04,  7.2868e-02,
 7.6098e-02,  1.6530e-01,  7.9563e-02,
 3.8939e-02, -2.0747e-02, -5.7947e-02,
-1.2958e-02, -7.2288e-02, -1.2560e-01,
 1.9081e-01, -1.4503e-01,  1.5193e-01,
-9.8426e-02, -5.3179e-02, -5.6634e-02,
-8.2376e-02, -2.5052e-01,  1.2358e-01,
 6.8578e-02,  2.5537e-01, -1.8597e-02,
-3.3457e-02, -1.2729e-02, -1.0734e-02,
-2.5427e-01,  2.1288e-01,  1.0601e-01,
 2.0793e-01, -3.1142e-02, -3.9566e-01,
 5.9573e-02, -2.4914e-01,  3.7802e-02,
 5.6152e-02,  2.1599e-01,  1.0172e-01,
-9.6459e-02,  1.3210e-01, -1.1257e-01,
-1.0093e-02, -2.6411e-02,  3.2016e-02,
-4.6980e-02,  5.9547e-03, -7.4781e-03,
 5.3980e-02, -9.2175e-02,  2.9904e-02,
-2.5560e-02,  5.0707e-02,  3.6427e-02,
-2.6222e-02, -2.8691e-02,  5.8748e-02,
-6.8207e-02,  2.0746e-01, -2.1217e-02,
-1.5203e-03,  7.4029e-02, -1.3360e-01,
 6.9732e-04,  2.1887e-02, -6.5914e-02,
 8.4041e-02,  1.0294e-01, -1.0208e-01,
-6.4547e-02,  1.2947e-02,  1.8802e-02,
-2.8208e-02,  6.1354e-04, -7.5622e-02,
-2.7965e-02,  1.6670e-01,  1.4863e-02,
-3.9030e-02,  4.4913e-02, -4.3190e-02,
-4.1890e-03, -3.4690e-02, -1.3523e-02,
 6.5418e-02, -1.7734e-01, -9.0739e-02,
 1.2964e-01,  1.0544e-01, -5.3703e-03,
-4.2420e-02,  1.3613e-01,  1.1739e-01,
 9.3468e-02,  3.9639e-01,  1.5257e-01,
 3.1675e-02,  2.4513e-01,  6.4383e-02,
-7.5424e-03,  9.7941e-02, -7.3230e-03,
 1.2823e-01,  1.4628e-01,  1.0280e-01,
-2.1140e-02, -3.3449e-01, -1.1302e-01,
 1.5172e-01, -1.3710e-01,  4.7690e-02,
-1.8346e-01,  2.8494e-01,  5.4008e-03,
-8.2883e-02, -8.9155e-02,  9.8168e-03,
 1.4221e-02, -7.2514e-02, -1.5075e-01,
-3.7863e-01, -1.0862e-01, -7.1934e-02,
-2.4173e-01,  8.3331e-02,  2.3093e-02,
-1.2706e-01, -5.4303e-02, -8.4963e-02,
 1.9434e-01,  2.8273e-01,  1.4961e-02,
-1.2506e-01,  1.6473e-02, -8.0191e-02,
-1.7917e-01, -2.2095e-01, -3.9601e-02,
 1.2276e-03,  2.2876e-01, -2.2004e-02,
-1.6175e-01, -1.3375e-01, -9.5515e-02,
 6.1224e-02, -2.0110e-01, -1.5703e-01,
 2.3073e-01,  4.8938e-02, -1.2253e-02,
 8.8781e-02,  3.2108e-01,  2.2462e-02,
 8.5587e-03,  1.4212e-01,  9.1531e-02,
-2.2330e-01, -9.0909e-03, -6.1855e-05,
-2.4947e-01, -9.6245e-02, -1.9082e-02,
-5.9159e-02, -4.2217e-02, -5.7761e-02,
 2.3891e-01,  1.7832e-01,  8.8288e-02,
-2.3427e-01, -7.5928e-02,  2.2096e-02,
 7.1516e-02,  7.5609e-03, -3.0768e-02,
-1.2421e-02, -2.1726e-01, -5.6448e-03,
-1.8596e-01, -1.4628e-01,  7.1371e-02,
-2.4888e-01, -4.7092e-02,  1.4764e-02,
-3.7654e-01,  1.7358e-01,  3.7786e-02,
-6.9945e-02,  5.8890e-03,  5.0599e-02,
-8.4273e-03,  2.8437e-02, -1.0851e-02,
 1.9830e-02,  1.5947e-02,  7.0837e-03,
-9.9512e-02, -2.9114e-01, -1.6577e-01,
-1.2317e-02,  4.4098e-02, -9.9847e-02,
 2.3167e-03,  1.7731e-02, -2.5885e-02,
 7.6381e-05, -2.6880e-03, -1.1335e-01,
-3.1962e-03,  3.1757e-02, -8.7162e-02,
-6.4189e-02,  1.4908e-01, -3.7301e-02,
-5.8663e-03, -8.6204e-02,  3.4202e-02,
 2.1034e-02,  6.3676e-02, -1.1067e-01,
-1.6279e-01,  1.2666e-01, -1.1280e-01,
-3.7217e-02,  1.4879e-01,  2.1549e-02,
 4.3648e-02, -2.0008e-02,  5.6290e-02,
-5.9007e-02, -1.8034e-01,  1.3739e-01,
-4.0764e-01, -2.1976e-01, -8.1439e-02,
-1.2560e-02,  7.7694e-02, -5.0005e-02,
-7.5693e-02,  3.1406e-01,  5.4990e-02,
-3.1828e-01, -2.1515e-01,  7.3737e-02,
-4.1974e-02,  8.3588e-02,  1.8526e-03,
-3.0856e-02,  3.9223e-01, -9.9465e-02,
-8.0604e-02,  5.4078e-02,  9.2650e-03,
-2.8841e-02, -6.9434e-02,  2.6255e-02,
 1.7887e-01,  4.4402e-01, -6.9519e-03,
-2.7231e-02, -1.2482e-01,  3.8045e-02,
-2.4271e-02,  2.7682e-02, -2.4453e-02,
-4.2968e-02, -2.4313e-01,  7.9359e-02,
 4.9169e-02,  3.9988e-02,  2.1381e-02,
 2.2848e-02,  9.0229e-02,  9.1418e-02,
 5.5207e-02, -5.3192e-02,  1.7801e-01,
-5.5054e-02, -1.5243e-01,  3.6868e-02,
 1.1479e-01, -1.8424e-01,  8.7466e-02,
 1.9308e-01, -2.7111e-01, -7.8829e-02,
 3.7374e-02, -5.9665e-02,  4.4351e-02,
-1.1755e-01,  1.1269e-01,  2.6228e-02,
-3.1594e-01,  2.1307e-01, -4.0538e-02,
 8.7597e-02, -1.3649e-01, -3.9263e-02,
-6.4796e-04, -4.7880e-02, -3.9439e-02,
 6.3212e-02, -5.7021e-02,  1.9964e-02,
 1.4444e-02,  5.5589e-02,  8.7772e-03,
-3.9081e-02,  2.5352e-01,  1.5027e-02,
 1.1082e-03,  9.2004e-02,  4.1644e-03,
-9.1702e-02, -1.6055e-01, -3.6069e-02,
-1.9388e-02,  8.7558e-02, -2.3601e-02,
-5.9829e-02,  4.3247e-01,  5.9034e-02,
 7.3547e-02,  2.9969e-01,  5.4821e-02,
-6.6660e-03, -1.1119e-01,  1.0944e-01,
 2.9385e-01,  2.3305e-01, -1.5432e-01,
 1.3569e-01, -2.7611e-02,  4.1474e-02,
-1.2849e-01, -1.0010e-01, -2.1456e-01,
 1.4546e-01, -3.6194e-01, -6.3833e-02,
 2.2973e-02, -1.5524e-01, -1.0043e-01,
 2.3880e-02,  1.4694e-02,  1.5615e-01,
 7.6788e-04, -2.4077e-01,  4.8979e-02,
 1.2381e-02, -1.1962e-02,  1.2489e-01,
 8.3686e-02, -3.3945e-01,  1.4806e-01,
 4.1149e-02, -4.8930e-01, -1.5602e-01,
 6.1109e-02, -5.0174e-02,  8.6515e-02,
-1.4763e-01, -7.0457e-02,  2.5476e-01,
-1.4556e-01, -5.9108e-01, -1.7153e-02,
-1.6666e-01, -4.0073e-01, -1.2165e-01,
 1.4910e-01,  1.4844e-01, -3.4032e-01,
 2.3586e-01,  5.0550e-01, -1.0319e-02,
 1.6353e-01,  2.2512e-01,  2.2557e-01,
-1.1973e-01,  1.6259e-01,  1.1331e-01,
-5.5398e-02,  8.4291e-02, -1.0896e-01,
-1.0490e-03,  2.1220e-03,  6.3577e-02,
-1.9180e-03,  5.2529e-02, -3.3156e-02,
-1.4843e-03,  6.8839e-02,  1.2176e-01,
-7.4941e-03,  1.0489e-01, -5.5650e-02,
 2.7062e-01,  3.4581e-01,  2.8482e-02,
 1.0037e-01, -9.1680e-02,  2.5829e-02,
 4.3927e-02,  5.2626e-02,  1.0651e-02,
-3.6613e-01, -2.1787e-01, -2.8277e-02,
-2.6189e-01, -2.8647e-02, -1.6495e-01,
-1.8207e-01, -1.7935e-01, -2.2356e-01,
 2.3376e-02, -1.5426e-01,  6.9370e-02,
 1.8375e-01,  1.6174e-01, -3.2298e-03,
-1.3784e-01,  6.6850e-02, -7.6188e-02,
-7.5593e-02, -1.8752e-01,  1.4364e-03,
 2.8904e-02, -4.6072e-02, -1.0297e-01,
-4.3277e-02,  1.4367e-02, -4.1096e-02,
-6.8872e-02, -1.0933e-01, -3.5656e-02,
 1.9224e-01, -8.0228e-02,  1.8951e-02,
 2.6508e-03, -6.5340e-02, -5.3869e-02,
 1.3385e-01,  3.0106e-01, -6.0302e-03,
 3.2288e-01,  6.7647e-02,  1.3296e-01,
 1.5801e-01,  2.2801e-01,  3.8054e-02,
 1.6161e-01, -9.7931e-02, -3.9507e-02,
 1.3613e-01,  2.1193e-01, -1.4942e-01,
-3.0580e-02, -1.6892e-02, -1.2110e-01,
-4.7136e-02,  8.8742e-03,  5.9674e-02,
-7.2658e-02,  2.7273e-01, -2.2309e-02,
-3.2272e-02,  6.0526e-02,  7.4599e-02,
 5.4231e-02, -1.6147e-01,  4.9560e-02,
-1.2621e-01, -6.0050e-02, -1.1529e-01,
-7.7633e-03, -8.3354e-02,  3.1217e-02
}
,)"
R"(
{
 3.6052e-02,  3.4621e-02, -5.6890e-02,
-1.8456e-01,  3.4451e-01, -2.0225e-01,
-7.7507e-03,  1.4962e-01, -3.2353e-02,
-6.6544e-02, -1.9405e-02,  4.6193e-02,
 1.9466e-02,  1.4022e-01,  1.8383e-01,
 1.1064e-01,  2.9516e-01,  3.0413e-01,
-8.2010e-02,  1.5463e-01, -1.0474e-01,
 2.0379e-01,  1.5831e-01, -2.4316e-01,
 2.9324e-03, -1.4474e-01,  6.3222e-02,
 1.2785e-01,  9.4686e-02, -5.2423e-02,
 1.6976e-01, -3.9742e-01, -4.1255e-01,
-2.6479e-02, -1.4171e-01, -3.1756e-01,
 1.0108e-01,  1.0822e-01, -5.5083e-03,
 3.4204e-02,  2.6739e-01, -2.7919e-01,
 1.1757e-02, -1.3338e-03,  9.8044e-02,
-1.1726e-01, -8.9320e-02,  1.8730e-01,
-6.2849e-02,  3.0453e-01, -2.7507e-01,
 3.9103e-02,  1.0060e-02, -2.2921e-02,
 5.8906e-03, -1.5092e-02,  2.2733e-02,
 8.1682e-02,  1.8022e-01,  2.3563e-01,
 1.5385e-01, -6.9944e-02, -2.8096e-02,
 5.8146e-02,  2.2228e-02, -4.9726e-02,
-1.0895e-01, -1.4289e-02, -2.8597e-01,
-3.3746e-02,  4.6049e-02, -1.0297e-01,
-1.2780e-02, -1.0326e-01,  5.9900e-02,
-1.3117e-01,  1.0328e-01, -5.5761e-02,
 9.3783e-04,  9.0984e-02, -8.2746e-02,
-2.4299e-02, -1.1664e-01,  4.5992e-02,
-9.5047e-02,  6.6922e-02,  1.7089e-01,
 2.3512e-02,  1.9950e-01, -7.8010e-02,
-7.0675e-02,  1.3956e-04, -2.3979e-02,
-1.5210e-02,  1.2387e-01, -3.2405e-02,
 2.9205e-02, -4.5021e-02,  4.4769e-02,
 4.8699e-02,  1.8674e-01,  2.6148e-02,
 3.2858e-02, -6.6058e-02, -2.1850e-01,
-2.9703e-02, -1.5401e-01,  2.9471e-02,
-2.5232e-02, -1.0809e-01, -4.0960e-02,
-3.9731e-02, -1.1070e-01, -5.6436e-02,
-2.5963e-02, -6.7217e-02, -1.7604e-02,
 1.9412e-02,  1.0432e-02,  1.2097e-01,
 4.8418e-02,  2.5475e-01,  1.6736e-01,
 8.0102e-02,  1.1064e-02, -9.1235e-03,
-3.5466e-02,  8.5489e-03, -2.2876e-01,
 7.9774e-02,  4.3996e-01,  7.1788e-02,
-6.7694e-02,  2.4337e-02,  2.8796e-02,
 1.7331e-02, -5.7807e-02, -1.8174e-01,
 1.5716e-03, -1.2830e-01,  3.2977e-02,
 2.1258e-03, -7.7841e-02, -6.3963e-02,
-3.4208e-03,  2.6726e-03,  1.6739e-02,
 1.7681e-01,  8.4013e-02,  1.4205e-01,
 1.1346e-01,  8.1073e-02,  6.4598e-02,
 1.0881e-01,  1.7939e-01,  1.1230e-01,
 9.5038e-02, -1.5000e-01,  2.2263e-02,
-9.3159e-02, -3.9874e-02, -9.3774e-02,
 1.1033e-01,  6.6187e-02,  2.8421e-02,
-2.0727e-01, -2.0690e-01,  2.4204e-01,
 1.5663e-03, -1.5323e-01,  1.0063e-01,
 1.8832e-03,  1.6220e-01, -4.3773e-02,
-1.5621e-01,  2.6838e-01,  1.0533e-01,
-4.3969e-03,  2.7431e-02, -4.5390e-03,
-3.5250e-02, -5.9808e-02,  3.7575e-02,
-1.4553e-02, -1.7267e-01,  6.1080e-03,
 2.5201e-02, -5.8737e-02,  7.0402e-02,
 3.4582e-02, -1.8023e-02,  6.1123e-02,
-4.8048e-03, -2.2309e-01, -7.6041e-03,
-1.2846e-02, -1.2084e-01,  1.8511e-02,
 2.8638e-02,  5.0411e-02, -4.5000e-02,
-2.4603e-01,  6.8063e-02, -4.5123e-02,
-1.0779e-01, -4.1124e-01, -1.0606e-01,
-8.3005e-02,  6.8461e-02,  5.2094e-02,
 1.0122e-01, -3.9647e-02,  7.5280e-02,
-4.5633e-02,  6.3015e-02,  2.1763e-02,
-9.1871e-02, -1.6871e-04, -5.6709e-02,
-2.8143e-02, -2.1318e-01, -1.3768e-01,
-2.0336e-01, -1.9740e-01,  1.0032e-02,
-2.4394e-02,  7.8221e-02,  2.3240e-02,
-1.2603e-01,  1.9713e-01,  5.4913e-02,
 3.4564e-01,  4.2950e-01,  4.6219e-02,
-1.4071e-01, -1.7502e-01, -1.5092e-01,
 1.3900e-01,  1.9049e-01, -1.4400e-01,
 1.4983e-01,  1.0648e-01,  9.9875e-03,
-2.4008e-01, -3.5856e-01, -1.4906e-01,
-4.5971e-02, -1.4303e-01, -2.0625e-02,
 9.6510e-02,  1.9878e-01,  2.8691e-01,
 1.0373e-02, -6.0305e-03, -3.2120e-02,
 5.0543e-02,  3.4484e-01,  1.7003e-01,
-3.3169e-02,  1.9235e-02, -1.1550e-02,
 6.9375e-02,  2.0397e-01,  1.0215e-01,
-9.8078e-02,  2.3045e-01,  7.6335e-03,
 8.6979e-02, -1.1594e-02, -4.0665e-02,
 6.0576e-02,  1.9997e-02,  2.4582e-02,
 1.4573e-01,  1.9129e-01,  7.7291e-02,
 2.7395e-01,  1.2455e-01,  1.0897e-01,
-5.8939e-02, -2.1053e-01,  1.6640e-01,
 6.7542e-02, -2.8875e-01, -4.0225e-01,
-3.1788e-01,  8.2155e-02,  2.1137e-01,
-2.4335e-02,  7.3900e-03, -1.3835e-02,
-2.2684e-02,  1.1126e-01, -2.0632e-02,
-2.6828e-02, -1.4141e-01,  6.2634e-03,
 1.0871e-01,  9.1503e-02,  2.1299e-02,
-2.0827e-02,  1.5228e-01,  8.8957e-02,
 1.5061e-01,  2.9296e-01, -1.2439e-01,
-3.9480e-02, -2.2632e-01, -8.7455e-02,
 1.1304e-01,  2.6681e-01,  1.9107e-01,
-2.7437e-02, -1.2718e-01, -6.8369e-02,
-2.2449e-01, -1.7140e-01,  1.3401e-01,
-1.8162e-01, -6.8265e-02, -2.5593e-02,
-4.3985e-02, -7.7293e-02,  1.3849e-01,
 3.9150e-02, -2.2740e-02, -2.8375e-03,
 1.7295e-03, -1.4858e-02, -1.1571e-01,
 9.6109e-02, -1.5898e-02,  8.1759e-02,
 3.3988e-02,  1.1577e-01, -8.5900e-02,
-3.6616e-02,  3.1361e-01, -2.8557e-01,
-6.1019e-02, -5.5631e-02, -1.1660e-01,
 1.6716e-01,  8.4352e-02, -1.7170e-02,
 9.3826e-02,  1.9888e-01, -2.3386e-01,
-4.8372e-02,  1.6010e-01, -2.2395e-01,
-3.2495e-02, -1.1872e-01, -2.1650e-01,
-2.5982e-02, -1.6514e-01,  1.6470e-01,
 1.1963e-01, -7.8737e-02,  2.6773e-01,
-2.0137e-03,  2.3267e-02, -3.1859e-02,
-1.6277e-01,  3.8948e-01, -5.9805e-02,
-1.4438e-01, -1.1968e-01,  5.0378e-02,
 1.0882e-02, -7.6071e-04,  3.1897e-02,
-7.5720e-02,  9.9576e-02,  8.3422e-03,
 3.6179e-02, -7.4355e-02,  8.1388e-03,
-4.4986e-02,  1.7028e-01, -7.9266e-02,
 1.7443e-01,  4.0729e-01,  9.2195e-02,
 1.1313e-01,  1.2725e-01, -7.8103e-02,
-4.5798e-02,  6.6788e-02, -2.3221e-02,
 4.3944e-02, -6.1240e-02, -9.6664e-03,
-3.7411e-02,  2.0436e-02,  4.9633e-02,
 2.0762e-02, -1.0137e-02, -1.1357e-01,
-1.4857e-02,  7.8273e-01,  1.1514e-02,
 4.8804e-02,  3.5619e-02, -2.7690e-03,
-6.4675e-02, -1.6808e-01, -8.8774e-02,
-1.6714e-02,  3.5433e-01, -1.9564e-01,
 1.0789e-01,  1.0121e-01, -3.5958e-02,
-6.6239e-03,  7.1949e-02,  5.7952e-02,
 6.9324e-03, -2.7786e-01,  6.5953e-02,
 2.2491e-01, -7.2195e-02, -2.0342e-02,
 4.2696e-03, -1.8118e-02,  1.8205e-03,
-6.0759e-02,  4.3643e-02,  1.5660e-02,
-3.6834e-03,  7.9892e-03,  2.0052e-02,
-3.7952e-05,  7.4826e-02, -2.8808e-02,
-1.9485e-01,  7.5924e-03, -1.6730e-01,
 1.3757e-01, -1.5543e-02, -1.1203e-02,
-5.6025e-02, -4.8173e-02, -8.6085e-02,
 3.4496e-03,  9.8236e-02, -1.0958e-01,
-3.7635e-02,  1.5892e-01, -1.1476e-01,
 7.7868e-04, -6.9660e-02, -2.3182e-01,
 8.5991e-02,  4.8977e-01, -1.7908e-01,
-1.5522e-01,  7.9325e-02, -1.5806e-02,
 7.1463e-02,  1.1943e-01, -2.4067e-03,
 6.9099e-03,  6.0242e-02, -2.2940e-01,
 1.7161e-02,  9.6983e-02, -8.0688e-02,
-5.3576e-02,  8.5033e-02, -1.1045e-03,
 8.9352e-03,  1.2065e-01,  5.4350e-02,
 1.7589e-02, -3.8841e-02,  1.3500e-01,
 1.9115e-02, -1.6216e-01, -1.8559e-01,
 7.0868e-02, -5.3147e-02, -1.1175e-01,
-7.8024e-02, -1.6385e-02, -2.4993e-03,
-5.6343e-02,  1.7676e-01,  1.5329e-01,
 1.9889e-01,  3.0045e-01,  1.8859e-01,
-1.3113e-01,  4.9262e-01,  1.6968e-01,
 1.0565e-01,  1.5241e-02, -8.8263e-03,
 7.8390e-02, -4.8326e-02,  1.4439e-02,
 1.0730e-01, -3.5532e-01,  2.6071e-01,
-4.4641e-12, -5.2414e-09, -1.2333e-07,
-5.7045e-16, -1.3984e-10, -1.6916e-08,
-4.5784e-23, -4.4501e-13, -7.0578e-10,
-1.6163e-37,  1.8633e-40,  5.1512e-40,
-3.4094e-40,  5.8054e-40, -3.3747e-40,
-2.9503e-40,  2.9029e-40,  5.7716e-40,
-1.4396e-02, -1.4397e-02, -1.4338e-02,
-1.4415e-02, -1.4416e-02, -1.4353e-02,
-1.4388e-02, -1.4387e-02, -1.4327e-02,
-1.7647e-04, -1.9035e-04, -1.8455e-04,
-1.9058e-04, -2.0375e-04, -1.9581e-04,
-1.8747e-04, -1.9759e-04, -1.8860e-04,
-4.8962e-40, -3.5385e-40,  1.1985e-39,
-2.2287e-25, -3.9257e-32,  3.5052e-40,
-2.8011e-19, -1.3703e-22, -3.7616e-30,
-1.1686e-06, -5.0055e-07, -2.5874e-07,
-3.7931e-06, -2.1740e-06, -1.3146e-06,
-9.2777e-06, -6.3053e-06, -3.9821e-06,
-1.5939e-18, -1.5905e-20, -1.0249e-19,
-3.7030e-18, -3.4481e-21, -3.2694e-21,
-1.1017e-15, -1.9946e-18, -1.6842e-19,
-7.1601e-29, -2.6591e-27, -2.3018e-24,
-8.3783e-29, -3.1441e-29, -8.0479e-27,
-3.6262e-27, -2.5435e-29, -2.6553e-28
}
,)"
R"(
{
 6.6916e-04,  3.1874e-03, -1.3746e-01,
-5.8591e-02, -7.9962e-03,  5.7624e-02,
-7.7515e-02, -1.7938e-01, -2.1427e-02,
 2.3039e-02, -5.6216e-02,  2.1235e-03,
 4.2272e-02,  1.6714e-01, -1.1290e-02,
 3.3933e-02, -2.2132e-01, -1.6782e-01,
-5.6241e-03,  6.3489e-02, -3.6694e-02,
-9.2586e-02, -2.5583e-01,  2.0139e-02,
-8.0616e-02,  5.3868e-04, -9.9902e-02,
 8.3209e-02, -1.4185e-02, -7.5864e-02,
-3.1386e-02,  1.4883e-01, -6.1055e-02,
-7.2472e-02, -1.0909e-01,  1.5856e-01,
-8.4186e-02,  1.0248e-01, -2.0782e-01,
-1.2961e-02, -1.4433e-01,  3.1481e-01,
-2.8774e-03, -5.7338e-02,  8.8795e-02,
-4.1130e-02, -1.6074e-02,  1.9476e-01,
 4.7276e-02, -3.1862e-01,  9.1747e-02,
 1.9878e-02,  2.2818e-01, -2.1946e-01,
 1.6648e-01,  1.5779e-01, -1.1912e-01,
 5.9292e-02,  4.1234e-02,  2.8380e-01,
 7.6076e-03,  2.0563e-01, -5.6308e-03,
 7.2333e-40,  2.4377e-40, -3.2660e-40,
-1.7861e-40, -2.7248e-40, -4.1720e-40,
-6.3801e-40,  5.7522e-40,  6.2413e-40,
 2.3327e-01, -1.6373e-02,  1.8010e-02,
 5.2898e-01,  2.5041e-01,  4.8556e-02,
 2.0423e-01,  1.0486e-01, -1.2002e-02,
-6.4276e-02,  4.6269e-02, -2.8957e-02,
-7.0038e-02, -1.4359e-01,  5.7887e-02,
-2.2039e-01,  4.0756e-02, -1.8146e-02,
-2.1943e-01, -1.4091e-03, -8.0816e-03,
-1.0759e-01,  9.4940e-02,  1.0448e-02,
-5.8642e-02, -8.1514e-04,  2.0113e-02,
-1.0992e-01, -7.0436e-02,  5.5159e-02,
-2.1142e-02,  2.5856e-01, -1.3805e-02,
 1.9735e-02, -7.1419e-02, -3.3746e-02,
 1.7696e-01, -8.2871e-02,  4.7601e-02,
-5.7652e-03, -4.7097e-02,  7.5677e-02,
-1.1239e-01,  5.7584e-02,  3.1041e-02,
-1.6584e-01, -5.9645e-02,  3.6213e-02,
-3.6680e-01,  5.1072e-01,  4.6738e-02,
-3.1461e-02,  7.2001e-04,  2.7050e-02,
-6.5480e-02,  1.5279e-02, -4.7300e-02,
 7.6355e-02, -5.6827e-03, -1.0108e-02,
-2.8602e-02,  1.7155e-02,  1.1751e-02,
 4.3101e-40,  5.5874e-40,  7.7870e-42,
 3.1765e-40, -6.0061e-40, -3.4353e-40,
-3.4398e-41, -2.4809e-40,  3.1455e-40,
-1.0845e-01, -1.3541e-01,  1.5133e-01,
 1.1232e-01,  4.7579e-01, -6.3373e-02,
-2.9119e-02,  8.6968e-02,  3.6654e-02,
-1.1404e-01, -1.6859e-01, -1.3337e-01,
 4.6555e-02,  1.4465e-01,  6.6170e-02,
-1.6504e-01, -1.5540e-01, -2.4383e-02,
 1.5666e-01, -6.2407e-03, -1.1982e-01,
-6.3504e-02, -3.5070e-01, -1.4981e-01,
 4.9697e-02, -5.9366e-02,  6.1797e-02,
 1.5323e-02, -4.8405e-02,  8.3473e-02,
 1.2738e-02, -2.0584e-01,  1.0494e-01,
 1.2859e-01,  1.5031e-01,  5.0233e-02,
 4.2827e-02, -3.6495e-02,  2.1167e-02,
 6.4251e-02,  6.4282e-02,  9.0435e-03,
 5.6229e-03, -4.6462e-02,  6.1973e-02,
 1.3571e-02, -1.8193e-02,  2.1148e-02,
-2.9388e-02,  5.7459e-02, -5.7339e-02,
 1.2938e-03, -1.4518e-01,  1.3707e-01,
-2.6571e-01,  1.7921e-01, -1.5642e-01,
-2.5606e-01,  2.5149e-01, -1.0273e-01,
 4.2861e-02,  1.5573e-01, -3.3024e-02,
-6.6349e-40, -4.2997e-40, -8.7169e-41,
-3.4965e-40, -5.7424e-40,  3.7821e-40,
-9.5405e-41, -5.7776e-41,  5.6324e-40,
 1.7365e-01, -6.1816e-04,  2.8153e-02,
 5.0195e-01,  3.1764e-01,  2.6392e-02,
 1.5545e-01,  1.1956e-01,  3.6044e-02,
-3.1175e-04, -1.0516e-01, -1.3141e-02,
-1.1094e-01, -5.5638e-02,  4.1683e-02,
-2.0962e-01, -1.6514e-01,  2.7874e-02,
 5.7102e-02, -4.8029e-02, -1.2106e-03,
-1.2301e-02, -1.6989e-01,  4.6618e-02,
 1.4376e-01, -8.1030e-02, -2.0732e-02,
 5.0249e-02, -2.0540e-01, -1.3633e-02,
 1.0441e-01,  2.7835e-01, -1.2144e-02,
 4.5190e-03, -2.6486e-02, -1.6136e-01,
 2.3597e-02,  3.2873e-02,  3.7778e-02,
-6.0634e-01,  3.6170e-02, -1.4957e-01,
 7.6106e-02,  9.3954e-02,  5.9653e-04,
-4.7616e-02, -1.8220e-01,  4.9518e-02,
-3.7849e-01,  3.1061e-01,  2.2303e-01,
-4.0407e-02,  2.2789e-02,  3.6510e-02,
 5.1013e-02,  1.1777e-02, -6.4170e-02,
-2.2829e-01,  1.2593e-01, -1.3307e-02,
-4.0193e-02,  4.0568e-02,  2.9582e-02,
 5.3023e-40, -5.4830e-40, -5.6332e-40,
 4.8252e-40, -6.0797e-40, -5.1008e-40,
 6.4786e-41,  1.6673e-40,  2.0072e-41,
-8.0840e-02, -1.1304e-02, -1.6804e-04,
-2.1749e-01,  1.1713e-01,  1.8655e-02,
 3.9590e-02,  1.1467e-01,  2.1573e-02,
 2.1089e-02, -4.2112e-02,  1.1003e-02,
 6.5696e-02,  2.4914e-02, -2.8910e-02,
 6.9880e-02, -1.0695e-01,  2.8039e-02,
 1.0180e-01, -3.3237e-02,  9.6463e-03,
-2.3275e-02, -1.2128e-01, -1.3483e-02,
 1.0640e-01, -6.7646e-02,  4.3052e-02,
 8.3757e-03,  3.5086e-02,  2.3912e-02,
-4.6668e-02, -1.1349e-02,  2.8631e-03,
 5.6926e-02, -6.5684e-02, -1.0438e-02,
-2.7303e-02, -6.8306e-04, -1.8756e-02,
 1.6269e-02,  9.3231e-02, -9.0054e-03,
-8.1785e-03, -4.7662e-02,  1.5793e-02,
 8.2955e-02,  1.3604e-01,  2.5437e-03,
 1.4088e-01,  3.5408e-01,  1.5981e-01,
-6.1164e-02,  4.1376e-02,  2.3071e-02,
 5.6543e-02, -7.0230e-02, -5.1061e-04,
-4.3378e-02,  1.8573e-01, -5.5072e-02,
-1.2826e-01,  5.9947e-03,  4.0032e-02,
-2.7511e-03, -2.7547e-03, -2.7544e-03,
-2.7615e-03, -2.7662e-03, -2.7641e-03,
-2.7526e-03, -2.7543e-03, -2.7498e-03,
 8.8398e-02, -1.3367e-01,  9.0045e-02,
-1.2815e-01,  3.4471e-01,  4.1071e-02,
 1.3754e-01,  1.5231e-01, -5.9524e-02,
 2.3592e-01, -1.0362e-01, -5.6021e-02,
-8.5598e-02,  8.1474e-04, -3.1670e-02,
-3.8302e-02, -9.6867e-02,  5.8445e-03,
 1.4156e-01, -1.6419e-01, -1.3725e-02,
-6.1081e-02, -1.1076e-01, -3.1046e-02,
-1.6219e-01, -1.5143e-01,  8.4526e-02,
-6.1728e-02, -1.5561e-01,  6.8060e-02,
-7.7470e-04, -6.9344e-03, -1.3557e-02,
 2.9175e-03,  2.0224e-01,  4.3769e-01,
 4.6725e-02,  6.7913e-02,  4.4001e-02,
-7.3108e-02, -1.3493e-01,  1.3377e-01,
-4.9855e-02, -1.4106e-01,  2.6349e-01,
 3.6703e-02, -1.4666e-01,  4.7796e-02,
-4.6490e-02,  2.8430e-01,  2.4963e-03,
-2.3183e-04, -2.0093e-01,  8.5570e-03,
-1.1347e-01, -1.4716e-02, -1.6818e-01,
-1.6624e-01,  3.4358e-02, -2.0266e-02,
-2.2011e-01, -1.2921e-01, -4.1327e-02,
-4.3218e-40, -7.2586e-41,  4.8737e-40,
-5.0920e-40,  5.8356e-40, -7.8120e-41,
 3.3402e-40, -5.8354e-40, -5.3726e-40,
 9.5864e-03, -5.5700e-02, -2.7364e-02,
-1.7002e-01, -9.0819e-02, -9.4548e-02,
 5.6840e-02, -5.1824e-02,  1.4496e-02,
 9.1589e-02,  3.2514e-02,  3.0873e-02,
-2.6288e-02, -1.9913e-02,  4.0007e-02,
-5.9960e-02, -3.8061e-02, -5.1952e-02,
-6.5286e-02, -5.6120e-02,  1.4958e-01,
-1.5597e-01, -2.6091e-01,  6.8860e-02,
 1.3734e-01, -3.0896e-01, -1.7176e-01,
 7.2767e-02, -1.4260e-01, -1.3498e-01,
 3.7019e-02,  5.9384e-01, -5.2426e-02,
 3.5504e-02, -1.9460e-01, -4.2871e-02,
 9.0278e-03,  1.9967e-02,  7.5002e-02,
-2.9409e-01, -1.9825e-01,  3.8907e-02,
-4.5449e-02,  9.4358e-02,  2.4319e-01,
-8.5869e-02, -9.6772e-02,  1.5619e-01,
 2.3927e-01,  2.9876e-02, -2.6634e-01,
-1.3765e-01,  1.7665e-01, -2.9822e-02,
 2.9357e-01, -8.0757e-02, -2.7216e-02,
 1.7248e-01, -3.9154e-02,  1.2115e-01,
-1.3457e-01,  9.4292e-02,  7.4378e-02,
 1.8297e-40, -4.3618e-40, -3.0663e-40,
 6.3062e-40, -7.5320e-42, -4.0657e-40,
 6.7007e-40,  5.5825e-41, -6.7431e-40,
-1.1116e-01, -1.3766e-03, -8.2527e-03,
-1.9320e-01,  2.9853e-01,  5.2457e-02,
 3.4868e-02,  4.7345e-03, -1.1185e-01,
-4.0646e-02,  1.2677e-02,  8.1386e-03,
-9.5312e-02, -4.0466e-02,  2.2961e-02,
-2.7473e-01,  5.9571e-02, -4.8755e-02,
-7.2570e-02, -1.2962e-01, -1.1112e-03,
 7.9843e-03, -1.7316e-01,  1.0072e-01,
 4.6139e-03, -5.5408e-03, -2.9563e-02,
 1.0518e-02,  1.3693e-01,  1.4649e-02,
 1.2504e-01,  2.8381e-01,  1.9572e-01,
 3.6317e-02,  5.5138e-01,  4.4150e-01,
-1.4639e-02, -1.5812e-01,  8.3472e-02,
-6.1992e-02,  4.7314e-02,  5.3164e-02,
-8.9509e-02,  7.1378e-02,  1.9687e-01,
 3.1318e-02,  1.9998e-01,  2.3854e-02,
 9.3472e-03,  2.8380e-01,  9.1518e-04,
-7.9959e-03, -4.0400e-01, -1.4127e-01,
 4.2359e-02, -1.5292e-01, -2.3145e-02,
 1.4650e-01,  5.8455e-02,  1.1363e-01,
 7.4617e-02, -3.8583e-01, -6.0218e-03,
 7.8247e-41,  1.7224e-40,  1.4807e-40,
 5.7355e-40,  6.2419e-40, -5.7187e-40,
-4.6999e-40,  4.2516e-40, -5.4200e-40
}
,)"
R"(
{
 9.7784e-02, -1.9047e-02, -6.0894e-02,
-3.7544e-02, -4.0501e-02,  4.1994e-02,
-9.2672e-02,  1.3116e-02, -1.5788e-02,
-1.1723e-01, -3.3813e-01,  2.0887e-01,
 7.3603e-02,  1.6658e-01,  1.9317e-01,
-6.0557e-02, -1.0727e-01, -1.9165e-01,
 1.7845e-01,  2.6790e-02, -8.9555e-02,
-9.0342e-02,  9.8401e-02,  4.9884e-02,
-1.3466e-01, -2.6843e-03,  1.9476e-02,
-8.3200e-03, -1.0432e-01,  3.5804e-03,
 3.1634e-02,  1.7915e-01, -4.3102e-01,
 6.0618e-02,  2.9186e-02,  4.0251e-02,
-1.2253e-01, -5.1791e-01, -7.5149e-02,
 2.1570e-02,  3.6669e-01,  1.7338e-01,
-2.2803e-03,  8.6139e-02,  6.8943e-02,
 1.7945e-01, -7.3459e-02, -1.5813e-01,
-6.5252e-02,  1.9154e-01, -2.1735e-01,
 3.5793e-02, -3.7080e-02, -2.3088e-03,
-2.5324e-02,  9.4003e-02, -7.9960e-02,
 1.2226e-04, -4.8867e-02, -2.2364e-02,
 5.4152e-02, -1.4416e-01, -2.2058e-02,
 1.6463e-01,  6.4898e-01,  1.4870e-01,
 1.3158e-01,  1.5831e-01,  1.7908e-01,
-1.4228e-02,  5.7026e-02, -2.1598e-02,
-1.3854e-01, -1.6929e-02, -8.8711e-02,
-1.2444e-01, -2.5047e-01,  1.6840e-02,
 9.6345e-02, -3.3850e-02,  7.2973e-03,
-3.1133e-02,  2.2157e-01, -6.5667e-02,
 1.5837e-02, -1.5439e-01,  9.8671e-02,
-1.5642e-02, -2.8515e-02, -9.2717e-02,
-3.1107e-01, -3.8604e-02,  1.5255e-02,
 1.0322e-01, -4.5346e-01, -1.4242e-01,
-8.0592e-03,  2.3989e-02,  2.7626e-02,
 4.7022e-02,  3.0699e-01,  7.6293e-02,
 5.0893e-02, -3.1222e-02,  1.0255e-01,
 1.0289e-02, -4.6691e-03, -4.9110e-02,
-2.1581e-02,  1.2146e-01, -1.1966e-01,
 3.3710e-02, -1.1276e-01, -3.2634e-02,
 5.4850e-02, -1.8770e-02,  8.6898e-02,
 1.8064e-01, -4.3036e-03, -1.2264e-01,
 5.4254e-02, -2.3117e-01,  1.5762e-01,
 1.0688e-02,  3.8983e-03, -6.0189e-02,
-3.9834e-02,  2.8950e-04, -9.6055e-02,
 1.1264e-02,  2.6691e-02, -8.8824e-02,
-1.3645e-01, -8.8574e-02, -6.3354e-02,
 2.5053e-01,  4.4169e-01,  1.5129e-01,
 3.0753e-02,  3.3052e-02, -1.4972e-02,
 1.1874e-02, -2.9042e-03,  1.8045e-03,
-4.0450e-02, -7.3257e-02,  6.0984e-03,
 1.5779e-01,  1.2274e-01,  2.9877e-02,
-1.2276e-01, -5.3066e-02, -1.1187e-01,
 2.5863e-02, -3.9279e-02,  9.9383e-02,
 6.3848e-02,  7.1416e-04,  3.0601e-01,
 4.3572e-03, -2.8575e-02,  3.0453e-01,
 1.0130e-01,  5.7500e-02, -8.0857e-02,
 3.5150e-01,  2.4868e-01,  3.6362e-02,
 6.0696e-02, -1.8366e-01, -2.8096e-02,
-4.4489e-03,  6.1247e-02,  9.3602e-02,
-1.2524e-01,  1.2172e-01,  9.0550e-02,
 1.7781e-02,  1.2525e-01, -1.7725e-01,
 9.7795e-02,  1.3382e-02,  8.2701e-03,
 8.8958e-02, -2.2619e-01,  8.0614e-02,
 4.9340e-02, -1.3889e-01,  4.7252e-03,
-3.2564e-01,  1.4407e-01,  9.9118e-02,
 3.2227e-01, -4.6118e-01, -1.6862e-01,
 1.7177e-01,  8.8038e-02, -1.7083e-01,
 1.1068e-02,  7.0344e-02, -2.1633e-02,
-1.7319e-01,  1.3617e-01,  6.2093e-02,
-4.7447e-03,  2.1185e-01, -9.0024e-03,
-2.0200e-01,  1.0025e-01,  1.2558e-02,
-1.2997e-01,  1.9939e-01,  9.2253e-02,
-1.2285e-01, -5.4702e-02,  5.8550e-02,
-2.5825e-02,  1.8127e-01, -1.2258e-01,
 1.3978e-01,  9.3431e-02, -8.5427e-02,
 9.0244e-02,  1.4144e-01,  8.3679e-02,
 5.3243e-03,  1.5575e-02,  1.9979e-02,
-2.9458e-02,  1.0636e-01,  2.3163e-02,
 2.8361e-03, -3.9566e-03,  1.3074e-01,
 3.8622e-02,  5.0942e-02,  3.1525e-02,
-2.6363e-01, -1.0051e-01, -4.2178e-02,
-9.3723e-02,  3.4198e-02, -5.0277e-02,
 5.8086e-02,  9.4981e-02, -4.4595e-02,
 1.1479e-01,  5.2060e-01,  4.6659e-01,
-3.7111e-02, -5.3580e-02,  3.0472e-01,
 1.8741e-02, -3.2509e-03,  1.7151e-02,
 1.7200e-01,  3.7055e-01, -3.8168e-01,
-3.1204e-02,  6.3878e-02, -2.2902e-01,
-7.4525e-02,  2.2562e-01,  3.5280e-02,
 7.4593e-02, -7.2742e-03, -1.0391e-01,
 6.6409e-02, -2.1706e-01,  2.2010e-02,
-4.4044e-02,  2.1769e-03, -6.4350e-02,
 1.0573e-01,  6.9533e-01, -8.7509e-02,
 1.8314e-02, -5.3836e-02,  1.5615e-01,
-3.6310e-02, -7.7659e-02, -3.3333e-02,
 3.7567e-03,  3.8268e-01, -5.6688e-02,
-4.8939e-02,  4.5634e-02,  2.9123e-03,
 2.9972e-02,  1.6921e-01, -6.0792e-02,
 2.7705e-01, -5.0319e-01,  4.1929e-02,
 2.2772e-03,  7.7286e-02, -3.7105e-02,
 1.7203e-02,  1.3806e-01, -9.6242e-02,
 3.6280e-02,  6.3583e-02, -4.4922e-02,
-1.7310e-02, -1.9669e-02, -9.0212e-02,
-1.0139e-01, -2.6661e-02,  1.1706e-02,
 4.7093e-02,  1.4123e-01,  1.3452e-02,
-3.4276e-02, -2.3062e-02, -1.8407e-03,
 5.0725e-02,  2.0265e-01, -2.4801e-02,
 4.5220e-02,  2.3345e-01, -4.3922e-01,
-6.6486e-02,  2.1262e-02,  6.9452e-03,
 3.1680e-02,  1.8098e-01, -4.6501e-02,
 1.2477e-01,  3.4334e-01,  1.7554e-01,
-6.6337e-02,  5.6915e-02,  6.6049e-02,
-3.8204e-02, -4.3336e-02,  2.2155e-02,
 4.6509e-02, -7.7310e-04,  1.3308e-02,
-3.3315e-02,  1.9923e-02,  1.3475e-03,
 5.5111e-02,  2.4898e-02, -2.5205e-02,
 9.3928e-02, -1.8523e-01,  6.9310e-02,
 1.3159e-03,  8.4789e-02, -8.4635e-02,
-6.0387e-02, -3.1829e-01, -3.0154e-02,
 1.0561e-01,  2.2876e-01,  9.0520e-02,
 2.7342e-03,  3.9745e-02, -2.0664e-02,
-9.9552e-02, -5.1992e-02, -5.4372e-02,
-3.2622e-02, -3.4757e-02, -3.7574e-04,
 9.4224e-02, -1.3429e-01, -9.4288e-03,
 1.0943e-01,  7.5219e-02, -1.2214e-01,
 5.3702e-02, -1.2986e-02, -1.6328e-01,
 1.3472e-02,  2.0016e-01,  7.1386e-02,
-8.3247e-02,  1.7942e-03, -2.4733e-03,
 1.4945e-01, -1.4328e-01, -3.8857e-03,
 3.4001e-02,  4.1785e-02,  6.8574e-02,
-4.8375e-02,  3.1940e-03, -3.1367e-02,
-2.0493e-02, -3.0046e-02,  1.9686e-03,
-7.4269e-02,  1.1141e-02, -2.3398e-02,
 7.8170e-02,  6.8755e-03,  1.2646e-01,
 1.0633e-01, -3.3736e-01, -1.1184e-01,
 9.1462e-03,  6.4286e-02,  4.4512e-02,
 7.8595e-02, -3.8588e-02,  4.6196e-02,
-1.1449e-01, -2.0696e-02, -8.0936e-03,
-8.9129e-02,  1.9441e-02,  7.0976e-02,
-5.9219e-02, -2.3846e-01, -1.3684e-01,
 1.2295e-01,  7.3844e-02, -1.8082e-01,
-1.6536e-02,  8.5913e-03, -8.3399e-02,
-7.8385e-02, -2.0103e-02, -9.2349e-02,
 7.4634e-02, -2.0519e-01, -2.0244e-01,
 3.2534e-02, -8.9362e-03,  7.9016e-03,
 5.9846e-02,  1.9507e-01, -7.9011e-02,
 1.0690e-01, -2.3081e-01,  1.8202e-02,
-6.9964e-02,  2.0835e-01, -2.4623e-02,
 9.9198e-02, -9.2372e-04, -4.8201e-02,
-1.9080e-01,  5.1900e-01, -1.7883e-01,
 1.4624e-01,  2.7797e-01, -5.6560e-02,
 9.0493e-02, -6.5958e-02, -2.4018e-02,
 1.3698e-01, -1.5705e-01,  3.3377e-02,
-1.0555e-01, -2.5638e-01,  9.5253e-03,
-6.1413e-02,  1.1354e-01, -4.1817e-02,
-1.0977e-01, -5.7464e-02, -3.9158e-02,
-7.7212e-03,  1.0934e-01, -4.3752e-02,
 1.1572e-01,  4.8016e-02, -4.0790e-02,
-4.2949e-01,  2.4564e-01, -4.0663e-02,
 1.7918e-03,  4.1289e-02,  5.3083e-02,
-6.2922e-02,  2.8137e-02,  8.4592e-03,
 1.6676e-01, -5.4313e-04,  3.3341e-02,
-5.2130e-02,  4.9988e-02,  5.9758e-02,
 8.6286e-02,  1.4753e-02, -6.2096e-02,
 1.4151e-02, -4.1282e-02, -4.8012e-02,
 2.3124e-01,  3.8359e-01,  4.4645e-02,
-6.7420e-02, -1.6922e-01, -6.0490e-02,
 2.2222e-01,  1.3750e-01, -2.7471e-02,
 2.6088e-02,  5.0216e-02, -3.6990e-02,
 1.1360e-02,  6.8630e-02,  1.8793e-03,
 8.3345e-03, -1.7497e-02, -1.6709e-02,
-1.5939e-02, -6.2635e-02,  5.9351e-02,
 4.3283e-02, -1.0031e-01,  3.7942e-02,
 2.2985e-02, -8.8837e-02,  4.5585e-02,
 3.4066e-02, -1.2268e-02,  2.8693e-02,
-2.5070e-03,  7.9133e-02, -1.6121e-02,
 3.8833e-02, -3.1626e-02,  2.7661e-02,
-7.6756e-02,  1.3693e-02, -7.3220e-03,
 2.1250e-02, -4.7744e-03, -5.4816e-02,
-4.2327e-02,  2.2655e-03,  9.4470e-02,
-4.8786e-02,  7.2471e-02,  4.7575e-02,
-5.0204e-03, -5.1643e-02,  2.1149e-02,
 4.9175e-02, -3.9962e-01, -1.1032e-01,
-4.5844e-02, -1.0527e-01, -2.3593e-02,
-7.4662e-03,  2.4249e-02, -2.8737e-02,
 5.7131e-03, -6.6286e-02, -8.2992e-02,
 6.4820e-03,  3.8478e-02,  2.5685e-02,
-3.9035e-02, -7.4120e-02,  2.7920e-02,
 9.3458e-02,  2.5062e-02, -2.9215e-02,
 1.3464e-02,  5.1123e-02, -3.4850e-02,
-4.0033e-02,  1.0196e-01, -2.1469e-02,
 4.2605e-02, -2.7854e-02,  1.4719e-02,
 2.6109e-03, -5.6908e-03,  3.6508e-03
}
,)"
R"(
{
 4.8630e-02, -8.9172e-02,  4.6337e-02,
 2.8268e-02,  9.8047e-02,  1.5658e-01,
 7.8416e-03, -1.5026e-01,  2.0157e-01,
 2.4071e-03,  2.6036e-02, -1.7064e-02,
-6.5329e-02,  4.7152e-01, -1.5500e-01,
 6.6754e-03, -1.1208e-01, -1.1162e-01,
 4.4313e-02,  7.9470e-02,  2.5780e-02,
 2.8891e-02, -4.4353e-02, -1.0263e-01,
 3.5067e-02,  8.3623e-02, -1.8216e-02,
-4.0875e-02, -8.2637e-02, -6.2643e-02,
-2.5788e-01,  7.0098e-01,  4.2836e-02,
 2.2808e-02,  3.1592e-02, -7.6566e-02,
 4.3100e-02, -7.4791e-03, -4.4467e-02,
 2.2115e-01,  1.4686e-02, -1.3166e-01,
 3.2884e-02, -1.9368e-02, -1.1356e-01,
 3.3410e-02,  1.4324e-01,  5.8267e-02,
-1.9208e-02, -2.4500e-01,  1.1541e-01,
-1.0755e-01, -1.1787e-01,  7.1888e-02,
 4.5671e-02,  3.3879e-01,  5.5718e-01,
-6.9208e-02,  3.4116e-01, -1.5428e-01,
-2.1182e-02, -3.5145e-02, -2.3381e-03,
-1.8944e-02,  5.2162e-02,  3.3387e-03,
-4.8255e-02,  3.2422e-03,  9.4447e-03,
 2.5041e-02, -6.1807e-02, -1.2532e-02,
-6.5116e-02,  1.0104e-01,  5.8329e-03,
 1.6618e-01,  5.8475e-01,  4.6639e-02,
-2.6750e-02, -7.0100e-01, -1.1373e-01,
 3.1822e-02,  2.5074e-02, -6.8359e-02,
 9.0624e-02, -3.3605e-01,  4.4291e-03,
-6.5916e-04,  1.5311e-01,  9.4043e-02,
 5.4092e-02,  7.8150e-02,  1.4170e-01,
 1.8750e-01, -5.5252e-01,  7.7271e-02,
 1.2427e-02,  5.1544e-02, -5.8065e-02,
 8.8531e-02, -6.5304e-03, -1.6447e-03,
-1.9046e-01, -5.7032e-02,  1.1281e-01,
-8.3402e-02,  1.0706e-01,  2.0333e-02,
-5.2373e-02,  8.4149e-02,  6.3704e-02,
 8.5006e-02,  1.4137e-01,  1.0755e-01,
 5.6383e-02,  2.4268e-01,  1.6673e-01,
 5.7432e-02, -6.0549e-02,  2.5547e-02,
-7.2361e-02,  1.6806e-01, -3.2085e-02,
-3.9550e-02,  4.5163e-03, -5.4008e-02,
 4.5266e-02,  1.5569e-01, -3.2417e-01,
-3.3272e-02,  4.2706e-01, -2.2220e-01,
-3.4094e-02,  5.9191e-02, -6.5790e-02,
 1.0894e-01, -2.0150e-01,  4.9361e-02,
-1.9819e-01, -4.7319e-01, -1.3847e-01,
 4.1939e-02, -1.3558e-01, -2.4314e-02,
 1.5104e-03,  3.1515e-02, -4.2673e-03,
-7.9536e-03,  4.4036e-02,  8.1869e-02,
-1.2593e-01,  1.1506e-02,  2.2099e-01,
 1.4164e-02,  4.5580e-02,  2.9136e-02,
-1.0906e-01, -1.8024e-01, -2.3132e-02,
 4.7779e-02, -1.8977e-01,  3.0783e-02,
-1.6571e-03, -1.3087e-01,  1.2155e-02,
-3.4348e-02, -2.8844e-01,  1.5684e-01,
-2.7862e-01, -2.0422e-01,  1.6415e-01,
 4.4581e-02,  1.2074e-01,  6.1221e-02,
-3.3887e-01,  3.9224e-01,  4.1968e-02,
-2.2734e-01,  1.5158e-01,  6.6736e-02,
-2.2258e-02,  1.5278e-02,  1.7112e-02,
 1.0957e-01, -3.1884e-01,  1.4005e-01,
 1.4714e-01,  1.5380e-02, -1.2091e-01,
-3.7984e-02,  1.1469e-02, -9.0876e-02,
 8.9330e-02,  2.3741e-02, -3.6958e-02,
 1.3397e-01,  3.9602e-02,  2.2722e-02,
 1.7311e-02,  5.5049e-02,  9.3754e-02,
-6.4392e-02, -2.4704e-01,  3.8942e-01,
 2.4790e-02, -1.0027e-01, -1.1778e-01,
 3.9200e-02,  5.1065e-03, -3.8833e-02,
-7.5144e-02, -6.5384e-02, -5.6415e-02,
 1.4553e-01,  9.7921e-02, -5.5712e-02,
-1.4536e-40, -1.7049e-40, -2.6781e-40,
 4.5792e-40,  1.4489e-40,  1.3645e-40,
-5.8774e-40,  1.1099e-39, -4.7571e-40,
-6.5733e-40,  1.5398e-40, -3.3819e-40,
-1.1286e-39, -1.9434e-40,  1.0949e-39,
-4.3830e-40, -2.8750e-40, -3.0788e-41,
 5.6364e-40, -7.3823e-40, -2.3064e-41,
 2.8909e-40, -5.8115e-40,  2.9852e-41,
-1.5354e-39, -7.5503e-41, -6.0335e-40,
 5.8073e-40,  2.9252e-40, -1.3038e-40,
 5.2260e-40,  3.8172e-40, -2.0389e-40,
-2.1905e-41, -1.6587e-39, -2.9226e-40,
 2.9957e-41,  2.6068e-40,  6.1324e-40,
-7.3109e-40,  5.1421e-40, -4.1157e-40,
 2.1416e-41, -1.6614e-40, -3.0843e-42,
-4.3402e-40,  2.8507e-40, -1.1086e-39,
 3.8826e-40, -3.0797e-40, -6.0685e-40,
 5.4170e-40, -6.1858e-40,  9.3049e-41,
 1.6486e-39,  1.6514e-39, -1.1228e-39,
 3.9906e-40,  1.2356e-40,  3.8682e-40,
 2.8630e-40,  6.2303e-40,  5.3034e-40,
-4.1904e-40,  4.8916e-40, -3.6125e-40,
-5.5393e-40, -2.4980e-40, -6.1877e-40,
 2.7289e-40, -1.8348e-40, -5.6663e-40,
-1.1160e-03, -1.9048e-02,  2.7932e-02,
 2.3827e-01, -1.5163e-02, -6.7794e-03,
-2.7252e-01, -1.1534e-03,  1.4023e-01,
-1.3734e-02, -1.3198e-02, -5.7866e-02,
-4.4905e-02, -4.7593e-01,  1.7182e-02,
-1.4121e-02,  4.4672e-01, -1.5422e-01,
 1.6291e-02, -5.5224e-02, -1.5263e-03,
 2.5221e-01,  7.4622e-01, -2.3549e-01,
 1.0224e-03, -6.3479e-02,  3.3447e-02,
-5.4867e-02,  2.1041e-01,  5.8950e-02,
-3.1168e-01,  6.6426e-01, -1.3169e-01,
-6.7254e-02,  4.7232e-02, -3.7582e-02,
 9.6481e-04,  6.6533e-02,  5.3115e-02,
 1.3278e-01, -2.2540e-01, -2.0519e-01,
 8.1310e-02,  2.3159e-01, -1.4576e-01,
-1.2053e-01, -4.4855e-02,  1.2747e-02,
-5.9526e-02, -5.8998e-01,  1.6517e-02,
-7.7819e-02, -1.1769e-01, -9.7912e-03,
 1.6962e-02, -4.6417e-02,  8.6952e-02,
 1.1399e-02,  8.8225e-02, -1.3862e-01,
-4.8989e-04, -3.7879e-02,  1.6314e-01,
-5.1371e-02,  8.4317e-02, -6.1146e-02,
-3.3001e-02, -1.6974e-01,  2.3739e-01,
-3.8907e-02,  6.8445e-02, -2.0595e-02,
-2.6452e-22, -4.9169e-22, -5.7570e-22,
-6.0472e-24, -1.3341e-23, -2.2894e-23,
-1.4110e-30, -8.8349e-30, -2.8389e-29,
 2.9766e-41, -7.8683e-42, -5.6890e-40,
-6.1051e-40,  4.9029e-40,  2.5225e-40,
 5.4218e-40,  1.8024e-40, -4.8963e-40,
 4.3682e-40,  3.1911e-40,  3.5009e-40,
 5.6306e-40, -3.5192e-40, -6.0928e-40,
-5.7555e-40,  6.3056e-40,  4.8433e-40,
-1.2238e-20, -7.3582e-19, -1.3574e-18,
-5.2996e-19, -1.9477e-17, -2.8850e-17,
-2.9711e-18, -9.7473e-17, -9.9928e-17,
-3.7514e-03, -3.7613e-03, -3.7465e-03,
-3.7825e-03, -3.7863e-03, -3.7666e-03,
-3.7732e-03, -3.7727e-03, -3.7469e-03,
-1.7221e-40, -5.1903e-41, -2.8142e-40,
-2.7502e-40,  6.3294e-40, -1.8771e-40,
-2.3254e-40, -3.0288e-40,  1.1340e-40,
-1.0186e-26, -8.9519e-30, -5.4814e-38,
-2.8715e-25, -4.4903e-28, -2.7283e-35,
-7.8450e-25, -1.4412e-27, -1.5330e-34,
-1.3913e-05, -1.4631e-05, -1.3809e-05,
-1.4435e-05, -1.5263e-05, -1.4467e-05,
-1.3835e-05, -1.4658e-05, -1.4004e-05,
-7.2681e-41, -5.4331e-40,  2.4069e-40,
 1.0755e-40,  3.0446e-41, -5.1764e-42,
 2.3158e-40, -4.2360e-41, -2.8033e-40,
 5.3231e-41, -2.1838e-40,  6.4604e-40,
 2.2854e-40, -3.7350e-40, -5.0873e-40,
-4.4484e-40,  2.3254e-40, -6.5618e-40,
-5.3194e-40,  3.4724e-40, -4.7140e-40,
 4.0657e-40, -1.6635e-40, -2.9163e-40,
 4.6927e-40, -2.9305e-40,  3.6521e-40,
-1.7306e-31, -3.9220e-31, -3.5953e-30,
-4.6542e-31, -1.3715e-30, -9.7942e-30,
-1.5070e-30, -3.7862e-30, -2.3737e-29,
-1.0303e-03, -1.0329e-03, -1.0266e-03,
-1.0359e-03, -1.0388e-03, -1.0323e-03,
-1.0326e-03, -1.0347e-03, -1.0283e-03,
-2.1772e-41,  7.7555e-40, -9.0763e-40,
 3.2977e-40, -3.6690e-40, -8.3013e-40,
 9.4214e-40,  8.6701e-40,  6.0800e-40,
 3.3118e-40,  1.4790e-40, -8.4218e-42,
-5.4264e-40, -2.5763e-41,  6.2564e-40,
 3.8417e-40, -2.8680e-40, -3.7140e-40,
-4.3426e-40,  4.6252e-40,  3.9059e-40,
-2.4663e-40,  5.8737e-40,  5.5561e-40,
-3.1436e-40,  4.2678e-40, -3.2594e-40,
-4.9904e-02,  8.1412e-02,  7.8025e-03,
-1.9203e-02,  5.9916e-01,  4.9018e-02,
 8.4018e-02,  1.7758e-01,  1.0199e-01,
 4.1824e-02,  2.9905e-02, -7.4231e-04,
 9.1549e-02, -3.2007e-01, -8.6624e-02,
-1.2584e-02,  1.7021e-02, -4.4427e-02,
 5.7425e-02,  3.5835e-02,  1.0865e-01,
 1.8407e-01, -4.1361e-02,  2.2999e-01,
 3.7464e-02,  4.6411e-02, -1.5094e-02,
 6.9684e-02,  1.8875e-02, -3.6031e-02,
 1.7938e-01,  2.1605e-01, -2.5508e-02,
 1.7732e-01,  4.3565e-03, -1.8596e-02,
-4.2989e-02,  1.2750e-01, -1.4205e-02,
-1.9800e-01,  7.7319e-01, -4.9982e-02,
-4.7642e-02, -3.8361e-01, -1.8626e-01,
-1.8040e-02, -7.6084e-02, -8.9504e-03,
-1.0737e-01,  8.5186e-02,  4.4397e-02,
-3.8916e-02, -6.3372e-02,  1.3771e-01,
 1.5555e-02,  2.0737e-02,  9.8389e-03,
-2.1211e-02,  2.1237e-02,  4.1355e-01,
 1.1352e-03, -6.8324e-02,  5.6878e-02,
 1.9880e-02, -1.4545e-02, -1.7763e-02,
 9.4337e-02, -9.7670e-02, -4.9010e-02,
-3.8277e-02,  6.1473e-02, -4.2705e-02
}
,)"
R"(
{
-2.7601e-23, -2.9478e-22, -2.4663e-21,
-1.7588e-23, -3.5435e-22, -8.7366e-21,
-3.5920e-23, -1.8289e-22, -3.8856e-21,
-4.9467e-03, -4.9694e-03, -4.9524e-03,
-4.9517e-03, -4.9726e-03, -4.9571e-03,
-4.9017e-03, -4.9228e-03, -4.9126e-03,
-6.3339e-40, -3.4528e-40, -6.1200e-41,
 5.3523e-40, -1.6610e-40, -2.1308e-40,
-1.3293e-40, -5.8800e-40,  3.6412e-40,
-4.8855e-40, -4.8867e-40, -5.0492e-40,
-1.0706e-40,  5.3827e-40, -1.6413e-40,
 1.4714e-40, -3.4024e-40, -4.4881e-40,
-3.9098e-37, -3.9805e-31, -2.2274e-26,
 6.0786e-40, -1.2884e-32, -3.4611e-26,
-3.1292e-40, -1.4385e-33, -4.6258e-27,
 3.6248e-40, -4.7200e-40,  4.0264e-40,
-4.6002e-40,  2.6194e-40, -3.1839e-40,
 4.2398e-40, -2.7994e-40,  2.9860e-40,
 5.2692e-40, -6.3449e-41,  1.2075e-40,
-9.5288e-40, -6.3852e-40,  4.9139e-40,
-1.8284e-39,  3.1821e-41, -1.6779e-39,
-5.7836e-15, -1.7863e-15, -8.3606e-16,
-7.7591e-14, -4.3864e-14, -1.5946e-14,
-2.8127e-13, -2.3124e-13, -7.6219e-14,
-8.3932e-02,  5.8782e-02, -3.6003e-03,
 6.3018e-01,  1.0565e-01,  2.9311e-02,
-8.2006e-02,  4.4002e-02, -1.3558e-02,
 4.7562e-02,  1.1286e-03,  1.4443e-02,
-4.4195e-01,  2.8154e-01,  3.6407e-03,
 7.9056e-02,  4.5342e-03, -5.9810e-03,
-6.2995e-01, -2.3460e-02, -2.6125e-02,
-2.8355e-01, -8.3691e-02, -3.0387e-02,
 2.1255e-02,  2.3585e-02, -1.8388e-03,
 6.0313e-40, -1.6594e-39, -1.1380e-39,
 6.0697e-40, -9.1199e-41,  5.8965e-40,
 5.4830e-40,  1.3014e-40,  1.5585e-41,
-3.0765e-03,  1.0080e-02,  1.4938e-02,
 6.1626e-01,  3.8819e-02, -1.3347e-02,
 2.4983e-02, -8.0010e-02, -5.0512e-03,
-3.3025e-40,  3.9116e-41, -6.5568e-40,
-5.1763e-40,  4.6585e-40,  6.5962e-40,
 2.6868e-40,  2.2424e-40,  1.0088e-41,
 6.4703e-40,  2.7744e-40,  6.2542e-40,
-4.2216e-40, -8.7287e-42, -1.9778e-40,
 4.1122e-40, -2.7964e-40, -5.6069e-40,
 8.6316e-03,  1.2037e-02,  1.1042e-02,
-4.6705e-01,  1.4339e-01,  3.1422e-02,
 4.0571e-02, -4.5762e-03,  1.0131e-02,
-6.8064e-02,  9.5304e-02,  1.8808e-02,
-1.6764e-01,  1.6124e-01,  1.6834e-01,
-1.4986e-01, -1.1914e-01,  5.9416e-02,
 1.0864e-01,  2.2705e-01,  3.4855e-02,
 1.7589e-01,  3.8173e-01,  6.4160e-02,
 7.0264e-02,  1.0488e-01, -7.8918e-02,
-3.6711e-02, -2.3689e-02,  8.0383e-02,
 5.8510e-02, -1.0734e-01,  1.0463e-02,
 1.6342e-03,  2.3035e-02, -1.0289e-02,
 2.4428e-40, -3.0160e-40,  2.3184e-40,
-4.9114e-40,  5.6685e-40, -3.6020e-40,
 2.2618e-40, -2.8145e-40,  2.1149e-40,
 3.2852e-02, -5.0214e-02, -4.5507e-02,
-2.3273e-01,  6.5087e-01, -8.0622e-02,
-3.0102e-02, -1.5834e-01, -5.2238e-02,
-3.0471e-03, -3.0745e-03, -3.0454e-03,
-3.0729e-03, -3.0988e-03, -3.0679e-03,
-3.0430e-03, -3.0677e-03, -3.0348e-03,
 5.5421e-04,  5.6626e-04,  5.5029e-04,
 5.6662e-04,  5.7921e-04,  5.6129e-04,
 5.5245e-04,  5.6391e-04,  5.4715e-04,
-1.2740e-01, -3.3900e-01, -9.7352e-02,
-4.3173e-02,  6.4354e-01, -1.3650e-01,
 2.2516e-02,  8.4502e-02, -1.0792e-02,
 3.0709e-40, -4.4880e-40, -7.1445e-40,
-7.4730e-40, -5.3936e-40,  5.0995e-41,
 4.4003e-40,  7.0515e-40,  7.5467e-40,
-1.8989e-40, -3.6631e-40,  4.1392e-40,
-3.9057e-40, -5.5599e-40,  6.9979e-41,
 3.8983e-40,  5.6737e-41,  2.3997e-40,
-9.4862e-41,  9.7546e-40, -3.7040e-40,
 1.6374e-40,  3.5439e-42, -1.0385e-40,
 3.6145e-40, -2.4342e-41, -3.0115e-40,
-6.0009e-40, -5.2386e-41, -1.2504e-40,
-1.2403e-39, -1.2290e-40, -1.1502e-40,
-3.5887e-40, -1.1137e-39, -1.6289e-41,
-6.7264e-40,  5.1229e-40, -2.4915e-40,
 1.3516e-40, -8.8110e-40,  8.5831e-41,
-6.4868e-40,  3.7625e-41,  8.3851e-40,
-1.8406e-39,  1.4120e-39, -1.8386e-39,
-5.3110e-40, -1.9669e-39, -5.8162e-40,
 1.3634e-39, -5.5139e-40, -1.8819e-39,
 4.8855e-40, -9.3940e-41, -1.1209e-39,
 8.8389e-40, -2.4982e-40, -1.2956e-40,
-6.0047e-40, -1.8712e-41, -7.3274e-42,
-2.8519e-40, -8.1457e-40, -7.8957e-40,
-6.4499e-40,  7.1196e-40,  7.1206e-41,
-6.9177e-40, -2.5552e-40, -3.6189e-40,
 8.6656e-03,  7.9198e-05, -8.0448e-03,
 2.9554e-02, -1.8275e-01,  6.6735e-02,
 1.6740e-01, -8.8421e-02, -1.2086e-01,
-3.7190e-02, -4.5721e-02, -8.4016e-03,
-1.8560e-01, -3.8216e-01, -3.6438e-02,
-3.2674e-01, -6.6408e-01, -1.3566e-01,
-9.3963e-03, -7.4759e-03, -1.3881e-02,
 2.3513e-02,  1.4643e-02, -2.9517e-02,
 1.3060e-01,  9.6742e-02, -1.7671e-01,
 8.7873e-41, -5.3947e-40, -1.1071e-39,
-9.5264e-40,  1.1055e-39, -5.4012e-40,
-6.5981e-40, -1.4913e-40, -9.8365e-40,
-1.2340e-02, -3.0713e-02, -8.6609e-03,
 1.2714e-02,  1.5799e-01,  5.8613e-02,
 2.6072e-01, -2.3820e-01, -5.2958e-02,
-2.0473e-41, -5.2353e-40,  6.2201e-40,
-1.7993e-42, -5.9375e-40, -1.1848e-40,
 1.8033e-40,  2.4967e-40,  4.5004e-40,
 5.2880e-40,  5.9499e-42, -1.8992e-40,
 7.0975e-40, -3.0171e-40,  3.6209e-40,
-4.4059e-40,  4.4493e-40,  8.5092e-40,
 9.9361e-02,  8.9522e-02, -5.1759e-03,
 1.5998e-01,  2.8857e-01,  1.5367e-01,
 1.0982e-01, -6.4722e-03,  1.9513e-01,
 4.9852e-03,  5.4767e-02, -1.2245e-03,
-1.3020e-01, -4.8171e-01,  1.2271e-01,
 3.9661e-01, -1.3532e-01,  1.8635e-01,
 1.2134e-03,  2.9382e-02,  8.5669e-03,
-1.0185e-02, -4.3477e-01,  1.9047e-01,
-1.2247e-01,  3.0793e-01,  2.3281e-02,
-2.9184e-01, -7.8450e-02, -4.1383e-02,
-1.7024e-01, -1.4567e-01, -2.9762e-02,
-1.7446e-01, -2.5608e-02, -2.9202e-02,
 1.0260e-40,  5.0443e-40,  7.5150e-41,
 1.4402e-40, -5.1952e-40, -5.3810e-40,
 6.2240e-40,  1.6411e-39, -8.2983e-41,
 6.6958e-02,  1.4965e-02, -1.1216e-02,
 4.9588e-01,  2.1931e-01,  1.4016e-02,
 2.9691e-01,  2.4926e-01, -4.7425e-02,
-3.1058e-40, -3.4470e-40, -3.3421e-42,
 5.4501e-40,  2.7579e-40, -1.8800e-40,
 2.9392e-40,  3.4801e-40,  2.4611e-40,
-3.7456e-40,  6.3336e-40,  4.3223e-40,
-3.9469e-40,  3.8567e-40, -4.0730e-40,
 5.0237e-40,  6.1562e-40, -1.6683e-40,
-6.6609e-04,  3.1514e-02, -2.2454e-02,
-4.4336e-02,  5.1816e-01, -9.6633e-02,
-2.0691e-01,  8.4879e-02,  2.1960e-02,
 3.6063e-01, -3.6326e-01,  8.1607e-02,
-2.9820e-01,  1.6157e-01,  1.9347e-01,
 9.9297e-03, -2.5680e-03,  5.3031e-02,
-4.3722e-02, -5.4516e-01,  1.8536e-01,
 1.1152e-01,  1.6556e-01,  1.8273e-01,
-5.4415e-02,  6.1375e-02, -7.6135e-02,
 9.1995e-02, -4.6836e-01, -6.3725e-02,
-6.3396e-02, -1.8315e-02, -2.8792e-02,
 2.9103e-03,  7.3191e-03, -4.1617e-03,
 1.5943e-40,  4.6894e-40, -1.1229e-39,
 3.8054e-40, -3.7914e-41, -1.4429e-40,
 1.2213e-39,  5.1566e-41,  1.6644e-39,
-9.0813e-02,  2.4277e-01,  4.8625e-02,
-1.9282e-02,  3.3320e-01, -2.4325e-02,
 6.2311e-02, -6.3914e-02, -3.2129e-02,
-2.9134e-40, -3.3065e-40, -5.8882e-40,
 6.9981e-42,  1.3416e-40, -3.0427e-40,
-6.1465e-40,  3.1821e-40,  1.6648e-40,
 1.9943e-40,  2.5324e-40, -4.9674e-40,
 6.2842e-40, -3.2508e-40, -5.5499e-40,
-2.0781e-40,  4.2796e-40, -5.4990e-40,
-1.2391e-01,  3.1037e-01,  3.8953e-03,
-3.6695e-02,  9.1093e-02, -4.9451e-02,
-1.7936e-02,  4.6449e-03, -3.0355e-02,
-5.0399e-02, -1.6834e-01, -4.0599e-02,
 4.6910e-03,  3.8730e-01, -4.4896e-02,
 9.0825e-02, -1.3378e-01,  2.7535e-01,
 3.8422e-02,  9.6377e-02, -2.8826e-02,
 4.5983e-02, -6.6602e-02, -4.3081e-02,
-3.3928e-02, -1.1789e-01,  9.2436e-02,
 8.5005e-02,  5.2767e-02,  4.5970e-02,
 1.7717e-01, -2.8642e-01, -1.0588e-01,
-1.1895e-01, -4.0862e-03,  1.5838e-02,
-1.1963e-39, -1.4451e-39,  1.4443e-40,
-1.4282e-39, -2.5643e-40,  1.7360e-39,
 1.2203e-39,  1.1080e-39, -1.1106e-39,
 9.9446e-02,  1.0044e-01,  4.1131e-02,
-2.0167e-01, -4.5740e-01, -1.0046e-01,
 2.9519e-02,  2.7720e-01, -8.3477e-02,
-5.0335e-40,  6.2541e-40, -2.5459e-40,
-8.7892e-41,  5.9152e-40, -1.2438e-40,
 6.4047e-40,  5.2690e-40, -5.6096e-40,
-3.8469e-40,  4.3452e-40,  2.4821e-40,
-5.8765e-40, -1.3818e-40,  2.4208e-40,
-2.6909e-40,  6.0867e-40, -4.6797e-40,
-8.9935e-02, -2.5639e-01,  4.1687e-02,
 2.6106e-02,  5.7348e-01, -1.3483e-01,
-7.1229e-02, -8.3123e-02,  9.5152e-03
}
,)"
R"(
{
-1.7030e-04, -1.7801e-04, -1.7782e-04,
-1.7562e-04, -1.8386e-04, -1.8409e-04,
-1.7350e-04, -1.8101e-04, -1.8121e-04,
-2.1963e-02, -9.1402e-02, -1.0783e-02,
 8.3746e-03, -1.2897e-01,  7.7220e-02,
 1.1813e-02,  2.3293e-01,  3.1628e-02,
-3.2281e-02, -2.2667e-02, -1.1959e-02,
 6.4867e-02,  6.3450e-02,  1.9475e-02,
-3.2837e-02, -5.8163e-02,  8.1752e-03,
-1.7867e-39, -1.6915e-39, -1.8090e-39,
 1.5026e-39,  1.9332e-39, -1.6506e-39,
-1.3315e-39,  1.9463e-39,  6.4184e-41,
-9.9213e-02, -7.4952e-02,  3.8240e-02,
-1.0787e-01, -1.6454e-01, -4.7653e-02,
-2.3062e-02,  3.9521e-02,  2.9152e-02,
-1.3482e-02,  3.3519e-01,  1.4952e-02,
 2.4941e-02, -3.6220e-01,  6.3129e-02,
-2.7399e-02, -2.0148e-02,  9.6434e-03,
-1.4102e-02,  7.9000e-03,  1.2194e-02,
 4.8655e-02,  3.5432e-01,  2.0344e-03,
 1.6922e-02, -6.7593e-01, -8.5588e-02,
-1.1348e-01,  1.1932e-01, -2.4829e-03,
-1.0296e-01, -1.2199e-01, -9.8870e-02,
 2.0685e-02,  2.4240e-03, -6.8815e-03,
 2.8159e-04,  2.8243e-04,  2.7540e-04,
 2.9137e-04,  2.9369e-04,  2.8629e-04,
 2.8797e-04,  2.9135e-04,  2.8554e-04,
-9.2451e-03, -3.1369e-02, -4.0439e-02,
-9.4420e-04,  1.3384e-01,  1.0062e+00,
-6.5964e-03,  1.4390e-01, -1.3451e-01,
-2.2807e-02, -6.2464e-02, -1.8231e-02,
 7.0337e-02,  3.6995e-02,  6.4366e-03,
 5.0541e-02,  8.9391e-02,  1.7841e-02,
-1.7797e-39,  1.9402e-39, -8.6062e-40,
-1.0814e-39, -6.3131e-40,  1.8613e-39,
-1.4312e-39,  1.8398e-39,  1.8363e-39,
-4.6341e-02, -1.9174e-01,  6.4808e-02,
-2.0698e-02, -8.3370e-02, -4.8120e-02,
-1.7636e-02,  1.7017e-02, -2.3824e-02,
 1.1969e-02,  7.3750e-02,  7.2870e-02,
 1.4583e-02, -3.8024e-01,  5.9004e-02,
 6.9605e-03,  1.6570e-02,  4.2008e-03,
-1.8893e-03, -1.1631e-02,  8.4983e-03,
-1.3305e-02,  3.8823e-02, -6.5228e-02,
 5.3922e-02,  8.7719e-02, -4.4295e-02,
 7.7201e-02,  1.7585e-01,  2.0431e-03,
 3.2863e-02,  2.4426e-01,  4.3204e-03,
-2.0786e-02, -2.0963e-02, -3.1135e-03,
 5.2987e-05,  5.6358e-05,  5.7226e-05,
 5.5603e-05,  5.9397e-05,  6.0480e-05,
 5.5672e-05,  5.9281e-05,  6.0167e-05,
 4.9329e-03, -2.6131e-01,  5.6117e-03,
 2.8945e-02, -2.0056e-01,  1.2391e-01,
-1.8745e-03, -8.5052e-02, -5.6251e-02,
-1.1464e-01, -1.1181e-02, -1.8485e-02,
-8.1217e-02,  1.5121e-01,  9.6975e-02,
-5.1080e-04,  9.2377e-02,  8.7422e-03,
 1.9157e-39,  1.4731e-39, -1.3924e-39,
 1.8676e-39,  1.4593e-39, -1.3724e-39,
 1.8496e-39,  1.9136e-39,  1.9222e-39,
 1.3907e-01, -3.3731e-01, -1.0607e-01,
 3.6079e-03, -3.7843e-03,  3.5740e-02,
 1.7694e-02, -2.0527e-02, -1.1485e-02,
-5.9167e-02,  5.7577e-01,  1.5103e-02,
-6.0704e-03,  9.8473e-02, -2.2132e-01,
-3.3520e-03,  2.4323e-02,  1.4015e-02,
-6.7180e-03, -4.5596e-02, -5.3968e-03,
-9.3759e-02,  3.0142e-01,  3.0558e-02,
-3.4453e-02,  4.9623e-01,  1.0314e-01,
-5.4311e-03,  5.3082e-01, -6.0131e-02,
 2.9350e-02,  3.9440e-02, -1.4921e-02,
 2.8675e-03, -3.4458e-02, -7.8473e-03,
-3.9511e-03, -3.9751e-03, -3.9682e-03,
-3.9779e-03, -3.9953e-03, -3.9825e-03,
-3.9637e-03, -3.9783e-03, -3.9555e-03,
-4.5388e-03, -2.0374e-04, -2.1449e-02,
-9.3094e-03,  1.4560e-01, -1.7497e-01,
-7.5432e-03,  5.9122e-02,  8.3586e-03,
 3.7679e-02,  2.2169e-02,  5.0596e-03,
 7.5862e-02,  7.6727e-01,  1.3185e-01,
 1.9514e-02,  1.7012e-01,  2.5230e-02,
-9.4584e-40,  1.2671e-39,  1.2343e-39,
 1.6123e-39, -1.4324e-39,  8.8338e-40,
 1.2233e-39,  1.6197e-39, -1.3383e-39,
-1.6199e-01, -6.5513e-01, -1.6543e-01,
-2.2482e-03, -2.3622e-01, -4.7737e-02,
-1.6724e-02, -3.2477e-03, -1.3467e-02,
 4.2102e-02, -4.4714e-02,  8.6366e-02,
 1.7695e-02, -1.0747e-01, -5.7324e-02,
 7.5031e-03,  2.9247e-02, -4.3777e-03,
-8.6118e-04, -3.5731e-03,  1.5803e-03,
 3.2900e-02,  1.7115e-01, -1.9986e-02,
 2.7273e-02, -2.5018e-01,  1.2617e-02,
-4.5737e-02, -1.3041e-01,  1.4373e-02,
-2.4929e-02,  1.4655e-01,  5.6383e-03,
-1.6393e-02, -2.6611e-02, -1.0177e-02,
-4.2219e-04, -4.3646e-04, -4.3734e-04,
-4.1915e-04, -4.3393e-04, -4.3484e-04,
-4.1658e-04, -4.3023e-04, -4.3046e-04,
-1.1254e-02,  5.0225e-02, -8.0103e-03,
-1.3289e-02, -6.1826e-02,  2.8144e-01,
-7.0378e-03,  4.9404e-02, -4.6551e-02,
 2.1752e-02,  5.6104e-03,  9.7653e-03,
 4.1534e-02,  1.0439e-01, -4.4648e-02,
 1.5986e-02, -2.0438e-01,  4.8541e-02,
-2.0550e-40,  1.6684e-39,  3.3927e-40,
-7.2768e-40, -1.1124e-39,  1.8866e-40,
-1.0459e-39, -1.8012e-40, -1.9375e-40,
-7.5320e-02, -1.2971e-01, -7.3764e-03,
-5.7139e-02,  1.7700e-01,  6.5755e-03,
-7.0020e-03,  1.9964e-02, -3.8845e-02,
 5.4632e-03, -8.2337e-02, -6.9325e-02,
 2.6028e-02, -1.1532e-01,  4.0519e-01,
-6.4132e-03, -2.2109e-02, -4.4191e-02,
-7.1864e-03,  7.3690e-03, -4.3123e-02,
 2.9824e-02, -6.6124e-02,  3.1618e-02,
 2.1370e-02,  5.2056e-01, -2.5340e-01,
 1.0395e-02,  9.0949e-02, -2.4524e-02,
-1.6157e-03, -4.0161e-01,  3.2454e-02,
 7.2396e-03,  3.3799e-02, -1.0605e-02,
 3.1747e-05,  3.3569e-05,  3.4100e-05,
 3.2186e-05,  3.4045e-05,  3.4635e-05,
 3.2069e-05,  3.3785e-05,  3.4242e-05,
-1.0868e-02,  1.5624e-01, -2.2295e-02,
-2.0956e-02, -2.3542e-01, -1.9170e-01,
 2.3031e-02,  4.0011e-02,  5.1282e-02,
 5.4014e-02, -4.1220e-02,  1.8062e-02,
-1.1920e-01,  1.4186e-01, -5.8273e-02,
 5.0756e-02, -5.1149e-02,  3.3995e-03,
-7.9099e-40,  1.8640e-39, -9.6169e-40,
 1.4879e-39, -1.6956e-39, -8.4045e-40,
 1.4776e-39,  1.4631e-39,  1.9440e-39,
-6.9009e-02, -8.2243e-02, -6.9876e-02,
-5.2119e-02, -5.7577e-02, -7.7784e-02,
-3.7031e-02,  3.4148e-02,  1.0635e-02,
 2.4123e-02, -2.0254e-01, -3.9541e-02,
-2.4560e-02, -1.7441e-01,  2.1639e-01,
 7.1232e-04, -1.9246e-02, -9.7799e-03,
 1.4446e-03,  4.9154e-02,  1.0916e-02,
 5.2467e-02, -2.7957e-02,  6.0836e-03,
-8.5302e-02, -6.2757e-02, -1.1875e-01,
-1.1159e-01, -2.2404e-01,  8.4596e-02,
 1.0521e-01, -4.3003e-03,  1.6123e-02,
-2.1979e-02,  2.2875e-02,  1.3608e-02,
 9.1781e-41,  6.3142e-40,  1.8105e-41,
-5.7515e-40, -4.0824e-41, -5.9401e-40,
 5.3982e-40,  6.1891e-40, -2.9076e-40,
-2.3906e-02,  4.8018e-02, -3.5115e-02,
 2.0623e-02, -1.3587e-01,  3.6185e-01,
 4.0692e-03, -2.1100e-01, -4.4865e-02,
 2.8443e-02,  7.6749e-03, -6.6405e-03,
-9.0447e-02,  1.4240e-01,  3.7064e-02,
-2.7875e-02, -7.6721e-02, -1.7431e-02,
-1.5269e-39,  1.7853e-39,  1.6319e-39,
 8.6300e-40, -1.9435e-39,  1.0505e-39,
-1.9205e-39, -1.4405e-39, -1.4605e-39,
 5.4050e-02, -2.7651e-01, -2.2297e-02,
 2.5076e-02,  8.1837e-02, -5.9649e-02,
-1.2941e-02, -2.2492e-02, -1.1692e-02,
 1.4202e-02, -1.7460e-01,  1.0129e-01,
-6.5366e-02,  6.1277e-01,  7.9750e-03,
 1.6380e-02, -5.8905e-02, -2.3304e-02,
-1.0168e-02,  3.0121e-03, -3.4731e-02,
-2.6095e-03,  1.3773e-01, -7.1156e-02,
-1.0205e-01,  1.9643e-01,  2.9787e-03,
 3.5814e-03,  3.2795e-01, -2.0661e-02,
-7.8123e-02, -2.5496e-01,  3.0124e-02,
 2.3271e-03,  2.5998e-02, -1.3433e-02,
-1.1224e-04, -1.1463e-04, -1.1232e-04,
-1.1888e-04, -1.2158e-04, -1.1896e-04,
-1.1886e-04, -1.2179e-04, -1.1917e-04,
-5.4132e-03, -5.6992e-02,  4.8689e-02,
 6.0217e-03, -1.1068e-02, -4.8363e-01,
 1.6282e-03,  3.3312e-02,  8.4505e-02,
-2.5457e-02,  2.8920e-02,  9.7365e-03,
-1.3482e-02,  1.6856e-01,  1.7464e-02,
 3.1570e-02, -8.8221e-02, -1.5533e-02,
 1.8624e-39, -8.9462e-40,  1.8351e-39,
 1.8924e-39,  1.8453e-39, -9.1739e-40,
-9.4562e-40,  1.9765e-39,  1.8266e-39,
 5.1745e-02, -3.0180e-01, -3.7745e-02,
-4.6576e-02,  8.0404e-02, -2.1534e-02,
-1.2980e-02,  3.0678e-02, -3.7070e-03,
-1.9083e-03,  1.6389e-01, -6.7500e-02,
-1.9402e-04, -9.7902e-02,  9.7186e-02,
-7.9226e-05, -6.4229e-03, -9.4834e-03,
 8.1945e-04,  1.0264e-02,  2.0650e-03,
-2.0524e-02,  2.3688e-01,  3.7237e-02,
-1.3540e-02,  1.1319e-01, -5.2105e-02,
 4.5444e-02,  4.8792e-01, -1.8443e-02,
 6.3148e-02, -2.4687e-01,  9.2494e-03,
-7.1721e-03,  1.9751e-02,  1.0785e-02
}
};
)"+std::string(
R"(
__constant float biasL[8][8] = 
{
{
 0.0176, -0.5805, -0.0049, -0.0190, -0.0026, -0.0424, -0.0053, -0.0270
}
,
{
-0.0314, -0.0334,  0.0008,  0.0251, -0.0261,  0.0041,  0.0096, -0.0038
}
,
{
 0.0007, -0.1435,  0.0211, -0.0258,  0.0062,  0.0014, -0.0052, -0.0188
}
,
{
-0.0074,  0.0020, -0.0064, -0.0067, -0.0020, -0.0145,  0.0053, -0.0056
}
,
{
-3.3930e-03, -7.0365e-03,  2.7018e-04, -4.2781e-03,  1.1594e-01, 2.6517e-02, -5.7359e-03,  7.2912e-01
}
,
{
 4.0424e-02,  6.0326e-01, -6.8195e-04, -2.9932e-38,  1.0781e-02, -2.9386e-03, -2.4809e-04,  5.2628e-02
}
,
{
-2.7425e-03, -7.4154e-03, -4.4120e-02, -3.2650e-39,  7.8514e-02, 7.8860e-04, -2.2220e-03, -6.1027e-03
}
,
{
-0.0005,  0.0140,  0.0099,  0.1028, -0.0012, -0.0006, -0.0009,  0.0090
}
};

__constant float kernelsL10[4 * 8] = 
{
 0.3253, -0.0568,
-0.0937, -0.1085,
-0.0353, -0.2741,
 0.2962,  0.0285,
-0.2525,  0.0696,
-0.0008,  0.1982,
 0.3133,  0.2833,
 0.2296,  0.2678,
 0.2860,  0.0194,
 0.1992, -0.5335,
-0.2106,  0.0542,
-0.0026,  0.2344,
 0.1775,  0.1331,
-0.4549,  0.1348,
-0.0953,  0.2378,
 0.0623, -0.1812
};


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
}
)"
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
}
)"
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
}
)");

const std::string Anime4KCPP::Anime4KGPUCNN::ACNetHDNKernelSourceString =
R"(#define RELU(x) fmax(x, 0.0f)

__constant sampler_t samplerN = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__constant float kernelsL1[9 * 8] = 
{
-6.6444e-02, -2.2293e-01,  4.3874e-02,
 1.6922e-02, -6.8296e-01, -1.5910e-01,
 6.7607e-01,  3.2211e-01,  8.3825e-02,
-4.6628e-01, -6.8774e-02,  5.3469e-01,
-5.0927e-01,  7.0433e-02,  4.5642e-01,
-2.3189e-02,  3.5387e-02, -1.8559e-02,
-2.0223e-01,  4.7525e-02, -2.2261e-01,
 5.4403e-02,  7.0808e-01,  9.2090e-02,
-2.8478e-01, -1.9889e-02, -7.0991e-02,
 4.8378e-02, -9.3281e-02,  2.6368e-03,
 3.4669e-01,  7.2786e-02,  1.6626e-01,
 3.2117e-01,  3.2938e-01, -9.8396e-02,
 1.2761e-02,  2.0483e-01, -1.8375e-02,
 6.1438e-02,  8.3592e-01,  1.4112e-01,
-9.1790e-01, -2.4534e-01, -4.5670e-02,
-5.2576e-02,  4.3175e-01,  1.7042e-01,
 2.6664e-01,  5.5261e-01,  3.5812e-03,
 5.3634e-02, -4.6883e-02, -2.6455e-02,
 1.5687e-01,  4.7334e-01,  2.5085e-03,
 1.7840e-02,  4.1368e-01,  1.3071e-02,
 4.6789e-02,  1.0264e-01,  1.0549e-02,
-3.4509e-02, -7.6401e-01, -9.7871e-02,
-1.4232e-01,  2.9510e-01,  6.6217e-01,
-5.9506e-04,  1.2616e-01, -6.4225e-02
};

__constant float biasL1[8] = 
{
-0.0257, -0.0225, -0.3014, -0.2577, -0.0315, -0.0053, -0.7778,  0.0228
};
)"
R"(
__constant float kernelsL[8][9 * 8 * 8] = 
{
{
-7.8588e-41, -5.0770e-40, -2.3334e-40,
 5.7174e-40,  6.9060e-41,  2.2264e-40,
-4.1631e-40,  4.5667e-40, -1.8115e-40,
-3.1000e-40,  3.1019e-40,  5.5423e-40,
-5.8518e-40,  2.1290e-40, -5.4579e-40,
-3.7753e-40,  6.3749e-40, -1.7875e-40,
 4.2296e-40,  6.3138e-40,  1.4976e-40,
-6.9417e-40, -6.7681e-40, -5.9818e-40,
 3.2053e-40,  3.0821e-40,  5.1321e-40,
-1.1793e-13, -1.6966e-13, -1.6465e-13,
-1.1177e-13, -1.5462e-13, -1.5181e-13,
-8.0184e-14, -1.0260e-13, -1.0028e-13,
 5.9355e-40,  2.4052e-40, -1.0027e-40,
 2.2060e-40,  3.4864e-40, -5.7403e-40,
 4.6936e-40, -3.3951e-40, -4.7715e-40,
-4.3438e-07, -4.5829e-07, -4.2651e-07,
-4.6938e-07, -4.9560e-07, -4.5764e-07,
-4.5213e-07, -4.7381e-07, -4.3802e-07,
-4.6194e-26, -1.0317e-25, -7.9411e-26,
-6.5741e-26, -1.6796e-25, -1.3656e-25,
-4.6399e-26, -1.1875e-25, -9.6259e-26,
 4.2124e-40,  2.5024e-40,  4.5312e-40,
-2.4880e-40,  2.9838e-41, -2.7215e-41,
-2.6347e-40,  1.5950e-40,  9.3734e-41,
-1.5004e-01, -1.0376e-01,  3.1248e-02,
 1.5256e-02, -1.6772e-01, -8.6500e-02,
 5.0217e-02, -2.9619e-02, -1.1343e-01,
-3.4530e-02, -2.5162e-02, -1.8981e-02,
 3.0999e-02,  4.9583e-02, -1.3314e-01,
-8.0518e-03, -9.8829e-03, -4.9034e-02,
-6.5426e-02, -2.6032e-02,  1.9869e-02,
-4.0711e-02,  3.6935e-02,  3.1968e-02,
-5.6236e-02,  2.3972e-02,  3.1233e-02,
-3.2789e-02,  1.2350e-01,  2.5228e-02,
 1.7996e-02,  1.7222e-01,  4.1160e-02,
 9.6771e-03, -6.9245e-02, -1.0607e-01,
 4.8283e-02, -2.2491e-04, -1.4890e-02,
-2.2983e-02,  1.0012e-01,  1.4748e-02,
 3.5048e-02,  1.4723e-01,  7.9790e-02,
-4.3297e-02, -3.8592e-02, -1.9111e-01,
-3.1038e-02,  1.0863e-01, -5.1866e-02,
 8.1257e-02,  2.5826e-01, -9.6486e-02,
-1.1488e-01, -5.2721e-02, -2.7971e-02,
 6.8810e-02,  2.4044e-01,  2.0929e-01,
 2.8720e-02,  2.7838e-01,  2.1926e-01,
-1.5294e-02,  9.4736e-03, -8.0263e-02,
 9.0466e-02,  1.0441e-01, -1.9199e-02,
-3.1957e-02, -1.3685e-02,  3.4920e-02,
 2.3357e-40, -1.4361e-40,  2.0498e-40,
-5.2355e-40, -6.0151e-40, -2.9264e-40,
 1.9715e-41,  5.9793e-41, -1.3675e-40,
 5.3771e-40,  6.3135e-40, -3.8471e-40,
-3.0820e-40, -1.7004e-40, -1.9371e-40,
-5.1159e-40,  7.3244e-41,  3.5861e-41,
 2.8441e-40,  4.5248e-41,  1.9771e-40,
-2.4681e-40,  3.6054e-40,  6.6909e-40,
-6.5048e-42, -1.6001e-40,  4.8243e-41,
-4.0980e-05, -4.0053e-05, -3.8910e-05,
-4.2248e-05, -4.1314e-05, -4.0025e-05,
-4.2286e-05, -4.1548e-05, -4.0332e-05,
-2.8779e-39, -1.6319e-39,  5.3537e-40,
-9.3326e-37, -2.0208e-38,  3.5144e-40,
-1.1447e-33, -8.3919e-36, -2.6811e-39,
-4.9958e-04, -4.9007e-04, -4.8415e-04,
-4.9768e-04, -4.8710e-04, -4.8079e-04,
-4.9853e-04, -4.8814e-04, -4.8153e-04,
-2.7201e-08, -2.6325e-08, -2.5092e-08,
-2.8995e-08, -2.8472e-08, -2.6939e-08,
-3.0365e-08, -3.0438e-08, -2.8963e-08,
-2.2349e-22, -4.7941e-22, -8.4161e-22,
-1.0756e-22, -2.6231e-22, -5.0285e-22,
-9.5473e-24, -2.4576e-23, -4.5888e-23,
 5.1454e-03, -1.2210e-02,  1.6146e-01,
 6.7796e-02, -7.0563e-02,  1.7380e-01,
 1.7889e-03,  1.0884e-01, -3.5576e-02,
 5.2346e-02, -1.2711e-03,  5.0453e-02,
-6.9097e-02, -1.8827e-01, -1.6298e-01,
-2.6728e-02,  2.2278e-02,  6.2331e-02,
 1.3323e-02,  3.2935e-02,  4.7271e-02,
 4.4643e-02,  5.5947e-02,  1.2718e-02,
 4.1900e-02,  1.1954e-01,  6.4866e-02,
-7.3567e-02,  1.4508e-02,  9.6110e-02,
 4.3218e-02, -3.5009e-02,  1.3649e-01,
-2.0859e-01, -2.2406e-02, -8.6696e-02,
-6.5764e-02,  1.3745e-01, -9.7721e-02,
 3.6731e-02,  5.4085e-02,  1.1680e-01,
-1.2129e-01,  3.6003e-02,  7.5118e-02,
-3.4376e-02,  7.8322e-03,  2.5604e-02,
-9.8933e-03,  1.6758e-01,  1.2873e-01,
 3.1844e-02,  1.2159e-01, -1.1357e-02,
-7.1064e-02,  5.5140e-02, -5.0246e-02,
-8.3429e-02,  1.0869e-01,  1.3256e-01,
 2.8566e-02, -1.2673e-02,  8.9002e-02,
-3.1454e-02, -1.1207e-02,  1.0039e-02,
 1.2219e-01, -5.8559e-02,  5.7227e-02,
-5.1164e-02, -6.6645e-02, -7.3296e-02,
-7.7695e-02,  1.3922e-01,  1.1215e-01,
-3.2853e-02, -2.1474e-01, -1.1947e-01,
-1.6117e-02,  1.0425e-01, -8.7792e-02,
-2.0134e-02,  6.1399e-02, -2.5766e-02,
 6.2076e-02, -5.1526e-03, -9.9852e-03,
 6.5683e-02, -5.9467e-02, -5.4961e-02,
 9.6568e-02,  1.7501e-01,  4.7065e-03,
 1.9172e-01,  3.9761e-01,  1.2891e-01,
-6.9445e-02,  4.9033e-02, -3.3822e-02,
 1.1595e-01,  1.3398e-01,  2.9149e-01,
 1.0400e-02,  1.1222e-01,  1.7571e-01,
-1.7876e-02,  8.4856e-02,  5.0042e-02,
-3.8303e-02, -1.3064e-01, -1.1230e-01,
 1.8216e-01,  2.2923e-01, -1.6452e-01,
 4.5320e-02,  1.4885e-01, -8.6931e-02,
 6.9034e-04,  6.8016e-02, -3.3078e-03,
 1.7704e-01,  1.8429e-01,  2.2300e-01,
 1.2586e-01,  1.7702e-01,  1.5963e-01,
 6.0440e-02,  1.6309e-01,  1.4360e-01,
 2.2281e-01,  4.0980e-01,  3.1503e-01,
 1.9845e-01,  4.3088e-01,  3.0545e-01,
 1.2269e-01, -1.0935e-02,  5.9756e-03,
 1.5467e-02, -6.8641e-02,  5.1025e-02,
 5.3248e-02, -1.5389e-01, -1.0545e-02,
-8.6541e-02, -1.0240e-01,  3.2376e-02,
 4.0513e-02, -1.2597e-01, -5.5172e-01,
 8.8075e-02,  3.5842e-02, -1.5526e-01,
 3.6307e-03,  6.7334e-02,  1.0706e-01,
-2.7607e-02, -1.7590e-01, -3.5723e-01,
 2.5388e-02, -5.6200e-01, -5.9087e-01,
-9.3537e-02, -7.3637e-02, -3.5255e-02,
 9.7876e-03, -5.5830e-02, -4.7930e-02,
-7.5248e-02, -1.1652e-01, -7.1505e-02,
 9.1701e-03, -1.1749e-01,  4.7015e-02,
 1.9896e-02, -3.9479e-02,  1.3392e-02,
 4.2104e-02, -5.2126e-02, -1.2225e-01,
 2.4038e-02,  5.2750e-02, -2.4935e-01,
-4.2536e-02, -2.2382e-01, -4.0476e-01,
-3.5332e-02,  2.0007e-01, -1.5788e-01,
 1.3196e-01, -1.0514e-01, -6.7311e-02,
-2.5254e-01,  1.1143e-02,  2.5564e-01,
-3.3875e-01,  3.2018e-02,  5.9422e-01,
 4.0873e-02,  4.6350e-03, -1.9309e-02,
-6.7106e-02, -1.5509e-01, -3.0540e-02,
 8.9450e-02, -4.5764e-02,  6.9062e-02,
 8.6334e-02,  3.6184e-02, -1.9751e-02,
 4.6186e-02, -4.7738e-02,  1.0071e-01,
-2.5667e-01, -2.7710e-01,  6.2212e-02,
-7.3232e-02, -2.4296e-03,  9.5984e-02,
 2.2962e-02, -2.7861e-01,  1.4466e-01,
 1.4438e-02, -4.6571e-02,  9.9716e-02,
 1.4309e-02,  1.5547e-01,  1.5798e-01,
 6.2034e-02,  2.4734e-02, -1.2728e-02,
 1.3093e-01,  2.0641e-01,  1.0088e-01,
 1.4809e-02,  5.8048e-02, -1.8589e-02,
 2.1144e-01,  3.2701e-01,  4.4199e-02,
-6.2928e-02, -1.1757e-01, -1.1268e-01,
 7.3304e-04,  6.5775e-02, -1.8233e-01,
 1.3999e-01, -2.1044e-02,  8.6909e-02,
-4.0005e-02,  8.2727e-02, -1.0482e-01,
 4.4236e-02, -7.0033e-02,  1.6908e-01,
 2.7376e-02,  5.3112e-01,  2.4331e-01,
 1.4189e-03,  4.0125e-02,  7.0927e-03,
-3.5087e-02, -3.1060e-03, -2.2590e-01,
 1.2457e-01,  7.1812e-01, -4.1704e-01,
 7.5948e-02,  2.5947e-02, -2.8636e-01,
-7.7623e-02, -7.1195e-03, -5.6543e-02,
 8.0959e-02,  1.0478e-01,  7.5120e-03,
 2.2232e-02,  3.6463e-02, -1.1633e-01,
-1.1589e-01, -7.5466e-02,  1.3308e-02,
-2.9322e-01,  2.6201e-01, -2.3559e-02,
 1.2823e-01,  1.4328e-01, -9.0222e-02,
 3.0891e-02,  2.0511e-01,  1.9541e-02,
-7.8176e-02, -3.9327e-01, -1.5256e-01,
-5.4170e-02, -8.6919e-02, -5.4076e-02,
 7.0987e-02,  1.5289e-01, -8.9628e-03,
 2.0048e-02,  1.2123e-01, -7.0541e-02,
-1.0131e-02, -1.8656e-01, -9.7144e-02,
-6.4399e-02,  6.8395e-02, -4.9178e-02,
-8.8638e-03,  2.4169e-01,  6.0722e-02,
 3.6711e-02,  7.6478e-02,  1.7409e-02,
-5.2405e-02, -1.8927e-01, -1.1216e-01,
 3.2132e-01, -9.7989e-02,  1.4725e-01,
 8.7974e-02, -5.0940e-03, -2.7328e-03,
-9.1506e-03,  2.8101e-01, -1.5744e-01,
 4.7673e-02,  4.5162e-01, -1.6169e-01,
 5.8889e-02,  6.2325e-02, -2.7162e-02,
 2.0585e-01, -7.2349e-02,  2.2928e-02,
-3.5594e-01,  1.3201e-01, -7.5672e-02,
 3.9579e-02,  1.7996e-01, -1.1810e-01,
-4.6218e-02,  1.3432e-02,  2.7698e-02,
-1.9967e-01, -1.6724e-02,  1.1840e-02,
 1.2047e-02,  1.2665e-01,  6.9015e-02,
 2.0847e-01,  1.7427e-01,  6.9773e-02,
 5.5169e-01,  4.6981e-01,  1.8463e-01,
 2.2691e-01,  2.4360e-01,  1.3134e-01
}
,)"
R"(
{
-2.3228e-08, -2.3990e-08, -2.2220e-08,
-2.4225e-08, -2.5279e-08, -2.3452e-08,
-2.2305e-08, -2.3157e-08, -2.1818e-08,
 5.9433e-02,  1.0188e-01, -1.0479e-01,
 6.9514e-02,  1.2920e-01,  1.1508e-01,
-1.5732e-01,  9.6889e-03, -9.0008e-02,
 3.9038e-04,  3.9895e-04,  4.0370e-04,
 4.0191e-04,  4.1090e-04,  4.1473e-04,
 4.0100e-04,  4.0987e-04,  4.1231e-04,
 2.9557e-02,  5.4669e-02, -1.2058e-01,
 5.8595e-02,  2.2284e-01,  1.1189e-01,
-9.5949e-02,  8.3365e-02, -6.5673e-02,
 1.2033e-01,  1.9798e-01,  7.7711e-02,
 6.7254e-02,  3.6557e-01,  1.3817e-01,
 6.0975e-02,  4.6367e-02,  2.3762e-01,
-2.1251e-02, -2.0232e-02,  9.3753e-02,
-4.7032e-02, -1.5910e-01,  8.8088e-02,
-5.6709e-02, -1.7148e-01, -1.0498e-01,
-2.5535e-01,  1.6090e-01, -3.3798e-02,
-2.3176e-01,  5.4080e-01,  1.0018e-01,
-2.1621e-01, -6.8913e-03, -1.0708e-01,
-4.8733e-02,  2.5122e-01,  4.7934e-02,
 9.7404e-02,  5.5062e-01, -3.4611e-01,
 1.2906e-02, -1.3565e-01, -4.1943e-01,
-2.1359e-40, -1.4250e-40, -4.7123e-40,
-5.9433e-41,  1.9903e-41, -1.7701e-40,
-5.9941e-40, -5.8562e-40, -5.0226e-40,
-2.6581e-40,  1.3006e-40, -1.4201e-40,
 5.4264e-40,  2.3848e-40,  5.6412e-40,
-2.6378e-41, -5.7132e-40, -4.1343e-40,
-1.4587e-18, -1.6296e-18, -1.5164e-18,
-1.5889e-18, -1.7513e-18, -1.5982e-18,
-1.3861e-18, -1.5234e-18, -1.3620e-18,
-4.6733e-05, -4.6204e-05, -4.5817e-05,
-4.8078e-05, -4.7339e-05, -4.6623e-05,
-4.6842e-05, -4.6282e-05, -4.6050e-05,
-6.3486e-04, -6.2825e-04, -6.1269e-04,
-6.4546e-04, -6.3850e-04, -6.2347e-04,
-6.3480e-04, -6.3132e-04, -6.2091e-04,
 5.6433e-41, -3.0514e-40, -5.4526e-40,
 1.1125e-41,  2.9485e-40,  5.5282e-40,
 3.0229e-40,  1.5915e-40,  5.3759e-40,
-2.7153e-23, -4.1024e-22, -1.0792e-21,
-4.1670e-21, -4.5690e-20, -8.6652e-20,
-1.9427e-20, -1.9699e-19, -3.1310e-19,
-3.6239e-32, -1.4640e-33, -2.6379e-36,
-1.7577e-33, -4.6667e-35, -1.8886e-37,
-2.0268e-35, -7.6870e-37, -9.7433e-39,
-6.6958e-40, -3.9935e-40,  2.4649e-40,
 2.0207e-40, -3.0245e-40, -7.1986e-41,
 6.2938e-40, -3.6922e-40,  1.5296e-40,
-6.4982e-41,  5.0849e-41,  5.7873e-40,
 1.4327e-40, -4.2163e-40,  1.3807e-40,
 2.8569e-40,  1.9139e-40,  6.7548e-40,
-5.4410e-40,  7.9929e-40,  8.1660e-40,
-1.5964e-40, -8.0302e-40,  5.6766e-40,
 2.2533e-42, -7.6865e-40, -5.5822e-40,
 5.7249e-40,  5.3555e-40, -4.9107e-41,
 1.7538e-40, -1.2312e-40,  5.0077e-40,
 6.1500e-40,  1.9980e-40,  6.2953e-40,
-3.3445e-19, -4.1876e-19, -3.1682e-19,
-3.7809e-19, -4.9901e-19, -4.0179e-19,
-2.7550e-19, -3.7827e-19, -3.2425e-19,
 5.0613e-40,  1.5224e-40, -1.8977e-40,
 2.4108e-41, -5.1771e-40,  6.2317e-40,
 1.0465e-40,  2.8816e-41,  6.2500e-40,
 6.4133e-40,  4.2717e-40, -3.5900e-40,
-4.4831e-40,  6.5942e-40, -4.8293e-40,
-2.4133e-40,  3.1140e-40, -2.0777e-40,
-2.2906e-41,  6.3887e-40, -4.4443e-40,
-4.6615e-40, -2.1123e-40,  4.5700e-40,
-4.6360e-40, -6.3728e-40, -6.5868e-40,
-3.6575e-40, -6.4153e-40, -3.0530e-41,
 4.2531e-40, -1.2255e-40, -3.9607e-40,
 6.3907e-40, -5.4630e-40, -3.1460e-40,
 2.8820e-40,  4.9460e-40,  6.1461e-40,
 8.9118e-41, -4.6579e-40, -2.4172e-40,
-5.5474e-40, -8.1848e-41, -1.6910e-40,
-7.2263e-22, -8.3496e-22, -7.6511e-22,
-7.9268e-22, -9.0319e-22, -8.0980e-22,
-6.5344e-22, -7.4309e-22, -6.5194e-22,
-2.3308e-05, -2.3861e-05, -2.3933e-05,
-2.4601e-05, -2.5034e-05, -2.4702e-05,
-2.4329e-05, -2.4578e-05, -2.4070e-05,
-8.6303e-04, -8.6301e-04, -8.5264e-04,
-8.7934e-04, -8.7528e-04, -8.6090e-04,
-8.7896e-04, -8.7340e-04, -8.5532e-04,
-4.5042e-40,  2.6725e-40,  2.3181e-40,
-4.6274e-41, -1.1799e-40,  5.0685e-40,
-1.0765e-40,  3.3322e-40, -6.1905e-40,
-6.0632e-31, -1.5405e-29, -5.1414e-29,
-6.4145e-28, -9.7678e-27, -2.1612e-26,
-5.7576e-27, -8.9655e-26, -1.5082e-25,
-2.5057e-40,  7.2876e-41,  4.5731e-41,
-1.6525e-40,  5.0987e-40, -5.4683e-40,
 8.1836e-41,  6.2722e-40, -3.1057e-40,
 4.0987e-40,  3.5941e-40,  5.1680e-40,
 5.5563e-40,  3.1011e-40,  4.7068e-40,
 1.0426e-40, -1.0803e-40,  4.4867e-40,
-4.2341e-03,  1.5491e-01, -2.2148e-03,
-6.1977e-02,  2.0283e-01,  1.9561e-01,
 4.0616e-02,  1.6947e-01, -2.5971e-02,
 4.8134e-27,  8.2637e-27,  5.0563e-27,
 6.6005e-26,  8.6249e-26,  5.1686e-26,
 7.7732e-26,  9.8483e-26,  6.0524e-26,
 6.6783e-02,  6.8246e-02,  2.2906e-01,
-9.9780e-02,  7.8602e-02, -1.7383e-01,
-1.4780e-01,  1.0959e-01,  1.4324e-01,
-2.5950e-02, -2.7116e-02, -9.2170e-02,
 5.3424e-02,  9.2268e-02, -2.2852e-02,
-3.0577e-02,  5.9876e-02,  9.9966e-02,
-2.3211e-01, -1.3794e-01,  2.8721e-01,
-3.2479e-01, -2.6648e-01,  6.1669e-02,
 1.8940e-01,  4.8185e-02, -2.3550e-01,
 2.8471e-02, -1.2107e-01, -1.5591e-01,
-1.1998e-01,  5.2118e-01,  2.0335e-01,
-1.2343e-01,  1.0231e-01,  6.2664e-02,
-3.9162e-02, -9.1932e-02, -1.8909e-01,
 1.1345e-01,  5.7301e-01, -1.8074e-01,
-3.4684e-02, -2.1132e-01, -2.1975e-01,
 1.7543e-39, -3.4709e-40,  6.2940e-40,
-2.2134e-41, -3.6133e-40, -2.7075e-40,
-5.9664e-40, -2.3937e-40,  3.0876e-40,
 9.1814e-41,  9.5898e-41, -6.8908e-40,
 6.9916e-40,  2.7935e-40,  1.7966e-40,
-2.3967e-40,  4.0806e-40,  6.2012e-40,
-2.1016e-39, -1.9401e-39,  1.6943e-39,
-2.1198e-39,  1.7341e-39,  1.0160e-39,
-1.6996e-39,  2.0917e-39,  1.0378e-39,
-9.7756e-41,  2.7978e-40, -5.0791e-40,
-3.4321e-40, -7.0774e-41, -5.2651e-40,
 2.8034e-40, -3.3452e-40,  1.9535e-40,
-6.2300e-40, -8.5804e-40, -1.9038e-40,
-5.6564e-40, -6.1257e-40, -1.0338e-40,
-1.7191e-41, -1.2843e-41,  5.0707e-40,
-4.4587e-40,  2.7128e-40, -1.4155e-40,
-5.7475e-40, -6.5500e-40, -4.7424e-40,
 1.7235e-40, -6.0028e-40, -1.6342e-40,
-5.1072e-40, -2.4721e-40, -2.8477e-41,
 2.6598e-40, -4.4078e-40,  4.1763e-40,
-3.3947e-40, -5.5626e-40,  4.9713e-40,
 2.1733e-40, -2.9024e-40, -4.5514e-42,
-3.4873e-40, -1.0737e-40, -1.4297e-40,
 2.8514e-40,  7.5916e-40,  2.2827e-40,
 3.8908e-40, -4.2140e-40,  6.1433e-40,
-4.7825e-40, -3.0140e-40, -5.9563e-40,
 1.5280e-40,  2.6156e-40,  5.0361e-40,
 1.9478e-01,  2.3088e-01, -3.5943e-02,
 1.6791e-01, -1.8740e-02, -2.0454e-01,
 9.8125e-02, -6.9150e-02, -4.0988e-02,
-2.6313e-11, -2.9945e-11, -3.6355e-11,
-2.5468e-11, -2.9706e-11, -3.9893e-11,
-3.7350e-11, -4.7211e-11, -7.0822e-11,
 8.9952e-02,  6.8024e-02,  6.8061e-02,
 7.4208e-03, -8.6107e-02,  1.1549e-01,
-1.0157e-01,  1.5224e-01,  4.1973e-02,
 1.0130e-01,  1.0320e-01, -7.9211e-02,
 2.0842e-01,  1.7578e-01, -7.1475e-02,
-1.4212e-01, -3.8945e-02, -2.7511e-01,
 2.2116e-01, -6.3049e-02, -7.4146e-02,
-3.2359e-01,  6.5626e-02,  1.1042e-01,
 7.1619e-02, -1.7027e-01,  5.8008e-02,
-1.4787e-01, -2.0816e-01, -8.7404e-02,
-2.7163e-01,  5.7729e-01,  1.6102e-02,
-7.1720e-02,  1.2918e-01, -1.0514e-01,
-8.6943e-02,  3.8404e-02,  1.8243e-01,
 1.7560e-01,  3.8299e-01,  3.4889e-02,
 6.0014e-02, -1.6827e-03,  5.1467e-01,
-2.2651e-14, -2.4885e-14, -2.6078e-14,
-2.6161e-14, -2.5925e-14, -2.5007e-14,
-2.3357e-14, -2.3019e-14, -2.0561e-14,
 6.6309e-02,  2.1031e-02, -1.0374e-01,
 2.2336e-03,  2.0105e-01, -9.0405e-03,
-9.0985e-02, -1.1671e-01, -1.6309e-02,
 6.6389e-05,  6.5975e-05,  6.6718e-05,
 6.8908e-05,  6.8170e-05,  6.8201e-05,
 7.2125e-05,  7.1255e-05,  7.0266e-05,
-1.7294e-01, -1.2234e-01, -5.0848e-02,
-3.4702e-02, -1.2433e-01,  8.8317e-03,
-7.4894e-02, -1.1316e-01,  2.2855e-03,
 8.5116e-02,  4.2196e-02, -5.0224e-02,
 1.2854e-01,  1.9671e-01,  1.0569e-01,
-1.0097e-01,  6.2852e-02,  8.8960e-02,
 3.4823e-01,  5.4396e-01, -2.2363e-01,
 5.1724e-01,  5.7095e-01, -6.7593e-02,
-3.5834e-02,  1.0596e-01, -2.3007e-01,
-1.5568e-01, -6.1368e-03,  4.7717e-02,
-1.3284e-01,  2.8156e-01, -1.1676e-02,
-3.6141e-02, -6.8803e-02, -1.3188e-02,
 1.1679e-02,  1.2175e-01, -1.3512e-02,
 9.2292e-02, -6.4212e-02, -1.6618e-01,
 1.9408e-01, -1.5273e-01, -8.3138e-02
}
,)"
R"(
{
-5.0529e-03, -5.0816e-03, -4.9826e-03,
-5.1735e-03, -5.1934e-03, -5.0781e-03,
-5.1344e-03, -5.1473e-03, -5.0280e-03,
 2.8622e-41, -1.2033e-41,  1.2609e-40,
-4.9379e-40, -5.1047e-40,  5.5085e-41,
-4.7002e-40, -5.0136e-40, -4.5629e-40,
-5.1095e-40,  1.8741e-40,  1.8435e-40,
 4.1851e-40, -8.9558e-41, -9.6681e-41,
-1.8244e-40,  2.7992e-40,  1.8116e-40,
 2.8655e-40, -3.0193e-40,  8.0916e-40,
-1.2974e-39,  3.3049e-40,  6.9542e-41,
-3.3329e-40,  4.2212e-40, -1.3453e-40,
-3.7526e-11, -4.9289e-11, -4.1821e-11,
-4.3866e-11, -5.2258e-11, -4.2132e-11,
-3.0111e-11, -3.3111e-11, -2.5767e-11,
 6.0452e-40,  6.9262e-41,  2.9300e-40,
-6.1511e-40, -4.1269e-40,  4.4012e-40,
 1.0860e-39, -2.9020e-40, -4.5529e-40,
-5.4574e-19, -6.2048e-18, -2.4733e-17,
-7.9285e-18, -7.8793e-17, -2.5201e-16,
-3.0424e-17, -2.7690e-16, -7.1956e-16,
-6.3341e-04, -6.4536e-04, -6.1857e-04,
-6.6643e-04, -6.8061e-04, -6.5019e-04,
-6.5324e-04, -6.6813e-04, -6.3939e-04,
-1.4480e-02, -1.1904e-01, -1.8555e-02,
-3.0013e-03,  2.1761e-01,  3.0322e-02,
 1.4337e-01,  3.5242e-02, -1.0355e-02,
-1.8517e-08, -2.1669e-08, -2.8434e-08,
-1.0289e-08, -1.2380e-08, -1.6553e-08,
-7.1840e-09, -8.8908e-09, -1.1741e-08,
-5.9514e-40, -4.4339e-40, -3.0315e-40,
 3.5756e-40,  2.5390e-40, -1.2253e-40,
 2.1417e-40,  4.0569e-40,  5.3962e-40,
-2.4791e-09, -3.0432e-09, -4.1515e-09,
-1.2951e-09, -1.6413e-09, -2.2729e-09,
-8.3058e-10, -1.0986e-09, -1.5107e-09,
-2.7166e-01, -4.5113e-01,  2.5916e-02,
-8.0217e-02,  5.0732e-01, -7.9285e-02,
-3.7216e-02, -6.0969e-03, -3.9797e-01,
 1.8705e-39,  1.8064e-39,  1.0486e-39,
 2.1538e-39,  1.7290e-39,  1.2229e-39,
 1.5448e-39,  2.0811e-39,  2.0965e-39,
 4.2106e-01, -1.5795e-02, -6.5534e-02,
-2.5237e-02,  5.6213e-01,  1.6966e-01,
-1.9723e-05,  1.8912e-01, -5.9164e-02,
-3.0686e-01, -1.7284e-01, -8.6658e-02,
-1.6788e-01,  1.8817e-02, -1.7066e-01,
 8.5936e-02,  3.0021e-01, -7.6131e-02,
 1.1691e-02,  1.0680e-01, -5.1996e-02,
-1.2783e-01,  4.2631e-01,  4.9520e-02,
-2.8305e-01, -1.3444e-01, -7.4049e-02,
-7.5661e-12, -4.9543e-12, -3.2589e-12,
-1.1702e-11, -7.2095e-12, -4.7065e-12,
-3.4576e-11, -2.0863e-11, -1.3336e-11,
-3.0246e-40, -4.1596e-40,  2.9013e-40,
 8.5195e-41, -2.2396e-40, -2.0322e-40,
-5.6200e-40,  2.4820e-40,  3.1309e-40,
-1.4132e-13, -7.3651e-14, -3.9353e-14,
-2.6604e-13, -1.3239e-13, -7.1614e-14,
-1.0840e-12, -5.5694e-13, -3.0138e-13,
 1.5892e-01, -1.4218e-01, -6.0966e-02,
 6.0029e-02,  3.4569e-01,  1.4269e-01,
 5.2811e-02, -1.8327e-01, -6.8749e-02,
 1.1844e-39,  5.9969e-40,  5.9290e-40,
 2.0585e-39,  2.1604e-39,  5.3120e-40,
 1.3612e-40,  1.6946e-39, -9.5433e-40,
-1.1555e-01, -1.2147e-02,  2.5986e-02,
-8.0523e-02,  2.9810e-01,  1.9044e-01,
 2.0375e-02,  1.9450e-03, -1.2702e-01,
-6.6233e-02, -3.5716e-02, -5.6364e-02,
 9.6167e-02,  2.2097e-02,  2.0502e-01,
 9.7640e-03,  7.6339e-03,  3.8114e-02,
 4.9251e-03,  2.1334e-01,  3.8940e-03,
 3.5709e-01,  4.6886e-02,  2.5348e-01,
 4.1790e-02,  1.7924e-01,  3.2425e-02,
-1.1825e-17, -3.4268e-18, -2.1798e-18,
-6.3593e-18, -1.2050e-18, -5.6450e-19,
-8.7252e-18, -1.5175e-18, -6.1706e-19,
-2.2836e-40,  6.8665e-40, -4.4396e-40,
 7.2465e-40,  6.0866e-40,  3.7804e-40,
-7.0432e-40, -2.4897e-40,  4.9891e-40,
-8.4177e-25, -1.5540e-25, -5.7348e-26,
-2.1199e-25, -1.5637e-26, -2.8422e-27,
-3.6818e-25, -1.4074e-26, -1.6391e-27,
-9.3870e-02,  1.5105e-01,  1.0805e-01,
 2.6058e-01, -6.6897e-01,  2.3624e-01,
 1.5015e-01,  1.9154e-01,  8.7674e-02,
 2.0175e-39, -2.0987e-39,  1.7775e-39,
 1.0741e-39,  1.8467e-39,  1.4887e-39,
 1.4505e-39, -1.2996e-39,  4.8221e-41,
 7.9016e-02,  1.6261e-01,  8.0809e-02,
 3.2329e-01, -2.6142e-01, -9.0138e-02,
-7.3671e-02,  1.5635e-02, -8.1942e-02,
-2.1593e-01, -4.4600e-01, -4.4737e-02,
-3.9393e-01, -8.2354e-01, -7.0635e-01,
-6.8438e-03, -4.9690e-01, -1.4217e-01,
-1.1410e-03, -1.1270e-03, -1.0762e-03,
-1.1528e-03, -1.1454e-03, -1.0935e-03,
-1.1389e-03, -1.1334e-03, -1.0880e-03,
-2.1073e-25, -1.9937e-25, -1.1264e-25,
-2.4988e-25, -2.4142e-25, -1.3712e-25,
-1.9454e-25, -1.8992e-25, -1.0912e-25,
 3.8855e-40,  3.5152e-40, -4.8707e-40,
-2.0638e-39, -1.7886e-40,  5.1970e-40,
 6.2864e-40, -1.8713e-39,  8.1025e-40,
-9.7454e-34, -8.5332e-34, -3.1646e-34,
-1.2075e-33, -1.1036e-33, -4.2669e-34,
-8.2178e-34, -7.6283e-34, -3.0038e-34,
-1.2906e-26, -1.3911e-26, -4.8299e-27,
-1.2183e-26, -1.6447e-26, -5.6935e-27,
-3.0565e-27, -4.3834e-27, -1.6844e-27,
 1.1669e-39, -1.0395e-39, -7.3436e-40,
 1.5164e-39,  2.0520e-39, -1.0345e-39,
-1.0656e-39, -1.7689e-39, -3.3205e-40,
-1.3650e-21, -3.6550e-20, -9.5258e-20,
-6.4257e-21, -2.0042e-19, -5.4076e-19,
-1.9025e-20, -5.8074e-19, -1.5963e-18,
-2.2849e-04, -2.2682e-04, -2.1358e-04,
-2.3154e-04, -2.3181e-04, -2.1866e-04,
-2.2647e-04, -2.2727e-04, -2.1545e-04,
-2.6015e-23, -3.9294e-23, -3.6037e-23,
-3.3012e-23, -5.3449e-23, -5.2884e-23,
-2.9312e-23, -4.7452e-23, -4.8001e-23,
-5.6355e-40, -2.3469e-40, -3.5885e-40,
-2.0755e-40,  2.0377e-40,  3.2259e-40,
-5.3947e-40,  4.2747e-41,  4.8967e-41,
-2.0698e-39, -1.7393e-39, -2.0473e-39,
 1.7147e-39,  1.7159e-39,  1.7959e-39,
 1.7962e-39,  1.7783e-39, -1.4223e-39,
-1.1702e-40, -2.3137e-41, -4.5405e-40,
-4.6797e-40,  6.5582e-41,  1.8111e-40,
 6.1477e-40, -1.6827e-40, -2.0288e-40,
-2.4220e-41,  4.7774e-40,  5.1050e-40,
-6.3227e-40,  5.6437e-41,  4.7749e-40,
-6.8037e-41, -5.5944e-41, -5.2248e-40,
-2.9382e-40, -1.1174e-39, -1.2855e-39,
-4.5290e-40, -5.2260e-41,  2.3726e-40,
 1.3281e-39, -7.9398e-40, -2.9736e-40,
-2.8081e-40, -5.2929e-40, -4.0786e-40,
-3.0303e-41,  3.1336e-40, -5.8450e-40,
-1.5091e-40, -2.7371e-40, -4.5927e-40,
-1.8429e-34, -3.0900e-34, -2.4399e-34,
-2.8078e-34, -5.1386e-34, -4.4337e-34,
-2.1799e-34, -4.0985e-34, -3.6000e-34,
-1.2997e-25, -1.9986e-25, -1.7715e-25,
-1.7099e-25, -2.8141e-25, -2.6652e-25,
-1.4549e-25, -2.4045e-25, -2.3009e-25,
 3.3692e-40,  1.0044e-40, -6.6821e-41,
 9.2910e-41,  6.2137e-40, -3.5625e-40,
 1.8601e-40,  3.1653e-40, -1.1506e-40,
-1.2388e-39,  1.8342e-39, -1.8294e-39,
 1.3807e-39,  1.3603e-39, -1.7869e-39,
-1.7227e-39, -1.7752e-39,  1.7258e-39,
-2.0721e-40, -6.3499e-40, -5.9485e-40,
 3.8843e-40, -6.0861e-41, -4.0542e-40,
-3.4308e-40, -4.2822e-40, -3.9605e-40,
-5.7429e-40,  4.9242e-40, -5.9141e-40,
 4.6267e-40, -2.4953e-40, -2.9300e-40,
 5.3466e-40, -5.2403e-40,  6.4802e-40,
-8.5894e-40,  2.9157e-40, -7.7367e-41,
-5.8922e-40,  3.2359e-40, -6.1293e-40,
 6.1138e-40, -1.3667e-39, -5.0657e-42,
 4.7910e-40, -1.4080e-40,  1.9220e-40,
-3.5670e-40,  3.4204e-40, -5.0215e-40,
 1.1877e-41,  2.3114e-40, -4.7794e-40,
-5.0255e-38, -9.0719e-38, -6.8319e-38,
-7.9126e-38, -1.5789e-37, -1.3023e-37,
-5.5680e-38, -1.1540e-37, -9.6620e-38,
 1.3756e-01,  1.8508e-01,  7.6456e-02,
 8.1960e-02,  3.3367e-01,  3.0120e-02,
-4.0177e-03,  2.0983e-01,  1.8836e-02,
 4.8694e-04,  5.1602e-04,  5.4896e-04,
 5.2765e-04,  5.5349e-04,  5.7725e-04,
 5.4815e-04,  5.6937e-04,  5.9273e-04,
-1.7150e-41, -2.4088e-41, -1.5593e-40,
 6.3817e-41,  4.8004e-41, -1.1053e-40,
-2.5225e-40, -2.7111e-40, -4.2970e-40,
 8.9238e-04,  9.0772e-04,  9.2404e-04,
 9.2361e-04,  9.3716e-04,  9.4715e-04,
 9.3736e-04,  9.4895e-04,  9.5734e-04,
 1.0472e-02,  1.5004e-01, -4.8637e-02,
 2.7297e-02,  1.9126e-01, -1.4059e-03,
-2.1728e-02, -1.5097e-01, -1.6049e-01,
 1.0667e-39, -1.0623e-39, -4.1857e-42,
-1.3507e-39,  1.1145e-39, -5.4341e-40,
-1.6960e-39, -1.9075e-39, -1.7366e-39,
-3.2926e-01, -2.5943e-01, -1.2990e-01,
-2.6478e-01,  3.2323e-01,  4.3230e-01,
-7.0801e-02,  1.9820e-01,  9.4808e-02,
-9.9595e-03,  7.4277e-02, -7.1363e-02,
 7.8027e-02,  1.5751e-01,  1.3305e-01,
-9.5237e-02,  1.1004e-01, -7.6579e-02
}
,)"
R"(
{
-5.5262e-40,  3.7699e-40, -1.4920e-40,
 4.0064e-40, -2.0632e-40, -4.4801e-41,
-3.6749e-40,  5.9043e-40, -1.5942e-40,
-5.9219e-42, -4.1286e-40, -1.6920e-40,
-2.5927e-40, -4.5458e-41,  2.0990e-40,
-4.6860e-40,  5.0483e-40,  2.8004e-40,
-4.0641e-40,  6.0770e-40, -3.8297e-42,
 5.7537e-40,  5.7772e-40, -1.0048e-40,
 1.5945e-40,  3.9582e-40, -2.6190e-40,
-5.1046e-40, -5.5028e-40,  5.8786e-40,
-3.5033e-40, -1.2031e-40, -3.4156e-40,
 3.0058e-40,  4.3043e-40,  5.9825e-40,
 4.9197e-40,  2.5974e-40,  2.0636e-39,
-4.1935e-40, -1.6383e-41,  1.2710e-39,
-5.3501e-40, -2.6348e-40,  3.0631e-40,
 1.7653e-39,  1.6600e-39, -1.3915e-39,
 2.0686e-39,  1.6850e-39,  1.7439e-39,
 1.6219e-39, -1.6769e-39,  7.2711e-40,
-1.7674e-39, -9.2292e-40,  5.1600e-40,
 1.3772e-39,  1.4867e-39, -1.7981e-39,
 1.0584e-39, -1.8018e-39,  8.2532e-40,
-8.8250e-07, -9.1724e-07, -8.7809e-07,
-9.3482e-07, -9.7422e-07, -9.3630e-07,
-9.2152e-07, -9.6113e-07, -9.2432e-07,
-8.2457e-40, -2.1208e-40, -6.6651e-40,
 3.2370e-40,  7.2180e-40, -3.6860e-40,
 6.5330e-40, -8.2954e-40, -3.8183e-40,
-7.8133e-02,  1.6316e-02,  3.1332e-03,
 4.0335e-02,  8.7919e-02, -3.5591e-01,
 1.1937e-02,  9.8093e-02,  1.2514e-01,
-2.7028e-02,  1.0112e-01,  5.7683e-02,
-3.9385e-02,  1.0868e-01, -5.4359e-01,
 1.3993e-02, -4.8072e-02, -1.0488e-01,
-4.1355e-03,  5.2486e-02, -1.6932e-01,
 5.2922e-02, -4.0189e-01,  7.4866e-01,
-8.9131e-02,  1.2193e-01, -1.8666e-01,
 3.7044e-40, -4.6951e-40, -1.9873e-40,
 5.3289e-41,  2.7689e-40, -4.6994e-41,
-3.1404e-40, -5.9106e-40,  6.0436e-40,
-6.0294e-40, -3.6565e-40, -1.1884e-40,
 5.5933e-40, -9.5741e-41,  4.4736e-40,
 4.3267e-40, -4.9583e-40,  6.5736e-40,
-1.7432e-40,  1.4518e-40,  2.1033e-40,
-6.5445e-40,  1.7222e-40, -2.5651e-40,
-5.2517e-40,  2.8983e-41, -1.3832e-40,
-1.4149e-01,  9.4228e-02, -9.8409e-02,
 2.0659e-01,  4.0850e-01, -1.1893e-01,
-1.4142e-01, -1.1047e-01, -8.1414e-02,
 3.4336e-41,  1.5625e-40,  2.7213e-40,
-5.3447e-40, -3.7330e-40, -3.3637e-40,
-4.3563e-40, -3.7094e-40,  1.2820e-41,
-8.1191e-02, -1.8393e-01, -1.6163e-01,
-1.4108e-01,  5.4285e-02, -3.7827e-02,
-1.1452e-01, -1.2443e-01, -8.4098e-03,
-6.2122e-02, -3.3929e-01,  4.5698e-03,
 1.5094e-01, -1.9232e-01, -5.5310e-02,
 6.2557e-02,  9.0761e-02,  1.2885e-02,
 5.2116e-02,  3.3973e-01, -1.7911e-01,
-2.4911e-01,  8.5095e-02,  2.0567e-01,
 5.8123e-02, -1.4893e-01,  7.1282e-02,
-4.6233e-40,  1.2244e-40, -3.9802e-40,
 5.8530e-40, -2.4162e-40,  4.6793e-40,
-4.8362e-40,  3.3071e-40,  1.7094e-40,
 3.5249e-40, -4.8579e-40,  1.9374e-40,
 6.2372e-42,  5.8402e-41,  3.2851e-40,
 6.1488e-40,  1.8086e-40, -5.2451e-40,
-3.0723e-40, -5.6704e-40, -5.9899e-40,
-6.3805e-40, -1.3818e-40, -2.7285e-40,
 2.4468e-40,  8.3606e-41,  1.8818e-40,
-2.3768e-01, -2.7017e-01, -1.3843e-03,
 1.4791e-01,  9.0405e-02,  2.6985e-02,
 1.8699e-01,  1.8145e-01, -1.1826e-01,
-8.3961e-40, -4.8879e-40, -6.8815e-40,
-1.0245e-40,  9.1421e-41,  5.3018e-40,
 2.2240e-40, -1.4666e-40, -4.4259e-40,
 1.1797e-01, -2.7590e-01,  1.1569e-01,
 1.3289e-01,  4.3092e-01,  1.3511e-01,
 2.9749e-02,  1.7162e-01, -1.1933e-01,
 3.6329e-02,  8.7132e-02,  6.6364e-02,
-8.2724e-02,  2.4208e-01,  6.2362e-02,
-6.8489e-02, -3.4891e-02,  3.2621e-02,
-6.4381e-02,  2.2852e-01,  1.8068e-01,
 1.8914e-01, -2.4848e-01,  1.6845e-01,
-3.3511e-03,  1.8927e-01, -7.6231e-02,
-1.2559e-26, -2.6187e-27, -1.0294e-27,
-7.1779e-27, -1.7064e-27, -7.4554e-28,
-8.6207e-27, -1.8649e-27, -7.8386e-28,
-2.0525e-40,  4.6680e-40,  5.9108e-41,
 1.0336e-40, -5.7226e-41, -6.1906e-40,
-1.8693e-40,  5.5777e-40,  6.0898e-40,
-3.4735e-41, -3.2674e-40, -2.3864e-41,
-6.6780e-40,  6.7382e-40,  1.0843e-40,
 5.1103e-40,  6.0598e-40, -3.6267e-40,
-3.2945e-03, -1.0626e-01, -7.5327e-02,
-1.2691e-01,  2.7256e-01,  1.0533e-01,
-2.1221e-01,  9.6603e-02,  3.4663e-02,
 5.0197e-09,  5.1577e-09,  4.3114e-09,
 5.9335e-09,  6.0823e-09,  5.0121e-09,
 5.6043e-09,  5.7454e-09,  4.7426e-09,
-8.5677e-02,  6.3241e-02,  2.2245e-02,
-2.1082e-02,  2.9110e-01,  8.5181e-02,
-1.3654e-01, -1.6047e-01, -4.7292e-02,
-1.3841e-01, -6.5895e-02, -7.7031e-02,
-5.1711e-02,  6.1234e-02, -4.9747e-02,
 1.1784e-01,  1.7539e-01,  3.0307e-02,
 6.6264e-03,  4.5099e-02, -3.4813e-02,
 1.8099e-02, -2.3795e-01,  2.1157e-01,
 7.8285e-02, -7.6366e-02,  2.1433e-02,
-1.4611e-03, -1.4491e-03, -1.4022e-03,
-1.4677e-03, -1.4588e-03, -1.4141e-03,
-1.4498e-03, -1.4438e-03, -1.4028e-03,
-3.2789e-07, -3.2362e-07, -3.0222e-07,
-3.5161e-07, -3.4508e-07, -3.2091e-07,
-3.4298e-07, -3.3578e-07, -3.1149e-07,
-6.7302e-07, -6.6628e-07, -6.3158e-07,
-7.0989e-07, -6.9877e-07, -6.6241e-07,
-6.9527e-07, -6.8257e-07, -6.4527e-07,
-9.0888e-02,  1.2403e-01, -3.7984e-02,
 2.2664e-01,  2.4879e-01, -4.2899e-02,
 1.4387e-01,  1.8461e-01,  1.1874e-02,
 6.1925e-40,  3.3333e-40,  1.8962e-40,
 6.8176e-40, -1.7566e-40, -3.0456e-40,
 2.7654e-40,  3.8422e-41,  4.9191e-40,
 7.3870e-02, -2.8769e-03,  3.0556e-02,
-4.8431e-02, -9.5185e-02, -2.6638e-02,
-5.0020e-02, -1.9538e-01, -1.1013e-01,
-1.0618e-02,  1.1990e-01,  2.6292e-01,
 2.3825e-02,  5.0637e-02,  3.4663e-01,
-6.6877e-04,  3.9199e-02,  1.2566e-01,
 1.9510e-02, -2.5621e-02,  1.3465e-01,
 1.7514e-02,  4.0338e-01,  3.2779e-01,
-4.4944e-01, -4.2444e-03,  3.9557e-04,
 8.1306e-41,  2.0311e-40,  2.9683e-40,
-5.7636e-40,  4.4291e-40,  4.3356e-40,
-7.1797e-41,  4.5366e-40,  3.9953e-40,
-4.5418e-40,  4.1805e-40, -3.2458e-41,
-9.4881e-41, -8.6365e-41, -1.9294e-40,
 7.1954e-41, -9.8565e-41, -5.5540e-40,
-5.3769e-40,  1.4094e-40, -1.5355e-40,
 8.8038e-41, -3.6848e-40, -1.2237e-40,
-2.8267e-41, -1.7583e-40, -5.9647e-40,
 1.0856e-01,  2.9049e-02, -1.5014e-01,
-1.1248e-01, -1.0547e-01, -1.4007e-02,
 2.2302e-01,  6.1377e-03, -1.7419e-02,
-1.5899e-40, -7.2549e-41, -2.6734e-40,
-6.6477e-40,  6.7206e-40,  4.2694e-40,
 5.2940e-40,  6.8204e-40, -3.7081e-40,
 6.3521e-02, -3.3659e-02, -2.3421e-02,
 1.9463e-01,  5.2135e-02,  1.8343e-02,
 1.6007e-01,  2.7619e-01,  1.5967e-02,
 9.8950e-04,  6.2254e-02, -1.6805e-02,
-3.8226e-02, -1.4132e-01, -3.7778e-02,
-1.5993e-02, -7.9499e-02, -2.5192e-02,
-5.1830e-02,  6.4576e-02, -3.6565e-03,
 8.4070e-02,  7.4203e-01,  1.3715e-01,
-1.4624e-01,  2.0818e-01,  4.0172e-02,
-2.0015e-41,  5.2988e-40,  2.7578e-40,
 4.1051e-40,  1.2834e-40, -3.4898e-40,
-1.1975e-40,  4.2374e-40, -3.0404e-41,
-6.3014e-40,  4.6330e-40, -4.4141e-41,
 2.5442e-41,  5.7456e-40,  2.3848e-40,
-1.0788e-40, -5.0563e-40, -5.3638e-41,
 3.5728e-40,  1.9752e-40,  6.1004e-40,
 2.8189e-41, -6.2151e-40,  1.1807e-41,
 6.5305e-41,  5.2028e-40,  1.3692e-40,
 6.3480e-02, -1.3208e-01, -3.9656e-02,
-3.2394e-01, -3.7300e-01, -8.1601e-02,
-2.6810e-01, -3.1263e-01, -1.3754e-02,
-1.2072e-40,  5.3996e-40, -3.4352e-40,
-8.0996e-41, -3.0208e-40,  3.1848e-40,
-5.6407e-40,  2.4674e-41, -2.1055e-40,
-9.1304e-02,  1.8139e-01, -4.3197e-01,
-7.5471e-02,  4.3650e-01, -4.4140e-02,
-2.1955e-02, -1.1747e-01,  1.0585e-01,
-5.1500e-02, -3.6782e-01,  1.1289e-01,
-2.2684e-02,  1.3262e-01, -1.5189e-02,
-9.9690e-03, -5.2877e-02, -4.1630e-02,
-1.5421e-01,  3.8697e-01, -1.4229e-01,
 1.7696e-01,  3.8244e-02,  4.9903e-01,
 6.0618e-02, -8.3266e-02, -7.5666e-02,
-1.6956e-40,  5.4293e-41, -2.5140e-40,
-3.1995e-40, -4.8337e-40,  2.5539e-40,
-1.1449e-40,  1.9503e-40, -1.7368e-40,
 5.4753e-40,  5.9720e-40, -4.7821e-40,
 3.8830e-40, -6.8801e-40, -2.7163e-40,
-5.3411e-40,  9.9695e-40,  4.3186e-40,
 4.6654e-40, -5.9540e-40, -2.8155e-40,
-1.4801e-40, -1.6945e-40,  1.9723e-40,
 5.8380e-40, -6.1587e-40,  6.6695e-40,
-2.9253e-02, -4.2522e-02, -1.4972e-01,
 8.6500e-02,  2.8199e-01,  1.3170e-02,
-2.0740e-01,  6.7694e-02, -3.6058e-02
}
,)"
R"(
{
 9.5728e-41,  5.3991e-40, -1.3764e-40,
-2.0389e-40,  2.4254e-40,  3.3492e-40,
 6.3100e-40, -7.0223e-40,  5.5850e-40,
 7.9395e-02,  2.3999e-02, -1.4908e-02,
-3.3850e-02,  5.6802e-01, -2.5435e-02,
-5.4835e-02,  5.4795e-02,  5.6436e-03,
 2.2894e-02, -4.1222e-02, -1.1675e-01,
 9.1731e-02,  1.2935e-01,  1.2358e-01,
-4.3578e-02, -1.4830e-02, -4.8830e-02,
-7.4979e-02, -2.0581e-01,  1.4103e-01,
-6.1131e-02,  5.7887e-02, -3.5880e-02,
 1.3952e-02,  4.0207e-03,  1.1924e-02,
 2.5288e-02, -1.3509e-01, -5.9642e-02,
-9.4956e-02,  1.1449e-01,  1.1605e-01,
-1.9422e-02,  6.1253e-02, -1.2697e-02,
 1.0544e-02, -2.0809e-02, -1.2336e-01,
 1.8117e-02, -2.8187e-02,  6.5993e-02,
-5.3879e-03,  7.5185e-02, -1.0908e-01,
 4.5794e-02,  3.4661e-01,  2.5943e-02,
 8.3490e-02,  5.2684e-01,  6.8395e-02,
-8.1413e-02,  1.9585e-01, -8.5180e-02,
 1.2717e-01,  3.9303e-02,  7.7022e-02,
 8.9179e-02,  4.6494e-01, -3.9521e-02,
 3.1977e-03, -7.2812e-03,  6.1050e-02,
 8.3977e-40,  1.5340e-39, -7.9692e-41,
-9.0895e-40, -1.3907e-39, -5.9796e-40,
 3.8209e-40,  1.1667e-39, -9.2501e-41,
 3.2003e-01,  1.5745e-01, -1.9259e-02,
-4.5987e-02,  1.0548e-01, -8.0514e-02,
 2.1304e-01, -3.1380e-01, -1.5369e-01,
 8.5996e-02, -1.3531e-01, -1.0763e-01,
 8.6016e-02,  1.4965e-05,  1.8685e-02,
-1.5417e-01, -1.2926e-01, -3.5453e-02,
-3.8170e-01,  3.8446e-01,  1.8818e-02,
 1.5186e-01,  1.9757e-01, -2.6066e-01,
-3.7839e-02, -8.1798e-02, -1.7627e-01,
-1.8743e-01, -1.2370e-02, -4.3000e-02,
-5.9060e-02,  2.4526e-01,  4.1474e-02,
 2.4157e-02,  7.5712e-03, -2.7695e-02,
 7.8077e-03, -4.2980e-02, -1.1186e-01,
-1.3167e-01, -7.9885e-02, -6.1016e-02,
-4.1150e-03,  5.8770e-02,  1.2713e-01,
 1.3190e-01, -5.6738e-02,  4.0037e-04,
 2.0416e-02, -2.6343e-02, -7.5970e-03,
-4.1940e-02, -3.0681e-05,  4.4473e-02,
 8.0878e-02,  2.7460e-02,  6.7052e-03,
 1.1567e-01,  1.0840e-01,  7.6688e-02,
-8.0028e-02, -5.4912e-02,  1.4908e-02,
-1.6593e-09, -1.6794e-09, -1.5955e-09,
-1.7895e-09, -1.8154e-09, -1.7334e-09,
-1.7501e-09, -1.7741e-09, -1.6998e-09,
-1.8026e-02, -1.6955e-02, -4.2360e-03,
 2.4289e-02,  7.7284e-02, -4.1659e-03,
 1.7088e-02,  6.0831e-02, -9.0745e-02,
 2.6511e-02,  8.8159e-02,  1.0561e-01,
 4.9517e-02,  1.2426e-02,  1.5537e-01,
-1.6286e-02, -2.2565e-01, -2.3849e-02,
-8.3432e-03, -1.4025e-02, -6.2832e-02,
 8.0614e-02,  1.8649e-01, -3.1684e-02,
-3.4939e-03,  1.8873e-03, -1.3997e-01,
-7.0853e-03,  1.5906e-01,  6.7454e-02,
 4.6216e-02,  2.4383e-01,  1.4701e-01,
-7.4514e-02,  6.7570e-02,  9.1006e-02,
-7.4745e-03, -1.1190e-02,  4.5522e-02,
-1.3870e-02, -6.9516e-02, -3.5489e-03,
 3.3374e-02,  4.7561e-02, -1.2599e-02,
-2.1119e-02, -8.7149e-02,  6.5706e-02,
-2.1975e-02,  1.3902e-01, -1.6201e-01,
-6.4217e-02, -2.7866e-02, -4.2819e-02,
 1.0826e-02, -2.2625e-02, -2.6815e-02,
-7.9245e-02,  1.9614e-01,  7.2776e-03,
 3.0953e-02,  2.3800e-02,  4.9492e-02,
-5.0273e-40,  1.2403e-41,  5.8127e-40,
 3.2777e-40, -6.4094e-40,  4.9781e-40,
-7.8534e-40, -4.6311e-40,  1.3330e-40,
-3.0605e-01,  1.7887e-01,  1.0631e-01,
 4.1584e-01,  1.9608e-01, -5.4098e-02,
 2.3902e-01, -1.6948e-01,  1.5523e-01,
 8.6082e-03,  1.8813e-01, -1.2238e-01,
-1.6245e-01, -1.1194e-02, -1.8042e-01,
-1.3526e-01, -3.4242e-02, -1.0507e-01,
-1.4271e-01,  1.9050e-01,  8.6028e-02,
 6.0812e-02, -2.2403e-01,  1.2804e-01,
 4.8136e-03,  3.9310e-02, -5.9199e-02,
 2.0393e-02,  2.6985e-01, -1.8622e-01,
-8.6311e-02, -1.3168e-01, -3.5810e-02,
 2.0536e-02, -9.1108e-02,  1.7130e-01,
-1.8226e-01,  6.8708e-02, -1.5522e-01,
-6.1624e-02, -1.2033e-01,  7.0617e-02,
 2.1330e-02,  9.4178e-02,  8.0328e-02,
-1.3430e-01, -7.5803e-02,  1.7173e-02,
-1.2285e-01, -1.1655e-01, -1.1711e-03,
 5.3530e-02,  1.4974e-01,  1.4624e-01,
-4.5552e-02, -1.1748e-01, -1.5010e-01,
-2.0430e-01, -9.5633e-02, -2.6155e-02,
 4.7479e-02,  1.8316e-02, -7.5710e-02,
 1.8831e-40, -7.6199e-40, -4.7602e-40,
-6.6168e-40,  8.9962e-40, -1.2425e-39,
 7.9973e-40,  1.4644e-40,  5.6365e-40,
-2.0277e-02,  6.6722e-03,  4.7406e-03,
 2.4475e-01,  4.1413e-01,  3.4286e-02,
 8.5284e-02,  2.0259e-01,  1.0259e-01,
-1.4212e-02,  1.0655e-01, -4.1769e-02,
-5.2571e-02,  2.4284e-02, -9.6049e-02,
-1.1090e-01, -6.5825e-02, -1.1855e-01,
-2.1166e-02,  1.6765e-02, -3.3954e-04,
 1.3364e-02,  2.0784e-02,  1.1219e-01,
 1.2310e-02,  1.0528e-01,  1.6728e-01,
 1.6451e-02,  2.0858e-01,  8.9734e-02,
-7.5138e-02,  3.3147e-03, -3.3353e-01,
 5.0920e-02,  1.3846e-01, -1.0723e-01,
 4.9367e-03,  9.0724e-02, -6.5666e-02,
-7.1177e-02, -3.5466e-01, -2.8721e-01,
-1.6778e-02,  2.5639e-02, -1.1711e-02,
-1.3646e-01, -9.5167e-02,  4.1304e-02,
 4.6341e-02,  1.4761e-01,  1.6152e-01,
 6.2647e-04, -7.7656e-02, -8.1906e-02,
 8.1988e-03,  3.7136e-02,  2.3505e-03,
-1.3645e-01,  2.3720e-01, -1.1189e-01,
 1.0462e-01,  1.1042e-01,  3.2753e-02,
-6.7748e-40,  5.9135e-40,  6.6560e-40,
-5.8903e-40, -1.0134e-39, -1.3380e-39,
-6.3604e-40, -4.7563e-40, -1.0222e-40,
 7.0032e-02, -7.9810e-02, -2.7158e-02,
-5.3864e-02,  5.2574e-02,  9.2793e-03,
 2.0356e-02,  1.0320e-02, -1.1818e-02,
 1.6988e-02,  1.0082e-01,  5.1208e-02,
 4.7605e-02,  2.4175e-01,  2.7277e-01,
 2.7711e-01,  1.9202e-01,  3.1184e-01,
-4.6932e-03, -1.1271e-01, -4.9992e-02,
 1.2044e-01,  3.1157e-01,  7.3591e-02,
-1.1665e-01, -1.7981e-01,  8.1636e-02,
-2.0538e-02,  3.6423e-01,  1.4591e-01,
-8.6599e-03, -4.4141e-02, -1.3456e-01,
 5.1546e-02, -3.1135e-01, -4.6444e-02,
 4.8596e-02, -1.3391e-02, -1.0411e-01,
-4.2742e-02, -1.4582e-01,  5.3358e-02,
-2.6172e-01,  2.0343e-02, -1.1556e-01,
 1.0714e-01, -9.3460e-02,  1.1775e-01,
-5.0128e-02, -1.1765e-01, -4.0532e-02,
 1.9679e-01,  9.9964e-02,  8.2307e-02,
 2.6539e-02, -4.7900e-02, -1.6127e-02,
 5.0179e-02,  6.5633e-02,  8.8518e-03,
-7.4343e-02, -3.7296e-02,  8.3110e-02,
 6.9361e-40,  3.6427e-40,  3.0437e-40,
 4.2856e-41, -4.7870e-40,  5.6317e-40,
-2.4673e-40, -6.9736e-41,  8.1050e-41,
 1.4750e-01,  8.4131e-02, -6.8416e-03,
 2.6148e-01,  2.7665e-01, -5.2113e-02,
 1.8354e-01,  1.2722e-01, -4.4965e-03,
 9.9266e-02, -4.9856e-03,  4.9290e-02,
 2.8627e-02, -1.8399e-01,  4.1959e-02,
 4.2181e-03, -6.2869e-03,  5.8692e-03,
-1.6420e-01,  9.8738e-02, -1.1666e-01,
-2.1157e-01,  4.1103e-01, -9.4918e-02,
 1.3203e-04, -1.6331e-01, -9.3356e-02,
-1.3518e-01,  2.0108e-01,  1.0110e-01,
-2.1268e-01, -1.5062e-01,  1.8200e-01,
 4.7569e-02, -1.3939e-01,  1.0215e-01,
 1.7826e-01, -2.4541e-01, -9.1302e-03,
 6.4010e-02, -4.0197e-01, -1.4086e-01,
 3.4038e-02,  1.7450e-01,  7.3735e-02,
 1.0008e-01, -1.4092e-01, -8.4556e-02,
 1.4555e-01, -2.1386e-01, -7.7191e-02,
 1.6096e-01,  3.7982e-02,  5.3724e-02,
 1.6859e-01,  1.7285e-01, -4.6769e-02,
 2.1401e-01,  4.7173e-01,  1.1834e-01,
 2.0381e-01,  2.4755e-01,  7.1882e-02,
-2.1165e-40, -1.9259e-40, -5.0990e-41,
-7.1298e-42, -4.2590e-41,  3.1709e-40,
 4.1065e-40, -4.2585e-41,  3.4243e-40,
-1.0338e-40,  4.6039e-40, -3.3818e-40,
-3.9589e-41,  5.9574e-40, -5.8014e-41,
 1.4505e-41, -3.5326e-40, -3.9806e-40,
 4.2423e-40, -1.7055e-40, -4.9666e-40,
 2.2853e-40, -2.4684e-40, -1.3794e-40,
-5.6764e-40, -1.7905e-40, -5.8915e-40,
-6.5524e-24, -9.0617e-25, -2.1617e-26,
-3.1597e-24, -4.3344e-25, -1.5660e-26,
-1.2193e-25, -2.5639e-26, -1.2715e-27,
-1.7378e-03, -1.7343e-03, -1.7255e-03,
-1.7419e-03, -1.7401e-03, -1.7344e-03,
-1.7368e-03, -1.7380e-03, -1.7349e-03,
-2.1439e-18, -3.1321e-18, -3.8714e-18,
-2.0576e-18, -2.8201e-18, -3.9274e-18,
-1.5047e-18, -1.8838e-18, -2.6202e-18,
-2.9574e-40,  4.0860e-40, -1.5966e-40,
-6.7527e-41,  7.6661e-41, -5.9491e-40,
 3.0843e-40,  8.1079e-41, -2.5140e-40,
-3.7315e-40,  9.4787e-41,  4.6794e-40,
 1.9383e-40,  5.0336e-41,  3.0561e-40,
-5.4286e-40,  5.5999e-40, -4.6977e-40
}
,)"
R"(
{
-1.7797e-01,  4.2792e-03,  1.4593e-02,
-9.5797e-02, -8.9939e-03, -5.1701e-02,
 1.7242e-01, -3.9279e-02, -5.1603e-02,
-9.5551e-02,  1.2364e-01, -1.5859e-02,
-5.2952e-01,  1.9360e-01, -8.8909e-02,
-9.0362e-02,  1.0001e-01, -1.4869e-01,
-1.8118e-01,  1.1365e-01, -2.6290e-02,
-1.7577e-02,  1.4090e-01, -7.0055e-03,
-2.3038e-02,  6.7733e-02, -7.8753e-02,
 4.7683e-02, -2.6399e-02,  1.5766e-02,
-4.3282e-02, -1.5328e-02, -3.4286e-02,
 1.3589e-01,  4.5216e-01,  2.1589e-01,
 2.8358e-02, -7.4942e-02,  1.0399e-03,
 1.8794e-01, -2.0583e-02, -7.5884e-02,
 1.4485e-01, -6.6054e-02,  4.1890e-02,
 1.4217e-01, -4.1486e-01,  2.2503e-03,
 1.6143e-01,  1.1475e-01, -1.7507e-02,
 1.8859e-02, -4.1361e-02, -2.4366e-02,
 1.6599e-01, -1.2791e-01, -1.4340e-03,
 2.9528e-01,  1.7499e-01, -2.6257e-01,
-1.0051e-01,  1.3705e-01,  1.6794e-02,
-3.7351e-40,  6.3010e-40, -1.2071e-40,
-4.6380e-40,  1.8442e-40, -3.5994e-40,
-2.1459e-40, -4.3455e-40, -6.1978e-41,
-2.3638e-40, -4.6965e-40, -3.4232e-40,
-1.6517e-40,  4.7178e-40, -1.6757e-40,
 6.7890e-41, -4.3000e-40,  1.8323e-40,
 4.5416e-40, -2.9010e-40, -1.5200e-40,
-6.4373e-40, -8.7351e-41,  6.5595e-42,
 5.1625e-40, -6.0418e-40, -7.3962e-40,
-9.6854e-07, -9.9337e-07, -9.4368e-07,
-1.0036e-06, -1.0410e-06, -9.9732e-07,
-9.6597e-07, -1.0048e-06, -9.5740e-07,
-6.7698e-40,  6.5977e-40, -1.3264e-40,
 7.2821e-40,  1.3843e-40,  3.0949e-40,
-3.7702e-40,  2.6194e-40,  8.1969e-40,
-3.2283e-40, -5.5487e-40,  5.8744e-40,
 1.6124e-40,  3.3512e-40,  3.1454e-40,
-6.4510e-40, -5.7692e-40,  5.5184e-40,
 6.4218e-40, -4.3187e-40, -6.4629e-40,
 4.9246e-40,  5.9593e-40,  8.3132e-41,
-2.3841e-40, -5.6196e-40, -3.2230e-41,
 4.3824e-40, -3.8344e-40, -9.9086e-42,
-2.9323e-40,  2.1916e-40,  4.4739e-40,
 5.6837e-41,  5.1796e-41, -2.4338e-40,
-2.2853e-40, -3.8920e-40,  6.1587e-40,
-2.9474e-41,  4.6214e-40, -3.6292e-40,
-1.4928e-40, -3.6708e-41,  5.2020e-40,
-5.7654e-09, -1.1785e-08, -8.8001e-09,
-2.9135e-08, -4.5536e-08, -2.9715e-08,
-3.6952e-08, -7.9172e-08, -4.9227e-08,
-4.9138e-40, -9.0061e-42,  4.6251e-40,
-2.9970e-41, -2.5468e-40, -3.4253e-36,
 2.5450e-40, -4.2678e-34, -1.4374e-28,
-8.8253e-04, -8.0017e-04, -9.3414e-04,
-6.8950e-04, -5.6309e-04, -6.7745e-04,
-6.9313e-04, -5.5764e-04, -6.2943e-04,
-3.9701e-33, -9.5883e-33, -3.1777e-37,
-6.9570e-32, -5.3399e-35, -2.3366e-40,
-2.4234e-34,  5.0487e-40, -3.3736e-40,
-6.8199e-10, -3.7572e-10, -8.5291e-13,
-2.4590e-09, -1.3672e-09, -4.6688e-12,
-4.6488e-10, -3.8222e-10, -9.9711e-13,
-2.1094e-10, -5.9622e-10, -4.9438e-10,
-6.1286e-11, -1.0669e-10, -1.5596e-10,
-8.8002e-14, -5.8746e-14, -1.4590e-13,
-6.3794e-15, -1.8271e-11, -4.1128e-10,
-4.9621e-16, -7.2096e-12, -6.9279e-10,
-5.6144e-18, -1.7576e-13, -5.3456e-11,
-2.9789e-40, -4.6452e-40,  1.5649e-40,
-1.8445e-40, -5.2942e-40,  2.5130e-40,
 6.2269e-40,  3.9166e-41, -2.4197e-40,
 9.0760e-02, -5.2868e-03, -2.6395e-02,
-9.9856e-02, -7.3340e-02,  1.3300e-01,
 1.0308e-01, -1.5246e-01,  1.2339e-01,
 4.6725e-02, -2.3542e-01, -1.3863e-02,
 5.7244e-02, -1.5891e-02,  9.8016e-02,
 7.2911e-03, -1.3940e-01, -2.0913e-01,
-1.5888e-01,  6.4212e-02, -2.1546e-01,
-3.4265e-02,  4.9644e-01,  1.3381e-03,
-1.4354e-01,  4.8149e-02, -9.2803e-02,
 1.9610e-01, -1.3477e-01,  1.2191e-01,
 1.6710e-01,  5.8697e-02,  1.1687e-01,
 6.0683e-02, -3.5600e-02, -1.1227e-01,
-8.4316e-02,  1.1657e-01, -4.8695e-02,
-5.2302e-03,  2.5815e-01, -1.8508e-01,
-6.2641e-02,  1.0480e-01,  1.3785e-02,
 4.6731e-02, -1.0164e-01, -1.7670e-01,
-1.7834e-01,  3.0852e-01, -2.5739e-01,
-2.2183e-02, -1.9036e-03, -4.5486e-03,
-4.5824e-02, -4.5149e-02,  5.1053e-01,
-1.9989e-01,  6.8123e-02, -6.2866e-02,
 2.7566e-03, -1.5939e-01, -3.1040e-02,
 5.6501e-40,  5.2828e-40, -5.9773e-40,
-4.3530e-40, -1.1658e-40,  4.9705e-41,
 4.8101e-40,  5.0236e-40,  2.0476e-40,
-1.1467e-01,  1.3332e-01, -1.2336e-01,
 1.4310e-01,  3.7580e-01,  7.0850e-02,
 6.9081e-02,  3.1938e-01, -1.7485e-02,
-4.6274e-02,  6.8184e-02, -2.5255e-02,
 3.5271e-02, -1.3705e-01, -3.5633e-02,
-1.8815e-02, -1.3214e-01,  1.1412e-01,
 2.0662e-02, -3.8208e-02, -7.2247e-02,
-2.1090e-01, -1.6468e-01, -2.3197e-02,
-1.1914e-02, -2.6586e-01,  1.5109e-03,
-2.9195e-02,  4.9515e-02,  4.2538e-02,
 1.4730e-01,  1.6152e-02, -1.0880e-01,
 5.6958e-02, -2.1066e-01,  2.2279e-02,
-2.4039e-02, -4.9974e-02,  1.0838e-01,
 1.7474e-01, -1.3279e-03, -1.6419e-02,
 7.7142e-02, -9.5464e-02,  1.0625e-03,
 2.2538e-02,  1.4066e-01,  1.0608e-01,
-5.5728e-02, -3.0788e-01, -5.7340e-02,
 1.0889e-01, -8.3718e-02,  4.2279e-02,
 1.8534e-02,  1.8143e-01, -3.2056e-03,
-1.1901e-02, -2.1405e-01,  9.9680e-02,
-6.1342e-02, -2.4007e-02,  6.2649e-03,
-2.7320e-40, -1.3336e-40,  2.4202e-41,
-7.1225e-41,  1.2848e-40,  1.5426e-40,
-4.2798e-40,  6.3079e-40,  6.2629e-40,
 1.7236e-01, -1.7372e-01, -2.4310e-02,
-2.9287e-01,  1.1828e-01,  7.9138e-02,
-4.6967e-03, -1.9892e-01,  5.6266e-02,
-3.1126e-02, -1.5562e-01,  4.9946e-02,
-1.7629e-02,  2.0615e-01, -2.8133e-01,
-5.1562e-02,  8.5690e-02, -7.4489e-02,
 8.9529e-02, -2.3073e-01, -1.9610e-01,
 3.1305e-01,  1.5354e-01, -1.9586e-01,
-1.7986e-02,  6.8774e-02,  9.4214e-03,
 8.2274e-02,  1.0623e-02,  7.9213e-02,
-1.4599e-01,  4.6377e-01, -3.1812e-01,
-1.0848e-01,  6.5773e-02, -2.3308e-02,
-1.1759e-01, -3.9590e-03,  2.1528e-02,
-3.1803e-01,  2.0293e-01, -1.7350e-01,
-1.0695e-02,  1.3857e-01, -6.2197e-02,
-3.5472e-02, -1.9470e-01,  2.7219e-01,
 5.7694e-02,  8.0505e-02, -2.3929e-01,
-3.8329e-02,  2.4917e-02, -1.6319e-02,
 1.1357e-01,  1.6542e-01,  3.6216e-01,
 8.2936e-02,  4.3700e-01,  3.0351e-01,
-5.2152e-02,  6.2574e-02,  1.2403e-01,
 2.1644e-40,  7.2343e-41,  5.5580e-40,
-4.3927e-40,  5.0561e-40, -1.5560e-41,
-3.2783e-40, -8.8219e-41,  5.4415e-40,
-7.0347e-02, -3.6123e-02, -2.6558e-02,
 1.0338e-01,  2.1157e-01, -1.6694e-01,
-1.2414e-01, -2.3954e-01, -7.4278e-02,
-3.2513e-02, -1.5172e-02,  8.7642e-02,
-2.4394e-02, -4.8092e-01,  2.5011e-02,
 6.4874e-02,  3.5122e-02, -2.9093e-01,
-2.8825e-01, -7.0915e-02,  7.8353e-02,
-1.4449e-01,  4.0775e-02,  6.2470e-02,
 4.5757e-03,  2.1859e-01,  9.0181e-02,
-1.0668e-01,  8.4464e-02,  3.0633e-02,
 3.2457e-01, -2.8182e-01,  4.2576e-02,
 8.3366e-02,  2.6827e-01, -1.3106e-01,
 3.8893e-01,  3.2303e-01,  1.0759e-01,
 7.4688e-02, -2.0267e-01,  1.3278e-01,
 3.9157e-02,  2.4545e-01, -8.3322e-04,
-1.0137e-01,  1.9100e-01, -7.9111e-02,
 6.0946e-03,  6.9015e-02,  1.3956e-01,
-2.0919e-02,  4.7313e-02, -2.0762e-03,
-5.2454e-03,  2.2381e-01, -8.3334e-02,
-1.5565e-01,  2.7169e-01, -2.7854e-02,
-3.0081e-03,  1.7734e-02, -8.7189e-02,
 2.1541e-40, -6.7487e-40,  1.7678e-40,
-3.9857e-40, -1.1965e-40, -8.6754e-41,
-4.0721e-40,  2.2073e-41,  4.2728e-40,
-1.0348e-02,  5.4534e-02, -1.6843e-02,
-5.9143e-02,  2.3817e-01, -8.0245e-02,
 2.4049e-02, -1.3761e-01, -3.1000e-02,
-9.3572e-03,  2.3699e-02, -6.8689e-02,
 2.1178e-02,  8.9732e-02,  1.0031e-01,
-1.0051e-02,  4.6489e-02, -1.5705e-01,
 8.5281e-02, -3.3804e-03,  1.4080e-02,
 9.0800e-02,  2.3685e-01,  1.8100e-01,
 2.8387e-02,  4.4003e-02,  2.3241e-02,
 3.8895e-02,  5.6259e-02, -5.5126e-03,
 3.6259e-02, -2.6326e-01,  2.1181e-01,
-4.0062e-02,  1.7112e-01, -6.7841e-02,
-1.1844e-01, -4.3610e-02,  2.0195e-02,
-1.4826e-01,  4.1095e-02,  1.5936e-01,
-4.5349e-02,  1.4110e-01, -2.0911e-02,
-5.9909e-04, -2.7910e-01, -8.8665e-02,
-1.7786e-02,  1.8444e-01,  1.1057e-01,
-2.7502e-03,  6.3586e-02, -4.7819e-03,
 3.2120e-02, -8.9057e-02, -2.1850e-02,
 8.1681e-02,  3.3128e-01, -2.0586e-01,
-3.5788e-02,  4.9861e-02, -9.2399e-02,
-3.4012e-04, -3.4292e-04, -3.3847e-04,
-3.4439e-04, -3.4796e-04, -3.4421e-04,
-3.3919e-04, -3.4336e-04, -3.4058e-04
}
,)"
R"(
{
-1.3657e-02,  1.1156e-01,  7.2200e-02,
 1.1480e-02, -8.4944e-01,  8.4269e-02,
-3.9708e-02, -2.2843e-02,  9.6383e-03,
 2.6648e-40,  9.1590e-41,  6.7161e-40,
-6.6959e-40, -2.5194e-40, -2.0946e-40,
 3.6800e-40, -1.1584e-40,  6.2195e-40,
-1.3560e-41, -8.0151e-41,  4.4048e-40,
-4.1209e-40,  2.7411e-40,  3.2419e-40,
 5.8333e-40,  1.1503e-40, -5.0783e-40,
-5.5529e-02, -2.5359e-02,  5.0955e-02,
-2.5656e-01,  1.6277e-01, -7.8177e-02,
 4.0336e-01,  2.9204e-02, -1.4069e-01,
 9.0289e-03,  5.0162e-02,  1.7636e-03,
-8.2033e-02, -1.9730e-01, -2.9780e-02,
 2.1504e-01,  6.8211e-02, -4.3901e-02,
-1.9179e-02, -2.0895e-02, -1.9322e-02,
 3.6823e-01, -5.3238e-02,  9.7242e-02,
 2.8175e-01, -1.4789e-01,  3.4802e-02,
 2.3631e-02,  4.1018e-02,  3.5676e-02,
-8.6522e-02,  1.9012e-01, -1.6566e-01,
-1.1459e-02,  3.2106e-01, -1.5671e-01,
 2.6787e-01,  5.2008e-02,  1.3043e-02,
-1.1212e-01, -1.7870e-01,  1.1380e-01,
-1.3808e-01, -1.8231e-02, -1.7496e-02,
 1.6126e-02, -2.3455e-01,  1.0819e-02,
-6.4042e-02,  8.0989e-02, -2.0084e-01,
-1.6731e-02, -6.2462e-02,  3.4582e-02,
-8.3275e-40, -7.1013e-40, -6.8322e-40,
 2.3575e-40,  5.8301e-41, -3.7432e-40,
-3.6291e-40,  6.6980e-40,  1.4574e-40,
-4.3792e-40, -2.5814e-40, -3.4986e-41,
-6.5130e-40, -4.4757e-40,  6.8536e-40,
 4.7222e-40, -7.3197e-41, -3.4635e-40,
 5.2032e-02,  7.9704e-02,  4.3231e-02,
-2.1183e-01,  1.9694e-01,  9.0796e-03,
 2.6422e-02,  2.6500e-02,  7.8346e-02,
 3.0334e-01, -1.3751e-01, -2.1607e-02,
-1.1428e-01, -7.4824e-02,  1.1323e-02,
-1.0918e-02,  5.0459e-02, -7.1639e-02,
-3.1260e-02, -2.0389e-01, -3.9946e-02,
 2.8608e-01,  1.2299e-01, -1.9546e-02,
-2.4429e-01, -3.8619e-02, -5.3084e-02,
-2.2943e-01, -2.4008e-02,  1.1432e-01,
 2.1442e-01,  3.2855e-01, -2.8829e-01,
 2.2023e-02, -5.8598e-02, -8.9337e-02,
-3.0052e-01, -1.8402e-01,  1.2089e-02,
 2.1355e-02,  2.9582e-01,  1.2974e-01,
 6.8963e-02,  8.1491e-03, -6.0322e-02,
 2.9124e-03, -1.6586e-02,  1.1665e-02,
-9.2850e-03, -9.9084e-02, -3.3469e-01,
-7.8724e-03, -1.4394e-02, -3.5968e-02,
 3.8770e-40, -2.8639e-40,  4.6953e-40,
 4.2631e-40,  6.2568e-41, -5.3500e-40,
-2.1987e-40,  1.3435e-40,  4.4101e-40,
-3.9973e-40,  6.3046e-40,  1.6046e-40,
 4.4338e-40,  1.6940e-41,  4.1598e-40,
 2.6132e-40, -2.9888e-40, -7.5708e-41,
-1.5834e-02,  8.3954e-02, -6.2336e-02,
-2.1114e-03,  4.3267e-02, -7.8905e-02,
-1.0977e-01,  6.7621e-01, -1.7418e-01,
-4.1036e-02, -8.1151e-02,  1.2406e-01,
 4.0172e-02, -9.1061e-02, -1.1172e-01,
-4.2933e-03, -3.9701e-03,  1.3028e-01,
 8.2880e-03, -6.3965e-02,  1.2244e-02,
-3.0537e-02, -4.8395e-02,  4.8473e-02,
-1.9965e-05,  2.1817e-01,  4.1274e-03,
-7.8153e-03, -2.6079e-02,  1.7038e-02,
-5.8247e-03,  1.3071e-01, -1.3396e-01,
 1.9204e-02,  4.2170e-02,  2.7013e-01,
-3.5441e-02, -8.4752e-03,  5.9933e-02,
 5.1357e-02,  4.6991e-01, -1.1519e-02,
 2.2325e-02, -1.8482e-01, -3.7481e-01,
-8.3646e-02,  9.7678e-02, -6.1224e-02,
 2.7994e-02,  8.8978e-02,  1.3171e-02,
 3.1002e-03,  1.7363e-02,  5.7592e-03,
-3.2792e-08, -3.3583e-08, -3.0789e-08,
-3.3877e-08, -3.4913e-08, -3.2361e-08,
-3.1342e-08, -3.2503e-08, -3.0403e-08,
 6.8578e-08,  9.2142e-08,  9.1126e-08,
 9.4299e-08,  1.1502e-07,  1.1535e-07,
 9.3175e-08,  1.1184e-07,  9.9799e-08,
-4.6616e-02, -2.4899e-02, -1.8697e-02,
 1.0603e-01, -3.9992e-03, -2.2587e-02,
-5.1283e-02, -1.6369e-01,  8.4375e-03,
-8.1947e-02,  4.2694e-02, -1.1240e-03,
-5.5653e-02, -1.1285e-01, -4.5949e-02,
-1.1032e-01,  2.1335e-02, -4.5373e-02,
-1.3731e-02, -2.7043e-03,  2.6805e-02,
-2.0211e-01,  4.3257e-01,  2.9983e-02,
-2.1784e-01, -2.7302e-02,  6.3212e-03,
 5.0083e-02,  1.5504e-01, -5.7119e-02,
 4.8236e-02,  3.7695e-02, -1.3592e-01,
-1.0103e-02, -5.7685e-02, -7.2164e-02,
 1.0287e-01,  7.4816e-02,  7.4852e-02,
 8.3892e-02,  1.5851e-01,  2.9315e-02,
 6.5781e-02,  7.7821e-02,  3.5852e-02,
-1.6637e-01, -2.3309e-01,  4.6097e-02,
 6.3994e-02, -3.8186e-01,  3.3319e-02,
-3.4832e-02, -2.2820e-02,  2.9905e-02,
-3.8813e-41, -2.8626e-40, -9.0218e-41,
 4.1216e-40, -4.4215e-40,  6.9768e-40,
 5.6281e-40,  2.0477e-40,  2.7797e-40,
-4.4903e-40, -6.2574e-41,  4.9971e-40,
 5.0135e-40, -3.1945e-40, -2.4694e-40,
 2.6587e-40, -4.9583e-40,  6.3300e-40,
 3.7396e-02,  5.0982e-04, -2.0477e-02,
-3.6219e-01, -5.4041e-02, -5.0102e-02,
 2.5413e-01,  2.4204e-01,  1.1825e-02,
 5.4188e-02, -4.7746e-02,  2.9555e-02,
 9.0741e-04, -2.3564e-02, -2.3329e-02,
 6.1327e-02,  1.2302e-04, -4.9922e-02,
 5.5537e-02, -1.6758e-02, -4.9906e-04,
 4.8292e-01, -1.1133e-01,  3.9674e-02,
 5.3155e-02, -2.0619e-03, -8.6256e-03,
-1.0191e-02, -1.8468e-01,  1.9335e-02,
 6.8215e-02,  7.6345e-01, -8.4551e-02,
-5.5900e-02, -1.8952e-01, -1.5742e-01,
 2.0310e-01,  7.3795e-02,  3.2519e-02,
-1.8205e-01, -1.9443e-01, -2.5689e-03,
-3.4457e-02,  3.4153e-02,  5.9127e-02,
 8.7102e-02,  9.9942e-02,  1.0179e-02,
-9.3659e-02, -3.1806e-01, -2.5840e-02,
-2.0781e-02, -2.3140e-02, -4.3124e-02,
-6.8758e-40,  6.8751e-40,  4.8747e-41,
 7.5546e-40,  6.1463e-40,  1.4176e-40,
-1.5286e-40,  7.0593e-40,  7.2032e-41,
-6.0758e-40, -3.6200e-40,  1.2123e-40,
 1.3627e-40,  3.2983e-40,  3.6171e-40,
-4.2148e-40,  1.1102e-40,  3.2714e-40,
-3.3770e-02, -2.9005e-02,  3.0949e-02,
-2.0980e-01,  1.3734e-01, -7.0116e-03,
-1.6143e-01,  1.0809e-01, -6.5394e-02,
 2.3822e-02,  1.2946e-03,  4.9062e-02,
-1.3962e-01, -1.0316e-02, -7.7881e-02,
 1.0840e-01,  1.2561e-01, -7.4837e-02,
-1.4141e-02, -2.5462e-02, -2.9232e-02,
 3.4793e-01,  7.2506e-02,  2.4321e-02,
 3.9514e-01,  7.3758e-05,  1.9667e-02,
-1.6211e-02,  5.2985e-02,  6.1726e-02,
-1.8887e-01,  2.7585e-01, -1.9595e-01,
 8.1957e-02,  9.4834e-02, -1.2457e-01,
 2.7614e-01,  1.1522e-01,  1.6558e-02,
-5.1675e-01,  1.4739e-01,  7.5779e-02,
-6.9298e-02, -9.3419e-02,  3.4919e-02,
 1.4700e-02, -8.0093e-02,  7.9963e-02,
 3.6180e-02, -1.3603e-01, -2.2494e-02,
-6.2491e-03,  7.0200e-02,  6.5341e-02,
-4.5940e-41,  2.5437e-40, -3.3111e-40,
 5.9661e-40,  6.2521e-40,  5.6418e-40,
 1.9187e-40, -5.8872e-40,  5.5747e-40,
-7.2827e-08, -9.8105e-08, -7.6474e-08,
-1.0102e-07, -1.3308e-07, -9.4260e-08,
-6.0778e-08, -7.7608e-08, -5.8082e-08,
 6.0874e-03, -2.7664e-03, -8.9427e-03,
-2.7690e-02, -8.0226e-02,  3.4118e-02,
 4.6862e-04, -3.3901e-02,  1.0307e-02,
 4.9866e-03,  1.4045e-02, -3.4145e-02,
 4.8688e-02,  6.3394e-01, -3.6632e-02,
-1.3692e-01,  2.6015e-02, -4.8379e-02,
 3.0737e-02,  6.6564e-02,  1.9202e-02,
 6.9903e-02, -1.7049e-01, -3.0475e-02,
 9.7842e-02,  9.5358e-03, -6.6992e-03,
 5.0058e-02, -6.4318e-03, -4.2592e-02,
-6.1522e-02, -9.2671e-02,  9.5445e-02,
-1.0664e-02,  9.0606e-03,  4.9525e-02,
-1.9478e-02, -1.2012e-01,  8.6040e-03,
-1.5761e-01, -2.1139e-01, -5.0805e-02,
-3.1109e-02, -6.5970e-02, -5.7202e-02,
-9.6879e-38,  5.7263e-40,  6.2889e-40,
-2.9557e-35, -1.9346e-37, -1.7152e-40,
-1.4661e-34, -2.3948e-36, -1.9343e-40,
-1.7352e-39, -1.8069e-39,  1.6512e-39,
-8.0461e-40, -1.2666e-39,  1.4288e-39,
 1.0588e-39, -1.7928e-39,  1.1679e-39,
 2.2827e-40, -5.4515e-41, -4.1768e-40,
 3.9033e-40,  6.1988e-41,  5.9877e-40,
-4.3355e-41, -5.1088e-40,  5.9845e-40,
-4.8238e-40, -1.8586e-40,  4.8699e-40,
-9.7225e-41,  4.3387e-40, -4.3683e-40,
-7.9278e-41, -5.3614e-40,  2.1911e-40,
-3.3982e-40, -5.3335e-40,  3.8540e-40,
 1.9051e-40, -2.0840e-40,  2.2868e-40,
-3.5020e-40, -3.4276e-40,  2.7395e-42,
 3.9197e-40,  6.1843e-40, -1.5888e-40,
 4.3516e-40, -6.1852e-40, -5.3692e-40,
-4.3268e-40,  3.5154e-40,  3.4477e-40,
-4.8414e-40,  2.2647e-40, -2.5591e-40,
 4.6326e-40, -3.0462e-40,  4.7817e-40,
 6.3219e-40, -5.3425e-40, -2.9848e-40,
-3.0476e-04, -3.1119e-04, -3.0074e-04,
-3.0543e-04, -3.1289e-04, -3.0298e-04,
-2.9241e-04, -2.9994e-04, -2.9138e-04
}
,)"
R"(
{
-2.2886e-02,  3.0069e-02,  1.2279e-01,
-2.9607e-02, -1.3458e-02, -1.8369e-01,
-1.7976e-02, -1.2919e-01, -6.6546e-02,
 7.6594e-04, -5.1065e-02,  1.9211e-05,
-1.0156e-02,  2.2367e-01, -1.2114e-01,
-2.3259e-02,  3.1726e-02, -9.7583e-03,
-3.0406e-01, -8.5172e-02, -1.4132e-02,
 1.2509e-01, -1.6910e-02, -3.6749e-02,
-8.8396e-02, -6.1596e-02, -2.1031e-02,
 1.3706e-02,  1.2434e-02, -4.3578e-02,
 6.4836e-02,  3.7039e-01,  3.8083e-02,
-1.3868e-02, -2.4857e-01, -2.5418e-01,
-3.6083e-03, -1.4006e-01, -4.8300e-02,
 5.5218e-02, -3.4282e-02, -2.1022e-02,
 3.7147e-02,  1.0268e-01, -7.1269e-02,
 1.6049e-02,  3.5472e-02, -1.9878e-02,
 2.8277e-02,  1.3740e-01,  6.6502e-03,
 3.8955e-03, -1.1759e-01,  4.3889e-02,
 6.8744e-03,  1.6272e-02,  1.6642e-02,
-8.8544e-02, -1.0881e-01, -8.9627e-02,
 3.2759e-02, -2.6696e-02, -8.4958e-02,
-1.6779e-40, -6.3646e-41, -6.2486e-40,
 2.3154e-40,  2.8049e-40,  3.7718e-40,
-3.3950e-40, -3.1501e-40,  5.8709e-40,
 2.0449e-02, -4.3580e-01,  1.5969e-02,
 3.4675e-02,  1.9941e-01, -8.0873e-02,
-3.2384e-02,  5.6344e-02,  8.9692e-03,
-1.1170e-02, -1.2673e-02,  4.3187e-02,
 5.5547e-02, -2.2235e-01,  2.4216e-01,
 4.0400e-02,  1.8354e-01, -1.4474e-01,
 2.6906e-01,  2.2741e-01, -2.4673e-03,
-6.9062e-02, -1.2156e-01,  1.6729e-02,
 2.4439e-02, -1.1874e-02, -1.4035e-02,
-1.2263e-02,  4.1108e-02,  4.1866e-02,
 1.8895e-02, -2.5243e-01,  9.7463e-02,
-4.1239e-02,  4.3700e-03,  7.5675e-02,
-2.0316e-02,  2.3573e-01,  2.3418e-02,
 3.3424e-02, -4.7173e-02,  3.0255e-02,
 7.4843e-03, -1.1335e-01, -8.8175e-02,
-2.3941e-02,  5.3023e-02, -7.2781e-02,
-3.7517e-02, -2.1453e-01,  5.4563e-04,
-4.3044e-02, -1.9716e-02, -1.0634e-01,
-1.3623e-01, -1.7242e-01, -3.6714e-02,
 1.1179e-01,  1.1549e-01,  9.4917e-02,
 2.3988e-02, -8.2220e-03, -3.5372e-02,
-2.3917e-40, -4.1869e-41,  3.7775e-41,
 2.8931e-40, -9.4850e-41,  2.5694e-40,
 3.3549e-40, -2.4334e-40, -5.5933e-41,
-1.9716e-02,  2.0360e-02, -4.5978e-02,
 2.5402e-02, -7.1254e-01,  4.7465e-02,
-6.2473e-03,  5.1689e-02,  2.3126e-02,
-5.0558e-03,  3.2192e-02,  1.1107e-02,
-4.7049e-03,  2.5278e-01,  5.9237e-02,
 2.8581e-02, -1.1434e-01, -2.4136e-01,
-6.9507e-02,  6.8347e-03, -1.6335e-02,
 1.0987e-01,  1.8080e-01, -5.7867e-03,
 9.3656e-02,  7.3564e-02,  2.4958e-02,
 5.6972e-02,  4.0580e-02,  2.6332e-03,
 9.4719e-02,  2.5239e-01,  6.9955e-02,
 4.4847e-02,  1.8087e-01,  1.5088e-01,
 2.1314e-02, -4.7133e-02,  2.2706e-02,
-5.6287e-03,  1.4596e-01,  2.0964e-01,
 2.8964e-02, -7.1557e-02,  1.5075e-02,
 1.0833e-02, -1.1626e-02,  2.8292e-02,
-1.0158e-02,  9.8859e-02, -4.0897e-01,
 2.7369e-02,  7.4120e-02,  5.7262e-02,
 2.8656e-02, -7.5623e-02, -9.0947e-03,
-1.2960e-01, -5.4158e-01, -1.3588e-01,
-1.2901e-03, -1.4325e-01,  8.2858e-03,
-5.9278e-06, -7.2368e-06, -7.5409e-06,
-5.4296e-06, -6.6148e-06, -6.9823e-06,
-4.8972e-06, -5.8767e-06, -6.1659e-06,
 1.5370e-02, -1.8240e-01, -8.5260e-02,
-4.0944e-02,  2.4427e-01, -5.7881e-02,
-2.7119e-02,  5.5116e-03,  3.5015e-02,
-2.9458e-02,  2.5316e-02,  8.8560e-02,
 2.2953e-02,  2.3985e-01,  5.0543e-02,
 1.4562e-02, -6.5178e-02,  1.3463e-02,
 1.3083e-01,  1.2521e-02, -1.3636e-02,
 1.0879e-01,  1.6766e-01, -3.2611e-02,
 6.3338e-02,  3.9920e-02,  3.9185e-02,
-6.1213e-03,  5.0655e-02, -1.0811e-01,
 1.5786e-02,  1.6093e-01, -6.1902e-02,
-5.6798e-02, -1.0651e-02, -3.8003e-02,
-3.8947e-02,  7.9906e-02,  2.1873e-02,
 3.8091e-02, -6.3571e-01,  2.3666e-01,
-8.6361e-03,  6.2839e-02, -8.0551e-02,
-1.7522e-02,  8.0021e-02,  6.6860e-02,
-2.1659e-03,  1.1946e-01,  3.1698e-02,
 9.8994e-03,  3.1685e-02,  7.7209e-02,
 5.9570e-02, -1.2125e-01,  7.8625e-02,
-2.6125e-02, -4.1262e-01,  2.2064e-02,
 3.4458e-02,  3.6127e-02, -5.7792e-02,
-1.2149e-26, -1.5445e-26, -1.3458e-26,
-6.7449e-27, -8.8317e-27, -8.3289e-27,
-3.4386e-27, -4.3329e-27, -4.3255e-27,
-1.5780e-02, -2.6614e-02,  1.4318e-01,
 1.3537e-02, -2.3703e-01,  8.4689e-02,
-8.7172e-03, -2.8145e-03,  7.8048e-03,
 7.5935e-04,  2.1485e-02,  3.1295e-02,
-1.0616e-02,  3.0246e-01,  5.1059e-02,
 1.7473e-02, -1.3711e-01,  9.1028e-02,
-1.5658e-01,  9.8222e-02, -3.8660e-04,
 3.1492e-01, -2.8222e-02,  2.2849e-03,
 1.8981e-02,  5.9734e-03, -2.4251e-03,
 5.2160e-02,  6.7007e-02,  7.6489e-03,
 2.5086e-02,  2.8090e-01,  2.9873e-02,
 2.2436e-02,  1.4690e-03, -7.4726e-02,
 1.8654e-02,  3.9925e-02,  4.6807e-02,
-2.2763e-02, -1.4600e-01, -3.8916e-01,
 2.3622e-02, -4.3300e-02,  6.5899e-02,
-6.7366e-03, -6.4845e-02, -1.8648e-01,
 4.7096e-03,  2.0520e-01,  2.9793e-01,
 1.7569e-02,  1.0420e-02, -9.7611e-02,
 9.0317e-05, -2.1156e-01, -4.2835e-02,
-4.2504e-02, -2.8200e-01,  2.9541e-02,
-1.4953e-03,  1.1330e-02,  2.3501e-03,
-2.5191e-12, -4.6044e-12, -5.0983e-12,
-2.4663e-12, -4.4377e-12, -5.1064e-12,
-1.7225e-12, -2.8429e-12, -3.4459e-12,
 4.5889e-02,  8.9519e-02,  9.6885e-03,
-3.7171e-02,  1.9273e-01, -1.2944e-02,
-4.5066e-02,  1.1940e-01,  7.7917e-03,
-3.7940e-02, -1.7852e-02,  1.6174e-01,
-2.7138e-02,  1.7405e-01,  1.2757e-01,
 1.7575e-02,  1.7501e-01, -1.7261e-01,
-3.0167e-01,  2.1264e-01, -5.4101e-02,
 5.7181e-02,  3.2086e-01, -2.5189e-02,
 6.8791e-02,  7.4073e-02,  6.2485e-02,
-1.1685e-01,  2.6102e-02, -1.3473e-01,
-1.6078e-01,  1.9415e-01,  1.1456e-01,
-7.3770e-02, -7.7910e-02,  7.5032e-03,
-9.9791e-02,  1.3601e-01,  2.1827e-02,
-8.6410e-02,  5.4442e-01, -4.0614e-02,
-3.3447e-02, -1.5097e-01, -3.8832e-02,
-3.3858e-02, -3.2723e-02, -9.5508e-02,
-6.6866e-03,  2.2605e-01, -2.4545e-02,
 1.5539e-02,  1.3255e-01,  1.6304e-01,
 9.2646e-02, -8.3526e-02,  7.9621e-02,
 1.2571e-01, -6.1765e-01, -1.0857e-01,
 6.3326e-02, -4.4147e-02, -3.1851e-02,
-1.1289e-17, -2.8354e-17, -4.2437e-17,
-8.5476e-18, -2.0919e-17, -2.8530e-17,
-6.5004e-18, -1.4187e-17, -1.6725e-17,
-6.6543e-03, -7.3452e-02,  5.6590e-02,
-1.1079e-02,  5.6300e-01,  4.8181e-02,
 1.3203e-02, -3.3697e-02, -5.9626e-02,
 9.5321e-04,  3.3130e-02, -1.1170e-02,
-4.0731e-02,  1.9926e-01,  1.4038e-01,
-1.9691e-02, -4.4156e-02, -2.0302e-01,
-3.1023e-01,  2.5532e-02, -1.8600e-02,
-8.8340e-04,  1.2686e-01,  2.9801e-02,
-3.8998e-02,  1.0548e-02, -1.0130e-02,
 1.3619e-02,  1.7058e-01,  8.9738e-02,
 6.6881e-02,  2.6444e-01,  1.7424e-01,
-5.5972e-04,  1.7636e-02,  1.3105e-01,
-2.8910e-03,  9.0412e-02, -3.2259e-02,
-9.4915e-03, -4.4019e-01,  5.4240e-02,
 9.4677e-04,  9.9924e-02,  3.9129e-02,
 3.0629e-02, -1.1738e-01, -7.5613e-02,
 2.8720e-02,  2.1297e-01, -4.0424e-01,
-8.1913e-03,  8.2714e-02,  4.5838e-02,
-1.1364e-01, -1.7665e-01, -5.7729e-02,
 8.0016e-02, -4.2798e-01, -1.0805e-01,
-3.0152e-02,  1.3113e-03, -2.0691e-03,
-4.9919e-07, -6.2994e-07, -6.8287e-07,
-4.3973e-07, -5.4969e-07, -5.9146e-07,
-3.9415e-07, -4.8158e-07, -5.0845e-07,
 2.9395e-02, -5.4668e-02, -1.0222e-01,
-1.5872e-02, -2.8545e-01,  1.9104e-01,
-2.5504e-02,  9.7708e-02, -1.4674e-02,
-2.9208e-02,  1.6061e-02,  2.6569e-02,
-8.3383e-02,  7.9743e-02, -4.7974e-02,
-3.2278e-02, -2.1418e-02,  1.2001e-02,
 1.4024e-01, -3.3952e-01,  6.0963e-03,
 2.4923e-01, -1.8617e-01, -2.2086e-03,
 3.2181e-02, -1.1813e-02, -1.8727e-02,
-1.3008e-01,  1.4290e-01, -5.5369e-02,
-3.5037e-02,  1.0950e-01, -8.0151e-02,
-6.2968e-03,  7.6215e-02, -7.4713e-02,
-8.1088e-02,  1.3237e-01, -7.1621e-02,
-6.2594e-02,  1.7229e-01,  4.0298e-01,
-6.3518e-03, -8.2744e-02, -2.1081e-02,
 1.6270e-02, -3.1866e-02,  5.8604e-02,
 2.6518e-02,  2.9905e-01, -3.5746e-02,
 1.0974e-02,  2.0761e-01, -2.1822e-02,
-4.9122e-02,  2.8042e-03,  8.5798e-02,
 8.6881e-02, -5.8088e-01,  3.4525e-01,
 4.9485e-02, -1.0452e-01,  4.1507e-02,
 2.5428e-40, -4.4558e-40, -2.2090e-40,
-2.9727e-40, -4.8454e-40,  3.0397e-40,
 1.1696e-40, -3.3028e-40, -2.2959e-40
}
};
)" + std::string(
R"(
__constant float biasL[8][8] = 
{
{
-1.0979e-04, -3.8216e-01, -3.3859e-03, -5.8220e-02,  9.0103e-04, -3.0372e-02, -5.6709e-04, -7.4283e-02
}
,
{
-1.7705e-01, -9.9115e-04, -4.5030e-38, -1.1432e-03, -1.0626e-02, -4.0294e-39,  1.5228e-02,  2.6352e-02
}
,
{
-5.4865e-04,  5.6345e-03, -3.4892e-04,  3.2401e-01, -2.7480e-04, -8.3964e-20, -8.7793e-22, -3.2553e-02
}
,
{
-9.5587e-11, -1.2587e-02, -1.9221e-02, -1.8023e-02, -2.9386e-02, -4.1898e-02,  7.6962e-02, -1.1523e-01
}
,
{
-0.0274, -0.0207,  0.0067, -0.0155, -0.0104, -0.0107, -0.0040, -0.0018
}
,
{
-1.3710e-03, -1.1058e-08, -8.6308e-04,  6.7664e-03,  1.3569e-01, -3.5797e-03, -6.9701e-03, -7.2400e-03
}
,
{
-3.3854e-03, -3.4835e-03, -6.1020e-03,  2.0489e-01, -1.3682e-03, -2.6996e-03,  5.1358e-01, -1.1836e-04
}
,
{
0.0326, 0.0135, 0.6751, 0.2662, 0.3646, 0.3592, 0.5598, 0.0823
}
};

__constant float kernelsL10[4 * 8] = 
{
 0.0878,  0.0418,
 0.3778,  0.4755,
-0.3220, -0.4874,
-0.0400,  0.0508,
 0.1034,  0.0174,
 0.5012,  0.3901,
 0.3621, -0.1646,
-0.1304,  0.0012,
 0.2230,  0.3027,
 0.1619, -0.4512,
-0.2099,  0.1890,
-0.0327,  0.1432,
 0.2421,  0.3364,
-0.0938,  0.3157,
 0.1137, -0.2162,
 0.2269, -0.1285
};


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
}
)"
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
}
)"
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
}
)");
#endif // BUILT_IN_KERNEL
