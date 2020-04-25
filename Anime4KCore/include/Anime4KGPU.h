#pragma once
#include "Anime4K.h"

#include<fstream>
#include<queue>

#include<CL/cl.h>

class Anime4KGPU :
    public Anime4K
{
public:
    Anime4KGPU(
        int passes = 2,
        int pushColorCount = 2,
        double strengthColor = 0.3,
        double strengthGradient = 1.0,
        double zoomFactor = 2.0,
        bool fastMode = false,
        bool videoMode = false,
        bool PreProcessing = false,
        bool postProcessing = false,
        uint8_t preFilters = 40,
        uint8_t postFilters = 40,
        unsigned int maxThreads = std::thread::hardware_concurrency()
    );
    ~Anime4KGPU();

    virtual void process();
protected:
    void runKernel(cv::InputArray img);
    void runKernelForVideo(std::queue<cv::Mat> &frames, std::queue<cv::Mat>& results);
    void initOpenCL();
    void releaseOpenCL();
    std::string readKernel(const std::string &fileName);
private:
    cl_context context;
    cl_command_queue commandQueue;
    cl_program program;
    cl_device_id device;
    cl_kernel kernelGetGray;
    cl_kernel kernelPushColor;
    cl_kernel kernelGetGradient;
    cl_kernel kernelPushGradient;

    cl_image_format format;
    cl_image_desc desc;
};

