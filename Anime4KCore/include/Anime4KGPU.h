#pragma once
#include "Anime4K.h"

#include<fstream>

#include<CL/cl.h>

class DLL Anime4KGPU :
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
    void initOpenCL();
    void releaseOpenCL();
    std::string readKernel(const std::string &fileName);
private:
    cl_context context;
    cl_command_queue commandQueue;
    cl_program program;
    cl_device_id device;

    cl_image_format format;
    cl_image_desc desc;
};

