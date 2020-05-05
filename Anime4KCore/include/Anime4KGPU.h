#pragma once
#include "Anime4K.h"

#include<fstream>

#ifdef __APPLE__
#include<OpenCL/opencl.h>
#else
#include<CL/cl.h>
#endif // SPECIAL OS


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
        uint8_t preFilters = 4,
        uint8_t postFilters = 40,
        unsigned int maxThreads = std::thread::hardware_concurrency(),
        unsigned int platformID = 0,
        unsigned int deviceID = 0,
        bool initGPU = true
    );
    Anime4KGPU(bool initGPU, unsigned int platformID = 0, unsigned int deviceID = 0);
    virtual ~Anime4KGPU();
    virtual void process();
    void initGPU();
    void releaseGPU();
    bool isInitializedGPU();
    static std::pair<std::pair<int, std::vector<int>>, std::string> listGPUs();
    static std::pair<bool, std::string> checkGPUSupport(unsigned int pID, unsigned int dID);
protected:
    void runKernel(cv::InputArray orgImg, cv::OutputArray dstImg);
    void initOpenCL();
    void releaseOpenCL();
    std::string readKernel(const std::string &fileName);
private:
    bool isInitialized;

    cl_context context;
    cl_command_queue commandQueue;
    cl_program program;
    cl_device_id device;

    uint64_t frameGPUDoneCount;

    cl_image_format format;
    cl_image_desc dstDesc;
    cl_image_desc orgDesc;

    double nWidth;
    double nHeight;

    const unsigned int pID;
    const unsigned int dID;
};

