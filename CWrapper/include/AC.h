#pragma once

#include <stddef.h>

#ifdef _WIN32
#define AC_API __stdcall
#ifndef AC_DLL
#define AC_DLL __declspec(dllimport)
#else
#undef AC_DLL
#define AC_DLL __declspec(dllexport)
#endif
#else
#define AC_API
#ifndef AC_DLL
#define AC_DLL
#endif
#endif

#define acSaveImageYUV acSaveImageRGB
#define acSaveImageYUV444Bytes acSaveImageRGBBytes

#ifdef __cplusplus
extern "C"
{
#endif

    typedef enum ac_processType
    {
        AC_CPU,
        AC_GPU,
        AC_CPUCNN,
        AC_GPUCNN
    } ac_processType;

    typedef enum ac_bool
    {
        AC_FALSE = 0,
        AC_TRUE = 1
    } ac_bool;

    typedef enum ac_error
    {
        AC_OK = 0,
        AC_ERROR_NULL_INSTANCE,
        AC_ERROR_NULL_PARAMETERS,
        AC_ERROR_INIT_GPU,
        AC_ERROR_PORCESSOR_TYPE,
        AC_ERROR_LOAD_IMAGE,
        AC_ERROR_LOAD_VIDEO,
        AC_ERROR_INIT_VIDEO_WRITER,
        AC_ERROR_GPU_PROCESS,
        AC_ERROR_SAVE_TO_NULL_POINTER,
        AC_ERROR_NOT_YUV444,
        AC_ERROR_YUV444_AND_RGB32_AT_SAME_TIME
    } ac_error;

    typedef enum ac_codec
    {
        OTHER = -1,
        MP4V = 0,
        DXVA = 1,
        AVC1 = 2,
        VP09 = 3,
        HEVC = 4,
        AV01 = 5
    } ac_codec;

    typedef struct ac_parameters
    {
        int passes;
        int pushColorCount;
        double strengthColor;
        double strengthGradient;
        double zoomFactor;
        ac_bool fastMode;
        ac_bool videoMode;
        ac_bool preprocessing;
        ac_bool postprocessing;
        unsigned char preFilters;
        unsigned char postFilters;
        unsigned int maxThreads;
        ac_bool HDN;
        int HDNLevel;
        ac_bool alpha;
    } ac_parameters;

    typedef struct ac_version
    {
        char coreVersion[6];
        char wrapperVersion[6];
    } ac_version;

    typedef void* ac_instance;

    AC_DLL ac_version AC_API acGetVersion(void);
    AC_DLL ac_instance AC_API acGetInstance(ac_bool initGPU, ac_bool initGPUCNN, unsigned int platformID, unsigned int deviceID, ac_parameters* parameters, ac_processType type, ac_error* error);
    AC_DLL void AC_API acFreeInstance(ac_instance instance, ac_bool releaseGPU, ac_bool releaseGPUCNN);
    AC_DLL ac_error AC_API acInitParameters(ac_parameters* parameters);
    AC_DLL ac_error AC_API acLoadImage(ac_instance instance, const char* srcFile);
    AC_DLL ac_error AC_API acLoadVideo(ac_instance instance, const char* srcFile);
    AC_DLL ac_error AC_API acProcess(ac_instance instance);
    AC_DLL ac_error AC_API acProcessWithPrintProgress(ac_instance instance);
    /*
    Processing with callback funciton:
    callBack(double progress);
    progress form 0 to 1
    */
    AC_DLL ac_error AC_API acProcessWithProgress(ac_instance instance, void (*callBack)(double));
    /*
    Processing with callback funciton:
    callBack(double progress, double elapsedTime);
    progress form 0 to 1
    */
    AC_DLL ac_error AC_API acProcessWithProgressTime(ac_instance instance, void (*callBack)(double, double));
    AC_DLL ac_error AC_API acStopVideoProcess(ac_instance instance);
    AC_DLL ac_error AC_API acPauseVideoProcess(ac_instance instance);
    AC_DLL ac_error AC_API acContinueVideoProcess(ac_instance instance);
    AC_DLL ac_error AC_API acShowImage(ac_instance instance, ac_bool R2B);
    AC_DLL ac_error AC_API acSaveImage(ac_instance instance, const char* dstFile);
    AC_DLL ac_error AC_API acSetSaveVideoInfo(ac_instance instance, const char* dstFile, ac_codec codec, double fps);
    AC_DLL ac_error AC_API acSaveVideo(ac_instance instance);
    AC_DLL ac_error AC_API acSetArguments(ac_instance instance, ac_parameters* parameters);
    AC_DLL ac_error AC_API acSetVideoMode(ac_instance instance, ac_bool flag);
    AC_DLL ac_error AC_API acInitGPU(void);
    AC_DLL void AC_API acReleaseGPU(void);
    AC_DLL ac_error AC_API acInitGPUCNN(void);
    AC_DLL void AC_API acReleaseGPUCNN(void);
    AC_DLL ac_error AC_API acLoadImageRGB(ac_instance instance, int rows, int cols, unsigned char* r, unsigned char* g, unsigned char* b, ac_bool inputAsYUV444);
    AC_DLL ac_error AC_API acLoadImageYUV(ac_instance instance, int rowsY, int colsY, unsigned char* y, int rowsU, int colsU, unsigned char* u, int rowsV, int colsV, unsigned char* v);
    AC_DLL ac_error AC_API acLoadImageRGBBytes(ac_instance instance, int rows, int cols, unsigned char* data, size_t bytesPerLine, ac_bool inputAsYUV444, ac_bool inputAsRGB32);
    AC_DLL ac_error AC_API acSaveImageRGB(ac_instance instance, unsigned char** r, unsigned char** g, unsigned char** b);
    AC_DLL ac_error AC_API acSaveImageRGBBytes(ac_instance instance, unsigned char** data);
    AC_DLL size_t AC_API acGetResultDataLength(ac_instance instance, ac_error* error);
    AC_DLL size_t AC_API acGetResultDataPerChannelLength(ac_instance instance, ac_error* error);
    //shape should be int[3]
    AC_DLL ac_error AC_API acGetResultShape(ac_instance instance, int* shape);
    //acGetInfo may need to run two times for getting length of info string first
    AC_DLL ac_error AC_API acGetInfo(ac_instance instance, char* info, size_t* length);
    //acGetFiltersInfo may need to run two times for getting length of info string first
    AC_DLL ac_error AC_API acGetFiltersInfo(ac_instance instance, char* info, size_t* length);
    //acCheckGPUSupport may need to run two times for getting length of info string first
    AC_DLL ac_bool AC_API acCheckGPUSupport(unsigned int pID, unsigned int dID, char* info, size_t* length);
    //acCheckGPUSupport may need to run two times for getting length of info string and length(platforms) of devices first
    AC_DLL void AC_API acListGPUs(char* info, size_t* length, size_t* platforms, size_t* devices);
    AC_DLL ac_bool AC_API acIsInitializedGPU(void);
    AC_DLL ac_bool AC_API acIsInitializedGPUCNN(void);
    AC_DLL void AC_API acGetLastCoreErrorString(char *err, size_t* length);
#ifdef __cplusplus
}
#endif
