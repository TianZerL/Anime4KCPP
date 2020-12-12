#define AC_DLL
#define C_WRAPPER_VERSION "1.5.0"

#include "Anime4KCPP.hpp"
#include "AC.h"

std::string lastCoreError("No error");

std::unique_ptr<Anime4KCPP::ACCreator> acCreator;

Anime4KCPP::Parameters getParameters(ac_parameters* c_parameters)
{
    if (c_parameters == nullptr)
        return std::move(Anime4KCPP::Parameters());

    Anime4KCPP::Parameters cpp_parameters(
        c_parameters->passes,
        c_parameters->pushColorCount,
        c_parameters->strengthColor,
        c_parameters->strengthGradient,
        c_parameters->zoomFactor,
        c_parameters->fastMode,
        c_parameters->videoMode,
        c_parameters->preprocessing,
        c_parameters->postprocessing,
        c_parameters->preFilters,
        c_parameters->postFilters,
        c_parameters->maxThreads,
        c_parameters->HDN,
        c_parameters->HDNLevel,
        c_parameters->alpha);

    return std::move(cpp_parameters);
}

Anime4KCPP::Processor::Type getProcessorType(ac_processType type, ac_error* error)
{
    switch (type)
    {
    case AC_CPU_Anime4K09:
        return Anime4KCPP::Processor::Type::CPU_Anime4K09;
    case AC_CPU_ACNet:
        return Anime4KCPP::Processor::Type::CPU_ACNet;
    case AC_OpenCL_Anime4K09:
        return Anime4KCPP::Processor::Type::OpenCL_Anime4K09;
    case AC_OpenCL_ACNet:
        return Anime4KCPP::Processor::Type::OpenCL_ACNet;
#ifdef ENABLE_CUDA
    case AC_Cuda_Anime4K09:
        return Anime4KCPP::Processor::Type::Cuda_Anime4K09;
    case AC_Cuda_ACNet:
        return Anime4KCPP::Processor::Type::Cuda_ACNet;
#endif
    default:
        if (error != nullptr)
            *error = AC_ERROR_PORCESSOR_TYPE;
        return Anime4KCPP::Processor::Type::CPU_Anime4K09;
    }
}

extern "C"
{
    ac_version acGetVersion(void)
    {
        return ac_version{ ANIME4KCPP_CORE_VERSION,C_WRAPPER_VERSION };
    }

    ac_instance acGetInstance(ac_bool initGPU, ac_bool initGPUCNN, unsigned int platformID, unsigned int deviceID, ac_parameters* parameters, ac_processType type, ac_error* error)
    {
        if (error != nullptr)
            *error = AC_OK;

        if (initGPU == AC_TRUE && !Anime4KCPP::OpenCL::Anime4K09::isInitializedGPU())
        {
            try
            {
                Anime4KCPP::OpenCL::Anime4K09::initGPU(platformID, deviceID);
            }
            catch (const std::exception& err)
            {
                if (error != nullptr)
                    *error = AC_ERROR_INIT_GPU;
                lastCoreError = err.what();
                return nullptr;
            }
        }

        if (initGPUCNN == AC_TRUE && !Anime4KCPP::OpenCL::Anime4K09::isInitializedGPU())
        {
            try
            {
                Anime4KCPP::OpenCL::ACNet::initGPU(platformID, deviceID);
            }
            catch (const std::exception& err)
            {
                if (error != nullptr)
                    *error = AC_ERROR_INIT_GPU;
                lastCoreError = err.what();
                return nullptr;
            }
        }

        switch (type)
        {
        case AC_CPU_Anime4K09:
            return reinterpret_cast<ac_instance>(new Anime4KCPP::CPU::Anime4K09(getParameters(parameters)));
            break;
        case AC_CPU_ACNet:
            return reinterpret_cast<ac_instance>(new Anime4KCPP::CPU::ACNet(getParameters(parameters)));
            break;
        case AC_OpenCL_Anime4K09:
            return reinterpret_cast<ac_instance>(new Anime4KCPP::OpenCL::Anime4K09(getParameters(parameters)));
            break;
        case AC_OpenCL_ACNet:
            return reinterpret_cast<ac_instance>(new Anime4KCPP::OpenCL::ACNet(getParameters(parameters)));
            break;
        default:
            if (error != nullptr)
                *error = AC_ERROR_PORCESSOR_TYPE;
            return nullptr;
        }
    }

    ac_instance acGetInstance2(unsigned int managers, ac_managerData* managerData, ac_parameters* parameters, ac_processType type, ac_error* error)
    {
        if (error != nullptr)
            *error = AC_OK;

        acCreator = std::make_unique<Anime4KCPP::ACCreator>();

        if (managers & AC_Manager_OpenCL_Anime4K09)
        {
            if (managerData == nullptr || managerData->OpenCLAnime4K09Data == nullptr)
            {
                if (error != nullptr)
                    *error = AC_ERROR_NULL_Data;
                return nullptr;
            }
            acCreator->pushManager<Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::Anime4K09>>
                (managerData->OpenCLAnime4K09Data->pID, 
                    managerData->OpenCLAnime4K09Data->dID, 
                    managerData->OpenCLAnime4K09Data->OpenCLQueueNum, 
                    static_cast<bool>(managerData->OpenCLAnime4K09Data->OpenCLParallelIO));
        }
        if (managers & AC_Manager_OpenCL_ACNet)
        {
            if (managerData == nullptr || managerData->OpenCLACNetData == nullptr)
            {
                if (error != nullptr)
                    *error = AC_ERROR_NULL_Data;
                return nullptr;
            }
            acCreator->pushManager<Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::ACNet>>
                (managerData->OpenCLACNetData->pID, 
                    managerData->OpenCLACNetData->dID,
                    static_cast<Anime4KCPP::CNNType>(managerData->OpenCLACNetData->CNNType),
                    managerData->OpenCLACNetData->OpenCLQueueNum,
                    static_cast<bool>(managerData->OpenCLACNetData->OpenCLParallelIO));
        }
        if (managers & AC_Manager_Cuda)
        {
#ifndef ENABLE_CUDA
            if (error != nullptr)
                *error = AC_ERROR_CUDA_NOT_SUPPORTED;
            return nullptr;
#else
            if (managerData == nullptr || managerData->CUDAData == nullptr)
            {
                if (error != nullptr)
                    *error = AC_ERROR_NULL_Data;
                return nullptr;
            }
            acCreator->pushManager<Anime4KCPP::Cuda::Manager>(managerData->CUDAData->dID);
#endif // !CUDA_ENABLE
        }

        try
        {
            acCreator->init();
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            if (error != nullptr)
                *error = AC_ERROR_INIT_GPU;
            return nullptr;
        }

        return reinterpret_cast<ac_instance>(acCreator->create(getParameters(parameters), getProcessorType(type, error)));
    }

    void acFreeInstance(ac_instance instance, ac_bool releaseGPU, ac_bool releaseGPUCNN)
    {
        if (instance != nullptr)
            delete reinterpret_cast<Anime4KCPP::AC*>(instance);

        if (releaseGPU == AC_TRUE && Anime4KCPP::OpenCL::Anime4K09::isInitializedGPU())
            Anime4KCPP::OpenCL::Anime4K09::releaseGPU();

        if (releaseGPUCNN == AC_TRUE && Anime4KCPP::OpenCL::ACNet::isInitializedGPU())
            Anime4KCPP::OpenCL::ACNet::releaseGPU();

        if (acCreator != nullptr)
            acCreator->deinit(true);
    }

    void acFreeInstance2(ac_instance instance)
    {
        acCreator->release(reinterpret_cast<Anime4KCPP::AC*>(instance));

        if (acCreator != nullptr)
            acCreator->deinit(true);
    }

    ac_error acInitParameters(ac_parameters* parameters)
    {
        if (parameters == nullptr)
            return AC_ERROR_NULL_PARAMETERS;

        parameters->passes = 2;
        parameters->pushColorCount = 2;
        parameters->strengthColor = 0.3F;
        parameters->strengthGradient = 1.0F;
        parameters->zoomFactor = 2.0F;
        parameters->fastMode = AC_FALSE;
        parameters->videoMode = AC_FALSE;
        parameters->preprocessing = AC_FALSE;
        parameters->postprocessing = AC_FALSE;
        parameters->preFilters = 4;
        parameters->postFilters = 40;
        parameters->maxThreads = 4;
        parameters->HDN = AC_FALSE;

        return AC_OK;
    }

    ac_error acLoadImage(ac_instance instance, const char* srcFile)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        try
        {
            reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(srcFile);
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_LOAD_IMAGE;
        }

        return AC_OK;
    }

    ac_error acLoadVideo(ac_instance instance, const char* srcFile)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        try
        {
            reinterpret_cast<Anime4KCPP::AC*>(instance)->loadVideo(srcFile);
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_LOAD_VIDEO;
        }

        return AC_OK;
    }

    ac_error acProcess(ac_instance instance)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        try
        {
            reinterpret_cast<Anime4KCPP::AC*>(instance)->process();
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_GPU_PROCESS;
        }

        return AC_OK;
    }

    ac_error acProcessWithPrintProgress(ac_instance instance)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        try
        {
            reinterpret_cast<Anime4KCPP::AC*>(instance)->processWithPrintProgress();
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_GPU_PROCESS;
        }

        return AC_OK;
    }

    ac_error acProcessWithProgress(ac_instance instance, void (*callBack)(double))
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        try
        {
            reinterpret_cast<Anime4KCPP::AC*>(instance)->processWithProgress(callBack);
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_GPU_PROCESS;
        }

        return AC_OK;
    }

    ac_error acProcessWithProgressTime(ac_instance instance, void (*callBack)(double, double))
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        try
        {
            time_t start = time(nullptr);
            reinterpret_cast<Anime4KCPP::AC*>(instance)->processWithProgress([&callBack, &start](double v)
                {
                    callBack(v, time(nullptr) - start);
                });
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_GPU_PROCESS;
        }

        return AC_OK;
    }

    ac_error acStopVideoProcess(ac_instance instance)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->stopVideoProcess();

        return AC_OK;
    }

    ac_error acPauseVideoProcess(ac_instance instance)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->pauseVideoProcess();

        return AC_OK;
    }

    ac_error acContinueVideoProcess(ac_instance instance)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->continueVideoProcess();

        return AC_OK;
    }

    ac_error acShowImage(ac_instance instance, ac_bool R2B)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->showImage(R2B);

        return AC_OK;
    }

    ac_error acSaveImage(ac_instance instance, const char* dstFile)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;
        try
        {
            reinterpret_cast<Anime4KCPP::AC*>(instance)->saveImage(dstFile);
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_NOT_YUV444;
        }

        return AC_OK;
    }

    ac_error acSetSaveVideoInfo(ac_instance instance, const char* dstFile, ac_codec codec, double fps)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        try
        {
            reinterpret_cast<Anime4KCPP::AC*>(instance)->setVideoSaveInfo(dstFile, Anime4KCPP::CODEC(codec), fps);
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_INIT_VIDEO_WRITER;
        }

        return AC_OK;
    }

    ac_error acSaveVideo(ac_instance instance)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->saveVideo();

        return AC_OK;
    }

    ac_error acSetArguments(ac_instance instance, ac_parameters* parameters)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->setArguments(getParameters(parameters));

        return AC_OK;
    }

    ac_error acSetVideoMode(ac_instance instance, ac_bool flag)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->setVideoMode(flag);

        return AC_OK;
    }

    ac_error acInitGPU(void)
    {
        try
        {
            if (!Anime4KCPP::OpenCL::Anime4K09::isInitializedGPU())
                Anime4KCPP::OpenCL::Anime4K09::initGPU();
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_INIT_GPU;
        }

        return AC_OK;
    }

    void acReleaseGPU(void)
    {
        if (Anime4KCPP::OpenCL::Anime4K09::isInitializedGPU())
            Anime4KCPP::OpenCL::Anime4K09::releaseGPU();
    }

    ac_error acInitGPUCNN(void)
    {
        try
        {
            if (!Anime4KCPP::OpenCL::ACNet::isInitializedGPU())
                Anime4KCPP::OpenCL::ACNet::initGPU();
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_INIT_GPU;
        }

        return AC_OK;
    }

    void acReleaseGPUCNN(void)
    {
        if (Anime4KCPP::OpenCL::ACNet::isInitializedGPU())
            Anime4KCPP::OpenCL::ACNet::releaseGPU();
    }

    ac_error acInitGPU2(unsigned int managers, ac_managerData* managerData)
    {
        acCreator = std::make_unique<Anime4KCPP::ACCreator>();

        if (managers & AC_Manager_OpenCL_Anime4K09)
        {
            if (managerData == nullptr || managerData->OpenCLAnime4K09Data == nullptr)
                return AC_ERROR_NULL_Data;
            acCreator->pushManager<Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::Anime4K09>>
                (managerData->OpenCLAnime4K09Data->pID,
                    managerData->OpenCLAnime4K09Data->dID,
                    managerData->OpenCLAnime4K09Data->OpenCLQueueNum,
                    static_cast<bool>(managerData->OpenCLAnime4K09Data->OpenCLParallelIO));
        }
        if (managers & AC_Manager_OpenCL_ACNet)
        {
            if (managerData == nullptr || managerData->OpenCLACNetData == nullptr)
                return AC_ERROR_NULL_Data;
            acCreator->pushManager<Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::ACNet>>
                (managerData->OpenCLACNetData->pID,
                    managerData->OpenCLACNetData->dID,
                    static_cast<Anime4KCPP::CNNType>(managerData->OpenCLACNetData->CNNType),
                    managerData->OpenCLACNetData->OpenCLQueueNum,
                    static_cast<bool>(managerData->OpenCLACNetData->OpenCLParallelIO));
        }
        if (managers & AC_Manager_Cuda)
        {
#ifndef ENABLE_CUDA
            return AC_ERROR_CUDA_NOT_SUPPORTED;
#else
            if (managerData == nullptr || managerData->CUDAData == nullptr)
                return AC_ERROR_NULL_Data;
            acCreator->pushManager<Anime4KCPP::Cuda::Manager>(managerData->CUDAData->dID);
#endif // !CUDA_ENABLE
        }

        try
        {
            acCreator->init();
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_INIT_GPU;
        }

        return AC_OK;
    }

    void acReleaseGPU2(void)
    {
        if (acCreator != nullptr)
            acCreator->deinit(true);
    }

    ac_error acLoadImageRGBPlanarB(ac_instance instance, int rows, int cols, unsigned char* r, unsigned char* g, unsigned char* b, ac_bool inputAsYUV444)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(rows, cols, r, g, b, inputAsYUV444);

        return AC_OK;
    }

    ac_error acLoadImageYUVPlanarB(ac_instance instance, int rowsY, int colsY, unsigned char* y, int rowsU, int colsU, unsigned char* u, int rowsV, int colsV, unsigned char* v)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(rowsY, colsY, y, rowsU, colsU, u, rowsV, colsV, v);

        return AC_OK;
    }

    ac_error acLoadImageRGBPackedB(ac_instance instance, int rows, int cols, unsigned char* data, size_t bytesPerLine, ac_bool inputAsYUV444, ac_bool inputAsRGB32)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        if (inputAsRGB32 && inputAsYUV444)
            return AC_ERROR_YUV444_AND_RGB32_AT_SAME_TIME;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(rows, cols, data, bytesPerLine, inputAsYUV444, inputAsRGB32);

        return AC_OK;
    }

    ac_error acLoadImageGrayscaleB(ac_instance instance, int rows, int cols, unsigned char* data, size_t bytesPerLine)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(rows, cols, data, bytesPerLine, false, false, true);

        return AC_OK;
    }

    ac_error acSaveImageRGBPlanarB(ac_instance instance, unsigned char* r, unsigned char* g, unsigned char* b)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        if (r == nullptr || g == nullptr || b == nullptr)
            return AC_ERROR_SAVE_TO_NULL_POINTER;
        try
        {
            reinterpret_cast<Anime4KCPP::AC*>(instance)->saveImage(r, g, b);
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_NOT_YUV444;
        }

        return AC_OK;
    }

    ac_error acSaveImageRGBPackedB(ac_instance instance, unsigned char* data)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        if (data == nullptr)
            return AC_ERROR_SAVE_TO_NULL_POINTER;

        try
        {
            reinterpret_cast<Anime4KCPP::AC*>(instance)->saveImage(data);
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_NOT_YUV444;
        }

        return AC_OK;
    }

    ac_error acLoadImageRGBPlanarW(ac_instance instance, int rows, int cols, unsigned short int* r, unsigned short int* g, unsigned short int* b, ac_bool inputAsYUV444)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(rows, cols, r, g, b, inputAsYUV444);

        return AC_OK;
    }

    ac_error acLoadImageYUVPlanarW(ac_instance instance, int rowsY, int colsY, unsigned short int* y, int rowsU, int colsU, unsigned short int* u, int rowsV, int colsV, unsigned short int* v)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(rowsY, colsY, y, rowsU, colsU, u, rowsV, colsV, v);

        return AC_OK;
    }

    ac_error acLoadImageRGBPackedW(ac_instance instance, int rows, int cols, unsigned short int* data, size_t bytesPerLine, ac_bool inputAsYUV444, ac_bool inputAsRGB32)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        if (inputAsRGB32 && inputAsYUV444)
            return AC_ERROR_YUV444_AND_RGB32_AT_SAME_TIME;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(rows, cols, data, bytesPerLine, inputAsYUV444, inputAsRGB32);

        return AC_OK;
    }

    ac_error acLoadImageGrayscaleW(ac_instance instance, int rows, int cols, unsigned short int* data, size_t bytesPerLine)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(rows, cols, data, bytesPerLine, false, false, true);

        return AC_OK;
    }

    ac_error acSaveImageRGBPlanarW(ac_instance instance, unsigned short int* r, unsigned short int* g, unsigned short int* b)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        if (r == nullptr || g == nullptr || b == nullptr)
            return AC_ERROR_SAVE_TO_NULL_POINTER;
        try
        {
            reinterpret_cast<Anime4KCPP::AC*>(instance)->saveImage(r, g, b);
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_NOT_YUV444;
        }

        return AC_OK;
    }

    ac_error acSaveImageRGBPackedW(ac_instance instance, unsigned short int* data)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        if (data == nullptr)
            return AC_ERROR_SAVE_TO_NULL_POINTER;

        try
        {
            reinterpret_cast<Anime4KCPP::AC*>(instance)->saveImage(data);
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_NOT_YUV444;
        }

        return AC_OK;
    }

    ac_error acLoadImageRGBPlanarF(ac_instance instance, int rows, int cols, float* r, float* g, float* b, ac_bool inputAsYUV444)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(rows, cols, r, g, b, inputAsYUV444);

        return AC_OK;
    }

    ac_error acLoadImageYUVPlanarF(ac_instance instance, int rowsY, int colsY, float* y, int rowsU, int colsU, float* u, int rowsV, int colsV, float* v)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(rowsY, colsY, y, rowsU, colsU, u, rowsV, colsV, v);

        return AC_OK;
    }

    ac_error acLoadImageRGBPackedF(ac_instance instance, int rows, int cols, float* data, size_t bytesPerLine, ac_bool inputAsYUV444, ac_bool inputAsRGB32)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        if (inputAsRGB32 && inputAsYUV444)
            return AC_ERROR_YUV444_AND_RGB32_AT_SAME_TIME;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(rows, cols, data, bytesPerLine, inputAsYUV444, inputAsRGB32);

        return AC_OK;
    }

    ac_error acLoadImageGrayscaleF(ac_instance instance, int rows, int cols, float* data, size_t bytesPerLine)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(rows, cols, data, bytesPerLine, false, false, true);

        return AC_OK;
    }

    ac_error acSaveImageRGBPlanarF(ac_instance instance, float* r, float* g, float* b)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        if (r == nullptr || g == nullptr || b == nullptr)
            return AC_ERROR_SAVE_TO_NULL_POINTER;
        try
        {
            reinterpret_cast<Anime4KCPP::AC*>(instance)->saveImage(r, g, b);
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_NOT_YUV444;
        }

        return AC_OK;
    }

    ac_error acSaveImageRGBPackedF(ac_instance instance, float* data)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        if (data == nullptr)
            return AC_ERROR_SAVE_TO_NULL_POINTER;

        try
        {
            reinterpret_cast<Anime4KCPP::AC*>(instance)->saveImage(data);
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_NOT_YUV444;
        }

        return AC_OK;
    }

    size_t acGetResultDataLength(ac_instance instance, ac_error* error)
    {
        if (error != nullptr)
            *error = AC_OK;

        if (instance == nullptr)
        {
            if (error != nullptr)
                *error = AC_ERROR_NULL_INSTANCE;
            return 0;
        }

        return reinterpret_cast<Anime4KCPP::AC*>(instance)->getResultDataLength();
    }

    size_t acGetResultDataPerChannelLength(ac_instance instance, ac_error* error)
    {
        if (error != nullptr)
            *error = AC_OK;

        if (instance == nullptr)
        {
            if (error != nullptr)
                *error = AC_ERROR_NULL_INSTANCE;
            return 0;
        }

        return reinterpret_cast<Anime4KCPP::AC*>(instance)->getResultDataPerChannelLength();
    }

    ac_error acGetResultShape(ac_instance instance, int shape[3])
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        std::array<int, 3> ret = reinterpret_cast<Anime4KCPP::AC*>(instance)->getResultShape();

        for (int i = 0; i < 3; i++)
            shape[i] = ret[i];

        return AC_OK;
    }

    ac_error acGetInfo(ac_instance instance, char* info, size_t* length)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        if (info == nullptr && length == nullptr)
            return AC_OK;

        std::string ret = reinterpret_cast<Anime4KCPP::AC*>(instance)->getInfo();

        if (length != nullptr)
            *length = ret.size() + 1;

        if (info != nullptr)
            memcpy(info, ret.c_str(), ret.size() + 1);

        return AC_OK;
    }

    ac_error acGetFiltersInfo(ac_instance instance, char* info, size_t* length)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        if (info == nullptr && length == nullptr)
            return AC_OK;

        std::string ret = reinterpret_cast<Anime4KCPP::AC*>(instance)->getFiltersInfo();

        if (length != nullptr)
            *length = ret.size() + 1;

        if (info != nullptr)
            memcpy(info, ret.c_str(), ret.size() + 1);

        return AC_OK;
    }

    ac_bool acCheckGPUSupport(unsigned int pID, unsigned int dID, char* info, size_t* length)
    {
        Anime4KCPP::OpenCL::GPUInfo ret = Anime4KCPP::OpenCL::checkGPUSupport(pID, dID);
        ac_bool rst = ac_bool(ret.supported);

        if (length != nullptr)
            *length = ret().size() + 1;

        if (info != nullptr)
            memcpy(info, ret().c_str(), ret().size() + 1);

        return rst;
    }

    ac_bool acCheckGPUSupport2(ac_GPGPU GPGPUModel, unsigned int pID, unsigned int dID, char* info, size_t* length)
    {
        std::string infoString;
        ac_bool rst = AC_FALSE;

        switch (GPGPUModel)
        {
        case AC_CUDA:
#ifdef ENABLE_CUDA
        {
            Anime4KCPP::Cuda::GPUInfo ret = Anime4KCPP::Cuda::checkGPUSupport(dID);
            rst = (ac_bool)ret.supported;
            infoString = ret();
        }
#else
        {
            rst = (ac_bool)false;
            infoString = "CUDA is not supported";
        }
#endif
        break;
        case AC_OpenCL:
        {
            Anime4KCPP::OpenCL::GPUInfo ret = Anime4KCPP::OpenCL::checkGPUSupport(pID, dID);
            rst = (ac_bool)ret.supported;
            infoString = ret();
        }
        break;
        }

        if (length != nullptr)
            *length = infoString.size() + 1;

        if (info != nullptr)
            memcpy(info, infoString.c_str(), infoString.size() + 1);

        return rst;
    }

    void acListGPUs(char* info, size_t* length, size_t* platforms, size_t* devices)
    {
        Anime4KCPP::OpenCL::GPUList ret = Anime4KCPP::OpenCL::listGPUs();

        if (length != nullptr)
            *length = ret().size() + 1;

        if (info != nullptr)
            memcpy(info, ret().c_str(), ret().size() + 1);

        if (platforms != nullptr)
            *platforms = ret.platforms;

        if (devices != nullptr)
            for (int i : ret.devices)
                *(devices++) = i;
    }

    ac_bool acIsInitializedGPU(void)
    {
        return ac_bool(Anime4KCPP::OpenCL::Anime4K09::isInitializedGPU());
    }

    ac_bool acIsInitializedGPUCNN(void)
    {
        return ac_bool(Anime4KCPP::OpenCL::ACNet::isInitializedGPU());
    }

    void acGetLastCoreErrorString(char* err, size_t* length)
    {
        if (length != nullptr)
            *length = lastCoreError.size() + 1;

        if (err != nullptr)
            memcpy(err, lastCoreError.c_str(), lastCoreError.size() + 1);
    }

    void acBenchmark(unsigned int pID, unsigned int dID, double* CPUScore, double* GPUScore)
    {
        double _CPUScore = Anime4KCPP::benchmark<Anime4KCPP::CPU::ACNet>();
        double _OpenCLScore = Anime4KCPP::benchmark<Anime4KCPP::OpenCL::ACNet>(pID, dID);

        *CPUScore = _CPUScore;
        *GPUScore = _OpenCLScore;
    }

    double acBenchmark2(ac_processType processType, unsigned int pID, unsigned int dID)
    {
        switch (processType)
        {
        case AC_CPU_Anime4K09:
            return Anime4KCPP::benchmark<Anime4KCPP::CPU::Anime4K09>();
        case AC_CPU_ACNet:
            return Anime4KCPP::benchmark<Anime4KCPP::CPU::ACNet>();
        case AC_OpenCL_Anime4K09:
            return Anime4KCPP::benchmark<Anime4KCPP::OpenCL::Anime4K09>(pID, dID);
        case AC_OpenCL_ACNet:
            return Anime4KCPP::benchmark<Anime4KCPP::OpenCL::ACNet>(pID, dID, Anime4KCPP::CNNType::ACNetHDNL0);
#ifdef ENABLE_CUDA
        case AC_Cuda_Anime4K09:
            return Anime4KCPP::benchmark<Anime4KCPP::Cuda::Anime4K09>(dID);
        case AC_Cuda_ACNet:
            return Anime4KCPP::benchmark<Anime4KCPP::Cuda::ACNet>(dID);
#endif // ENABLE_CUDA
        default:
            return 0.0;
        }
    }

    ac_processType acGetProcessType(ac_instance instance, ac_error* error)
    {
        if (error != nullptr)
            *error = AC_OK;

        if (instance == nullptr)
        {
            if (error != nullptr)
                *error = AC_ERROR_NULL_INSTANCE;
            return AC_CPU_Anime4K09;
        }

        Anime4KCPP::Processor::Type type = reinterpret_cast<Anime4KCPP::AC*>(instance)->getProcessorType();
        switch (type)
        {
        case Anime4KCPP::Processor::Type::CPU_Anime4K09:
            return AC_CPU_Anime4K09;
        case Anime4KCPP::Processor::Type::CPU_ACNet:
            return AC_CPU_ACNet;
        case Anime4KCPP::Processor::Type::OpenCL_Anime4K09:
            return AC_OpenCL_Anime4K09;
        case Anime4KCPP::Processor::Type::OpenCL_ACNet:
            return AC_OpenCL_ACNet;
#ifdef ENABLE_CUDA
        case Anime4KCPP::Processor::Type::Cuda_Anime4K09:
            return AC_Cuda_Anime4K09;
        case Anime4KCPP::Processor::Type::Cuda_ACNet:
            return AC_Cuda_ACNet;
#endif
        default:
            return AC_CPU_Anime4K09;
        }
    }

    ac_error acGetProcessorInfo(ac_instance instance, char* info, size_t* length)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        if (info == nullptr && length == nullptr)
            return AC_OK;

        std::string ret = reinterpret_cast<Anime4KCPP::AC*>(instance)->getProcessorInfo();

        if (length != nullptr)
            *length = ret.size() + 1;

        if (info != nullptr)
            memcpy(info, ret.c_str(), ret.size() + 1);

        return AC_OK;
    }
}
