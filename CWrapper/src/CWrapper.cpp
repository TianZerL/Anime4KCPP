#define C_WRAPPER_VERSION "1.6.0"

#include "Anime4KCPP.hpp"
#include "AC.h"

static std::string lastCoreError("No error");

static Anime4KCPP::ACInitializer initializer;

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
#ifdef ENABLE_OPENCL
    case AC_OpenCL_Anime4K09:
        return Anime4KCPP::Processor::Type::OpenCL_Anime4K09;
    case AC_OpenCL_ACNet:
        return Anime4KCPP::Processor::Type::OpenCL_ACNet;
#endif
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

#ifdef ENABLE_OPENCL
        if (initGPU == AC_TRUE && !Anime4KCPP::OpenCL::Anime4K09::isInitialized())
        {

            if (error != nullptr)
                *error = AC_ERROR_OPENCL_NOT_SUPPORTED;
            return nullptr;
            try
            {
                Anime4KCPP::OpenCL::Anime4K09::init(platformID, deviceID);
            }
            catch (const std::exception& err)
            {
                if (error != nullptr)
                    *error = AC_ERROR_INIT_GPU;
                lastCoreError = err.what();
                return nullptr;
            }
        }
#endif

#ifdef ENABLE_OPENCL
        if (initGPUCNN == AC_TRUE && !Anime4KCPP::OpenCL::Anime4K09::isInitialized())
        {
            try
            {
                Anime4KCPP::OpenCL::ACNet::init(platformID, deviceID);
            }
            catch (const std::exception& err)
            {
                if (error != nullptr)
                    *error = AC_ERROR_INIT_GPU;
                lastCoreError = err.what();
                return nullptr;
            }
        }
#endif

        switch (type)
        {
        case AC_CPU_Anime4K09:
            return reinterpret_cast<ac_instance>(new Anime4KCPP::CPU::Anime4K09(getParameters(parameters)));
            break;
        case AC_CPU_ACNet:
            return reinterpret_cast<ac_instance>(new Anime4KCPP::CPU::ACNet(getParameters(parameters)));
            break;
#ifdef ENABLE_OPENCL
        case AC_OpenCL_Anime4K09:
            return reinterpret_cast<ac_instance>(new Anime4KCPP::OpenCL::Anime4K09(getParameters(parameters)));
            break;
        case AC_OpenCL_ACNet:
            return reinterpret_cast<ac_instance>(new Anime4KCPP::OpenCL::ACNet(getParameters(parameters)));
            break;
#endif
        default:
            if (error != nullptr)
                *error = AC_ERROR_PORCESSOR_TYPE;
            return nullptr;
        }
    }

    ac_error acInitProcessor(ac_manager_t managers, ac_managerData* managerData)
    {
        initializer.release(true);
        if (managers & AC_Manager_OpenCL_Anime4K09)
        {
#ifndef ENABLE_OPENCL
            return AC_ERROR_OPENCL_NOT_SUPPORTED;
#else
            if (managerData == nullptr || managerData->OpenCLAnime4K09Data == nullptr)
            {
                return AC_ERROR_NULL_DATA;
            }
            initializer.pushManager<Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::Anime4K09>>
                (managerData->OpenCLAnime4K09Data->pID,
                    managerData->OpenCLAnime4K09Data->dID,
                    managerData->OpenCLAnime4K09Data->OpenCLQueueNum,
                    static_cast<bool>(managerData->OpenCLAnime4K09Data->OpenCLParallelIO));
#endif // !ENABLE_OPENCL
        }
        if (managers & AC_Manager_OpenCL_ACNet)
        {
#ifndef ENABLE_OPENCL
            return AC_ERROR_OPENCL_NOT_SUPPORTED;
#else
            if (managerData == nullptr || managerData->OpenCLACNetData == nullptr)
            {
                return AC_ERROR_NULL_DATA;
            }
            initializer.pushManager<Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::ACNet>>
                (managerData->OpenCLACNetData->pID,
                    managerData->OpenCLACNetData->dID,
                    static_cast<Anime4KCPP::CNNType::Value>(managerData->OpenCLACNetData->CNNType),
                    managerData->OpenCLACNetData->OpenCLQueueNum,
                    static_cast<bool>(managerData->OpenCLACNetData->OpenCLParallelIO));
#endif // !ENABLE_OPENCL
        }
        if (managers & AC_Manager_Cuda)
        {
#ifndef ENABLE_CUDA
            return AC_ERROR_CUDA_NOT_SUPPORTED;
#else
            if (managerData == nullptr || managerData->CUDAData == nullptr)
            {
                if (error != nullptr)
                    *error = AC_ERROR_NULL_DATA;
                return nullptr;
            }
            initializer.pushManager<Anime4KCPP::Cuda::Manager>(managerData->CUDAData->dID);
#endif // !CUDA_ENABLE
        }

        try
        {
            initializer.init();
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_INIT_PROCESSOR;
        }

        return AC_OK;
    }

    void acReleaseAllProcessors(void)
    {
        initializer.release(true);
    }

    ac_instance acGetInstance2(ac_manager_t managers, ac_managerData* managerData, ac_parameters* parameters, ac_processType type, ac_error* error)
    {
        ac_error err = acInitProcessor(managers, managerData);

        if (error != nullptr)
            *error = err;

        if (err != AC_OK)
            return nullptr;

        return acGetInstance3(parameters, type, error);
    }

    ac_instance acGetInstance3(ac_parameters* parameters, ac_processType type, ac_error* error)
    {
        return reinterpret_cast<ac_instance>(Anime4KCPP::ACCreator::create(getParameters(parameters), getProcessorType(type, error)));
    }

    void acFreeInstance(ac_instance instance, ac_bool releaseGPU, ac_bool releaseGPUCNN)
    {
        if (instance != nullptr)
            delete reinterpret_cast<Anime4KCPP::AC*>(instance);

#ifdef ENABLE_OPENCL
        if (releaseGPU == AC_TRUE && Anime4KCPP::OpenCL::Anime4K09::isInitialized())
            Anime4KCPP::OpenCL::Anime4K09::release();

        if (releaseGPUCNN == AC_TRUE && Anime4KCPP::OpenCL::ACNet::isInitialized())
            Anime4KCPP::OpenCL::ACNet::release();
#endif

        initializer.release(true);
    }

    void acFreeInstance2(ac_instance instance)
    {
        initializer.release(reinterpret_cast<Anime4KCPP::AC*>(instance));

        initializer.release(true);
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
#if ENABLE_IMAGE_IO
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
#else
        return AC_ERROR_IMAGE_IO_DISABLE;
#endif // ENABLE_IMAGE_IO
    }

    ac_error acLoadImageFromBuffer(ac_instance instance, const unsigned char* buf, size_t size)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        try
        {
            reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(buf, size);
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_LOAD_IMAGE;
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


    ac_error acShowImage(ac_instance instance, ac_bool R2B)
    {
#if ENABLE_PREVIEW_GUI
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->showImage(R2B);

        return AC_OK;
#else
        return AC_ERROR_PREVIEW_GUI_DISABLE;
#endif // ENABLE_PREVIEW_GUI
    }

    ac_error acSaveImage(ac_instance instance, const char* dstFile)
    {
#if ENABLE_IMAGE_IO
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
#else
        return AC_ERROR_IMAGE_IO_DISABLE;
#endif // ENABLE_IMAGE_IO
    }

    ac_error acSaveImageToBuffer(ac_instance instance, const char* suffix, unsigned char* buf, size_t size)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        try
        {
            std::vector<unsigned char> data;
            reinterpret_cast<Anime4KCPP::AC*>(instance)->saveImage(suffix, data);

            if (data.size() > size)
                return AC_ERROR_INSUFFICIENT_BUFFER_SIZE;

            std::copy(data.begin(), data.end(), buf);
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_FAILED_TO_ENCODE;
        }

        return AC_OK;
    }

    ac_error acSetParameters(ac_instance instance, ac_parameters* parameters)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->setParameters(getParameters(parameters));

        return AC_OK;
    }

    ac_error acInitGPU(void)
    {
#ifdef ENABLE_OPENCL
        try
        {
            if (!Anime4KCPP::OpenCL::Anime4K09::isInitialized())
                Anime4KCPP::OpenCL::Anime4K09::init();
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_INIT_GPU;
        }
#endif
        return AC_OK;
    }

    void acReleaseGPU(void)
    {
#ifdef ENABLE_OPENCL
        if (Anime4KCPP::OpenCL::Anime4K09::isInitialized())
            Anime4KCPP::OpenCL::Anime4K09::release();
#endif
    }

    ac_error acInitGPUCNN(void)
    {
#ifdef ENABLE_OPENCL
        try
        {
            if (!Anime4KCPP::OpenCL::ACNet::isInitialized())
                Anime4KCPP::OpenCL::ACNet::init();
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_INIT_GPU;
        }
#endif
        return AC_OK;
    }

    void acReleaseGPUCNN(void)
    {
#ifdef ENABLE_OPENCL
        if (Anime4KCPP::OpenCL::ACNet::isInitialized())
            Anime4KCPP::OpenCL::ACNet::release();
#endif
    }

    ac_error acInitGPU2(unsigned int managers, ac_managerData* managerData)
    {
        return acInitProcessor(managers, managerData);
    }

    void acReleaseGPU2(void)
    {
        acReleaseAllProcessors();
    }

    ac_error acLoadImageRGBPlanarB(ac_instance instance, int rows, int cols, size_t stride, unsigned char* r, unsigned char* g, unsigned char* b, ac_bool inputAsYUV444)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(rows, cols, stride, r, g, b, inputAsYUV444);

        return AC_OK;
    }

    ac_error acLoadImageYUVPlanarB(ac_instance instance,
        int rowsY, int colsY, size_t strideY, unsigned char* y,
        int rowsU, int colsU, size_t strideU, unsigned char* u,
        int rowsV, int colsV, size_t strideV, unsigned char* v)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(
            rowsY, colsY, strideY, y,
            rowsU, colsU, strideU, u,
            rowsV, colsV, strideV, v);

        return AC_OK;
    }

    ac_error acLoadImageRGBPackedB(ac_instance instance, int rows, int cols, size_t stride, unsigned char* data, ac_bool inputAsYUV444, ac_bool inputAsRGB32)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        if (inputAsRGB32 && inputAsYUV444)
            return AC_ERROR_YUV444_AND_RGB32_AT_SAME_TIME;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(rows, cols, stride, data, inputAsYUV444, inputAsRGB32);

        return AC_OK;
    }

    ac_error acLoadImageGrayscaleB(ac_instance instance, int rows, int cols, size_t stride, unsigned char* data)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(rows, cols, stride, data, false, false, true);

        return AC_OK;
    }

    ac_error acLoadImageRGBPlanarW(ac_instance instance, int rows, int cols, size_t stride, unsigned short* r, unsigned short* g, unsigned short* b, ac_bool inputAsYUV444)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(rows, cols, stride, r, g, b, inputAsYUV444);

        return AC_OK;
    }

    ac_error acLoadImageYUVPlanarW(ac_instance instance,
        int rowsY, int colsY, size_t strideY, unsigned short* y,
        int rowsU, int colsU, size_t strideU, unsigned short* u,
        int rowsV, int colsV, size_t strideV, unsigned short* v)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(
            rowsY, colsY, strideY, y,
            rowsU, colsU, strideU, u,
            rowsV, colsV, strideV, v);

        return AC_OK;
    }

    ac_error acLoadImageRGBPackedW(ac_instance instance, int rows, int cols, size_t stride, unsigned short* data, ac_bool inputAsYUV444, ac_bool inputAsRGB32)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        if (inputAsRGB32 && inputAsYUV444)
            return AC_ERROR_YUV444_AND_RGB32_AT_SAME_TIME;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(rows, cols, stride, data, inputAsYUV444, inputAsRGB32);

        return AC_OK;
    }

    ac_error acLoadImageGrayscaleW(ac_instance instance, int rows, int cols, size_t stride, unsigned short* data)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(rows, cols, stride, data, false, false, true);

        return AC_OK;
    }

    ac_error acLoadImageRGBPlanarF(ac_instance instance, int rows, int cols, size_t stride, float* r, float* g, float* b, ac_bool inputAsYUV444)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(rows, cols, stride, r, g, b, inputAsYUV444);

        return AC_OK;
    }

    ac_error acLoadImageYUVPlanarF(ac_instance instance,
        int rowsY, int colsY, size_t strideY, float* y,
        int rowsU, int colsU, size_t strideU, float* u,
        int rowsV, int colsV, size_t strideV, float* v)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(
            rowsY, colsY, strideY, y,
            rowsU, colsU, strideU, u,
            rowsV, colsV, strideV, v);

        return AC_OK;
    }

    ac_error acLoadImageRGBPackedF(ac_instance instance, int rows, int cols, size_t stride, float* data, ac_bool inputAsYUV444, ac_bool inputAsRGB32)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        if (inputAsRGB32 && inputAsYUV444)
            return AC_ERROR_YUV444_AND_RGB32_AT_SAME_TIME;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(rows, cols, stride, data, inputAsYUV444, inputAsRGB32);

        return AC_OK;
    }

    ac_error acLoadImageGrayscaleF(ac_instance instance, int rows, int cols, size_t stride, float* data)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::AC*>(instance)->loadImage(rows, cols, stride, data, false, false, true);

        return AC_OK;
    }

    ac_error acSaveImageRGBPlanar(ac_instance instance,
        unsigned char* r, size_t strideR,
        unsigned char* g, size_t strideG,
        unsigned char* b, size_t strideB)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        if (r == nullptr || g == nullptr || b == nullptr)
            return AC_ERROR_SAVE_TO_NULL_POINTER;
        try
        {
            reinterpret_cast<Anime4KCPP::AC*>(instance)->saveImage(r, strideR, g, strideG, b, strideB);
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_NOT_YUV444;
        }

        return AC_OK;
    }

    ac_error acSaveImageRGBPacked(ac_instance instance, unsigned char* data, size_t stride)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        if (data == nullptr)
            return AC_ERROR_SAVE_TO_NULL_POINTER;

        try
        {
            reinterpret_cast<Anime4KCPP::AC*>(instance)->saveImage(data, stride);
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_NOT_YUV444;
        }

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
#ifndef ENABLE_OPENCL
        return AC_FALSE;
#else
        Anime4KCPP::OpenCL::GPUInfo ret = Anime4KCPP::OpenCL::checkGPUSupport(pID, dID);
        ac_bool rst = ac_bool(ret.supported);

        if (length != nullptr)
            *length = ret().size() + 1;

        if (info != nullptr)
            memcpy(info, ret().c_str(), ret().size() + 1);

        return rst;
#endif
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
#ifdef ENABLE_OPENCL
        {
            Anime4KCPP::OpenCL::GPUInfo ret = Anime4KCPP::OpenCL::checkGPUSupport(pID, dID);
            rst = (ac_bool)ret.supported;
            infoString = ret();
        }
#else
        {
            rst = (ac_bool)false;
            infoString = "OpenCL is not supported";
        }
#endif
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
#ifdef ENABLE_OPENCL
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
#else
        std::string ret = "OpenCL is not supported";
        if (length != nullptr)
            *length = ret.size() + 1;

        if (info != nullptr)
            memcpy(info, ret.c_str(), ret.size() + 1);

        if (platforms != nullptr)
            *platforms = 0;
#endif
    }

    ac_bool acIsInitializedGPU(void)
    {
#ifdef ENABLE_OPENCL
        return ac_bool(Anime4KCPP::OpenCL::Anime4K09::isInitialized());
#else
        return AC_FALSE;
#endif
    }

    ac_bool acIsInitializedGPUCNN(void)
    {
#ifdef ENABLE_OPENCL
        return ac_bool(Anime4KCPP::OpenCL::ACNet::isInitialized());
#else
        return AC_FALSE;
#endif
    }

    void acGetLastCoreErrorString(char* err, size_t* length)
    {
        if (length != nullptr)
            *length = lastCoreError.size() + 1;

        if (err != nullptr)
            memcpy(err, lastCoreError.c_str(), lastCoreError.size() + 1);
    }

    void acBenchmark(const int  pID, const int  dID, double* CPUScore, double* GPUScore)
    {
        double _CPUScore = Anime4KCPP::benchmark<Anime4KCPP::CPU::ACNet, 1920, 1080>();
#ifdef ENABLE_OPENCL
        double _OpenCLScore = Anime4KCPP::benchmark<Anime4KCPP::OpenCL::ACNet, 1920, 1080>(pID, dID);
#else
        double _OpenCLScore = 0.0;
#endif

        *CPUScore = _CPUScore;
        *GPUScore = _OpenCLScore;
    }

    double acBenchmark2(ac_processType processType, const int pID, const int dID)
    {
        switch (processType)
        {
        case AC_CPU_Anime4K09:
            return Anime4KCPP::benchmark<Anime4KCPP::CPU::Anime4K09, 1920, 1080>();
        case AC_CPU_ACNet:
            return Anime4KCPP::benchmark<Anime4KCPP::CPU::ACNet, 1920, 1080>();
#ifdef ENABLE_OPENCL
        case AC_OpenCL_Anime4K09:
            return Anime4KCPP::benchmark<Anime4KCPP::OpenCL::Anime4K09, 1920, 1080>(pID, dID);
        case AC_OpenCL_ACNet:
            return Anime4KCPP::benchmark<Anime4KCPP::OpenCL::ACNet, 1920, 1080>(pID, dID, Anime4KCPP::CNNType::ACNetHDNL0);
#endif // ENABLE_OPENCL
#ifdef ENABLE_CUDA
        case AC_Cuda_Anime4K09:
            return Anime4KCPP::benchmark<Anime4KCPP::Cuda::Anime4K09, 1920, 1080>(dID);
        case AC_Cuda_ACNet:
            return Anime4KCPP::benchmark<Anime4KCPP::Cuda::ACNet, 1920, 1080>(dID);
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
#ifdef ENABLE_OPENCL
        case Anime4KCPP::Processor::Type::OpenCL_Anime4K09:
            return AC_OpenCL_Anime4K09;
        case Anime4KCPP::Processor::Type::OpenCL_ACNet:
            return AC_OpenCL_ACNet;
#endif
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

    ac_error acSaveImageBufferSize(ac_instance instance, size_t* dataSize, size_t dstStride)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        if (dataSize != nullptr)
            reinterpret_cast<Anime4KCPP::AC*>(instance)->saveImageBufferSize(*dataSize, dstStride);
        else
            return AC_ERROR_NULL_DATA;

        return AC_OK;
    }

    ac_error acSaveImageBufferSizeRGB(ac_instance instance,
        size_t* rSize, size_t dstStrideR,
        size_t* gSize, size_t dstStrideG,
        size_t* bSize, size_t dstStrideB)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        if (rSize != nullptr && gSize != nullptr && bSize != nullptr)
            reinterpret_cast<Anime4KCPP::AC*>(instance)->saveImageBufferSize(
                *rSize, dstStrideR,
                *gSize, dstStrideG,
                *bSize, dstStrideB);
        else
            return AC_ERROR_NULL_DATA;

        return AC_OK;
    }

    ac_error saveImageShape(ac_instance instance, int* cols, int* rows, int* channels)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        if (cols != nullptr && rows != nullptr && channels != nullptr)
            reinterpret_cast<Anime4KCPP::AC*>(instance)->saveImageShape(*cols, *rows, *channels);
        else
            return AC_ERROR_NULL_DATA;

        return AC_OK;
    }

#ifdef ENABLE_VIDEO

    ac_videoProcessor acGetVideoProcessor(ac_parameters* parameters, ac_processType type, ac_error* error)
    {
        if (error != nullptr)
            *error = AC_OK;
        return reinterpret_cast<ac_instance>(new Anime4KCPP::VideoProcessor(getParameters(parameters), getProcessorType(type, error)));
    }

    ac_videoProcessor acGetVideoProcessorFromInstance(ac_instance instance)
    {
        return reinterpret_cast<ac_instance>(new Anime4KCPP::VideoProcessor(*reinterpret_cast<Anime4KCPP::AC*>(instance)));
    }

    void  acFreeVideoProcessor(ac_videoProcessor instance)
    {
        if (instance != nullptr)
        {
            delete reinterpret_cast<Anime4KCPP::VideoProcessor*>(instance);
            instance = nullptr;
        }
    }

    ac_error acLoadVideo(ac_videoProcessor instance, const char* srcFile)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        try
        {
            reinterpret_cast<Anime4KCPP::VideoProcessor*>(instance)->loadVideo(srcFile);
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_LOAD_VIDEO;
        }

        return AC_OK;
    }

    ac_error acProcessVideo(ac_videoProcessor instance)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        try
        {
            reinterpret_cast<Anime4KCPP::VideoProcessor*>(instance)->process();
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_GPU_PROCESS;
        }

        return AC_OK;
    }

    ac_error acProcessWithPrintProgress(ac_videoProcessor instance)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        try
        {
            reinterpret_cast<Anime4KCPP::VideoProcessor*>(instance)->processWithPrintProgress();
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_GPU_PROCESS;
        }

        return AC_OK;
    }

    ac_error acProcessWithProgress(ac_videoProcessor instance, void (*callBack)(double))
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        try
        {
            reinterpret_cast<Anime4KCPP::VideoProcessor*>(instance)->processWithProgress(callBack);
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_GPU_PROCESS;
        }

        return AC_OK;
    }

    ac_error acProcessWithProgressTime(ac_videoProcessor instance, void (*callBack)(double, double))
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        try
        {
            time_t start = time(nullptr);
            reinterpret_cast<Anime4KCPP::VideoProcessor*>(instance)->processWithProgress(
                [&callBack, &start](double v)
                {
                    callBack(v, static_cast<double>(time(nullptr) - start));
                });
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_GPU_PROCESS;
        }

        return AC_OK;
    }

    ac_error acStopVideoProcess(ac_videoProcessor instance)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::VideoProcessor*>(instance)->stopVideoProcess();

        return AC_OK;
    }

    ac_error acPauseVideoProcess(ac_videoProcessor instance)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::VideoProcessor*>(instance)->pauseVideoProcess();

        return AC_OK;
    }

    ac_error acContinueVideoProcess(ac_videoProcessor instance)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::VideoProcessor*>(instance)->continueVideoProcess();

        return AC_OK;
    }

    ac_error acSetSaveVideoInfo(ac_videoProcessor instance, const char* dstFile, ac_codec codec, double fps)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        try
        {
            reinterpret_cast<Anime4KCPP::VideoProcessor*>(instance)->setVideoSaveInfo(dstFile, Anime4KCPP::CODEC(codec), fps);
        }
        catch (const std::exception& err)
        {
            lastCoreError = err.what();
            return AC_ERROR_INIT_VIDEO_WRITER;
        }

        return AC_OK;
    }

    ac_error acSaveVideo(ac_videoProcessor instance)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        reinterpret_cast<Anime4KCPP::VideoProcessor*>(instance)->saveVideo();

        return AC_OK;
    }

#endif
}
