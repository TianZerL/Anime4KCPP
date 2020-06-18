#define AC_DLL

#include "Anime4KCPP.h"
#include "AC.h"

Anime4KCPP::Parameters getParameters(ac_parameters *c_parameters)
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
        c_parameters->HDN);

    return std::move(cpp_parameters);
}

extern "C"
{
    ac_instance acGetInstance(ac_bool initGPU, ac_bool initGPUCNN, unsigned int platformID, unsigned int deviceID, ac_parameters *parameters, ac_processType type, ac_error *error)
    {
        if (error != nullptr)
            *error = AC_OK;

        if (initGPU == AC_TRUE && !Anime4KCPP::Anime4KGPU::isInitializedGPU())
        {
            try
            {
                Anime4KCPP::Anime4KGPU::initGPU(platformID, deviceID);
            }
            catch (const char *err)
            {
                if (error != nullptr)
                    *error = AC_ERROR_INIT_GPU;
                return nullptr;
            }
        }

        if (initGPUCNN == AC_TRUE && !Anime4KCPP::Anime4KGPUCNN::isInitializedGPU())
        {
            try
            {
                Anime4KCPP::Anime4KGPUCNN::initGPU(platformID, deviceID);
            }
            catch (const char *err)
            {
                if (error != nullptr)
                    *error = AC_ERROR_INIT_GPU;
                return nullptr;
            }
        }

        switch (type)
        {
        case AC_CPU:
            return static_cast<ac_instance>(new Anime4KCPP::Anime4KCPU(getParameters(parameters)));
            break;
        case AC_GPU:
            return static_cast<ac_instance>(new Anime4KCPP::Anime4KGPU(getParameters(parameters)));
            break;
        case AC_CPUCNN:
            return static_cast<ac_instance>(new Anime4KCPP::Anime4KCPUCNN(getParameters(parameters)));
            break;
        case AC_GPUCNN:
            return static_cast<ac_instance>(new Anime4KCPP::Anime4KGPUCNN(getParameters(parameters)));
            break;
        default:
            if (error != nullptr)
                *error = AC_ERROR_PORCESSOR_TYPE;
            return nullptr;
        }
    }

    void acFreeInstance(ac_instance instance, ac_bool releaseGPU, ac_bool releaseGPUCNN)
    {
        if (instance != nullptr)
            delete static_cast<Anime4KCPP::Anime4K *>(instance);

        if (releaseGPU == AC_TRUE && Anime4KCPP::Anime4KGPU::isInitializedGPU())
            Anime4KCPP::Anime4KGPU::releaseGPU();

        if (releaseGPUCNN == AC_TRUE && Anime4KCPP::Anime4KGPUCNN::isInitializedGPU())
            Anime4KCPP::Anime4KGPUCNN::releaseGPU();
    }

    ac_error acInitParameters(ac_parameters *parameters)
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

    ac_error acLoadImage(ac_instance instance, const char *srcFile)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        try
        {
            static_cast<Anime4KCPP::Anime4K *>(instance)->loadImage(srcFile);
        }
        catch (const char *err)
        {
            return AC_ERROR_LOAD_IMAGE;
        }

        return AC_OK;
    }

    ac_error acLoadVideo(ac_instance instance, const char *srcFile)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        try
        {
            static_cast<Anime4KCPP::Anime4K *>(instance)->loadVideo(srcFile);
        }
        catch (const char *err)
        {
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
            static_cast<Anime4KCPP::Anime4K *>(instance)->process();
        }
        catch (const char *err)
        {
            return AC_ERROR_GPU_PROCESS;
        }

        return AC_OK;
    }

    ac_error acShowImage(ac_instance instance)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        static_cast<Anime4KCPP::Anime4K *>(instance)->showImage();

        return AC_OK;
    }

    ac_error acSaveImage(ac_instance instance, const char *dstFile)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        static_cast<Anime4KCPP::Anime4K *>(instance)->saveImage(dstFile);

        return AC_OK;
    }

    ac_error acSetSaveVideoInfo(ac_instance instance, const char *dstFile, ac_codec codec)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        try
        {
            static_cast<Anime4KCPP::Anime4K *>(instance)->setVideoSaveInfo(dstFile, Anime4KCPP::CODEC(codec));
        }
        catch (const char *err)
        {
            return AC_ERROR_INIT_VIDEO_WRITER;
        }

        return AC_OK;
    }

    ac_error acSaveVideo(ac_instance instance)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        static_cast<Anime4KCPP::Anime4K *>(instance)->saveVideo();

        return AC_OK;
    }

    ac_error acSetArguments(ac_instance instance, ac_parameters *parameters)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        static_cast<Anime4KCPP::Anime4K *>(instance)->setArguments(getParameters(parameters));

        return AC_OK;
    }

    ac_error acSetVideoMode(ac_instance instance, ac_bool flag)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        static_cast<Anime4KCPP::Anime4K *>(instance)->setVideoMode(flag);

        return AC_OK;
    }

    ac_error acInitGPU(void)
    {
        try
        {
            if (!Anime4KCPP::Anime4KGPU::isInitializedGPU())
                Anime4KCPP::Anime4KGPU::initGPU();
        }
        catch (const char *err)
        {
            return AC_ERROR_INIT_GPU;
        }

        return AC_OK;
    }

    void acReleaseGPU(void)
    {
        if (Anime4KCPP::Anime4KGPU::isInitializedGPU())
            Anime4KCPP::Anime4KGPU::releaseGPU();
    }

    ac_error acInitGPUCNN(void)
    {
        try
        {
            if (!Anime4KCPP::Anime4KGPUCNN::isInitializedGPU())
                Anime4KCPP::Anime4KGPUCNN::initGPU();
        }
        catch (const char *err)
        {
            return AC_ERROR_INIT_GPU;
        }

        return AC_OK;
    }

    void acReleaseGPUCNN(void)
    {
        if (Anime4KCPP::Anime4KGPUCNN::isInitializedGPU())
            Anime4KCPP::Anime4KGPUCNN::releaseGPU();
    }

    ac_error acLoadImageRGB(ac_instance instance, int rows, int cols, unsigned char *r, unsigned char *g, unsigned char *b, ac_bool inputAsYUV)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        static_cast<Anime4KCPP::Anime4K*>(instance)->loadImage(rows, cols, r, g, b, inputAsYUV);

        return AC_OK;
    }

    ac_error acLoadImageRGBBytes(ac_instance instance, int rows, int cols, unsigned char *data, size_t bytesPerLine, ac_bool inputAsYUV)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        static_cast<Anime4KCPP::Anime4K*>(instance)->loadImage(rows, cols, data, bytesPerLine, inputAsYUV);

        return AC_OK;
    }

    ac_error acSaveImageRGB(ac_instance instance, unsigned char **r, unsigned char **g, unsigned char **b)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        try
        {
            static_cast<Anime4KCPP::Anime4K *>(instance)->saveImage(*r, *g, *b);
        }
        catch (const char *err)
        {
            return AC_ERROR_SAVE_TO_NULL_POINTER;
        }

        return AC_OK;
    }

    ac_error acSaveImageRGBBytes(ac_instance instance, unsigned char **data)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        try
        {
            static_cast<Anime4KCPP::Anime4K *>(instance)->saveImage(*data);
        }
        catch (const char *err)
        {
            return AC_ERROR_SAVE_TO_NULL_POINTER;
        }

        return AC_OK;
    }

    size_t acGetResultDataLength(ac_instance instance, ac_error *error)
    {
        if (error != nullptr)
            *error = AC_OK;

        if (instance == nullptr)
        {
            if (error != nullptr)
                *error = AC_ERROR_NULL_INSTANCE;
            return 0;
        }

        return static_cast<Anime4KCPP::Anime4K *>(instance)->getResultDataLength();
    }

    size_t acGetResultDataPerChannelLength(ac_instance instance, ac_error *error)
    {
        if (error != nullptr)
            *error = AC_OK;

        if (instance == nullptr)
        {
            if (error != nullptr)
                *error = AC_ERROR_NULL_INSTANCE;
            return 0;
        }

        return static_cast<Anime4KCPP::Anime4K *>(instance)->getResultDataPerChannelLength();
    }

    ac_error acGetInfo(ac_instance instance, char *info, size_t *length)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        if (info == nullptr && length == nullptr)
            return AC_OK;

        std::string ret = static_cast<Anime4KCPP::Anime4K *>(instance)->getInfo();

        if (length != nullptr)
            *length = ret.size() + 1;

        if (info != nullptr)
            memcpy(info, ret.c_str(), ret.size() + 1);

        return AC_OK;
    }

    ac_error acGetFiltersInfo(ac_instance instance, char *info, size_t *length)
    {
        if (instance == nullptr)
            return AC_ERROR_NULL_INSTANCE;

        if (info == nullptr && length == nullptr)
            return AC_OK;

        std::string ret = static_cast<Anime4KCPP::Anime4K *>(instance)->getFiltersInfo();

        if (length != nullptr)
            *length = ret.size() + 1;

        if (info != nullptr)
            memcpy(info, ret.c_str(), ret.size() + 1);

        return AC_OK;
    }

    ac_bool AC_API acCheckGPUSupport(unsigned int pID, unsigned int dID, char *info, size_t *length)
    {
        std::pair<bool, std::string> ret = Anime4KCPP::Anime4KGPU::checkGPUSupport(pID, dID);

        ac_bool rst = ac_bool(ret.first);

        if (length != nullptr)
            *length = ret.second.size() + 1;

        if (info != nullptr)
            memcpy(info, ret.second.c_str(), ret.second.size() + 1);

        return rst;
    }

    void acListGPUs(char *info, size_t *length, size_t *platforms, size_t *devices)
    {
        std::pair<std::pair<int, std::vector<int>>, std::string> ret = Anime4KCPP::Anime4KGPU::listGPUs();

        if (length != nullptr)
            *length = ret.second.size() + 1;

        if (info != nullptr)
            memcpy(info, ret.second.c_str(), ret.second.size() + 1);

        if (platforms != nullptr)
            *platforms = ret.first.first;

        if (devices != nullptr)
            for (int i : ret.first.second)
                *(devices++) = i;
    }

    ac_bool acIsInitializedGPU(void)
    {
        return ac_bool(Anime4KCPP::Anime4KGPU::isInitializedGPU());
    }

    ac_bool acIsInitializedGPUCNN(void)
    {
        return ac_bool(Anime4KCPP::Anime4KGPUCNN::isInitializedGPU());
    }
}
