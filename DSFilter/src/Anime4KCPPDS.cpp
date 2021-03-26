#include "Anime4KCPPDS.h"

// Filter info
const AMOVIESETUP_MEDIATYPE sudPinTypes =
{
    &MEDIATYPE_Video,           // Major type
    &MEDIASUBTYPE_NULL          // Minor type
};

const AMOVIESETUP_PIN sudPins[] =
{
    {
        L"Input",
        FALSE,
        FALSE,
        FALSE,
        FALSE,
        &CLSID_NULL,
        L"Output",
        1,
        &sudPinTypes
    },
    {
        L"Output",
        FALSE,
        TRUE,
        FALSE,
        FALSE,
        &CLSID_NULL,
        L"Input",
        1,
        &sudPinTypes
    }
};

const AMOVIESETUP_FILTER sudAnime4KCPPDS =
{
    &CLSID_Anime4KCPPDS,
    L"Anime4KCPP for DirectShow",
    MERIT_DO_NOT_USE,
    2,
    sudPins
};

CFactoryTemplate g_Templates[] = {
    {
        L"Anime4KCPP for DirectShow",
        &CLSID_Anime4KCPPDS,
        Anime4KCPPDS::CreateInstance,
        NULL,
        &sudAnime4KCPPDS
    }
    ,
    {
        L"Anime4KCPP Settings",
        &CLSID_ACProp,
        ACProp::CreateInstance
    }
};

int g_cTemplates = sizeof(g_Templates) / sizeof(g_Templates[0]);

STDAPI DllRegisterServer()
{
    return AMovieDllRegisterServer2(TRUE);

} // DllRegisterServer

STDAPI DllUnregisterServer()
{
    return AMovieDllRegisterServer2(FALSE);

} // DllUnregisterServer

// Entry point
extern "C" BOOL WINAPI DllEntryPoint(HINSTANCE, ULONG, LPVOID);

HMODULE hDLLInstance;
BOOL APIENTRY DllMain(HANDLE hModule,
    DWORD  dwReason,
    LPVOID lpReserved)
{
    hDLLInstance = (HMODULE)hModule;
    return DllEntryPoint((HINSTANCE)(hModule), dwReason, lpReserved);
}

Anime4KCPPDS::Anime4KCPPDS(TCHAR* tszName,
    LPUNKNOWN punk,
    HRESULT* phr) :
    CTransformFilter(tszName, punk, CLSID_Anime4KCPPDS),
    srcH(0), srcW(0), dstH(0), dstW(0), dstDataLength(0),
    colorFormat(ColorFormat::YV12)
{
    //Get DLL path and set config path
    GetModuleFileName(hDLLInstance, lpPath, MAX_PATH);
    int iPathLen = lstrlen(lpPath);
    lpPath[iPathLen - 3] = L'i';
    lpPath[iPathLen - 2] = L'n';
    lpPath[iPathLen - 1] = L'i';

    TCHAR _zoomFactor[10];
    TCHAR _GPGPUModelString[10];
    //read config
    pID = GetPrivateProfileInt(L"Anime4KCPP for DirectShow Config", L"pID", 0, lpPath);
    dID = GetPrivateProfileInt(L"Anime4KCPP for DirectShow Config", L"dID", 0, lpPath);
    CNN = GetPrivateProfileInt(L"Anime4KCPP for DirectShow Config", L"ACNet", 1, lpPath);
    H = GetPrivateProfileInt(L"Anime4KCPP for DirectShow Config", L"H", 1080, lpPath);
    W = GetPrivateProfileInt(L"Anime4KCPP for DirectShow Config", L"W", 1920, lpPath);
    parameters.HDN = GetPrivateProfileInt(L"Anime4KCPP for DirectShow Config", L"HDN", 0, lpPath);
    parameters.HDNLevel = GetPrivateProfileInt(L"Anime4KCPP for DirectShow Config", L"HDNLevel", 1, lpPath);
    OpenCLParallelIO = GetPrivateProfileInt(L"Anime4KCPP for DirectShow Config", L"OpenCLParallelIO", 0, lpPath);
    OpenCLQueueNum = GetPrivateProfileInt(L"Anime4KCPP for DirectShow Config", L"OpenCLQueueNum", 4, lpPath);
    if (OpenCLQueueNum < 1)
        OpenCLQueueNum = 1;

    GetPrivateProfileString(L"Anime4KCPP for DirectShow Config", L"GPGPUModel", L"OpenCL", _GPGPUModelString, 10, lpPath);
    GetPrivateProfileString(L"Anime4KCPP for DirectShow Config", L"zoomFactor", L"2.0", _zoomFactor, 10, lpPath);
    zf =  _wtof(_zoomFactor);
    zf = parameters.zoomFactor = zf >= 1.0 ? zf : 1.0;

    if (!wcsicmp(_GPGPUModelString, L"OpenCL"))
        GPGPUModel = GPGPU::OpenCL;
    else if (!wcsicmp(_GPGPUModelString, L"CUDA"))
        GPGPUModel = GPGPU::CUDA;
    else
        GPGPUModel = GPGPU::OpenCL;

    if (!CheckGPUSupport())
        GPUCheckResult = E_FAIL;
    else
        GPUCheckResult = S_OK;

    Anime4KCPP::CNNType type;
    if (parameters.HDN)
    {
        switch (parameters.HDNLevel)
        {
        case 1:
            type = Anime4KCPP::CNNType::ACNetHDNL1;
            break;
        case 2:
            type = Anime4KCPP::CNNType::ACNetHDNL2;
            break;
        case 3:
            type = Anime4KCPP::CNNType::ACNetHDNL3;
            break;
        default:
            type = Anime4KCPP::CNNType::ACNetHDNL1;
            break;
        }
    }
    else
        type = Anime4KCPP::CNNType::ACNetHDNL0;

    acCreator.deinit(true);
    switch (GPGPUModel)
    {
    case GPGPU::OpenCL:
        if (CNN)
            acCreator.pushManager<Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::ACNet>>(
                pID, dID,
                type,
                OpenCLQueueNum,
                OpenCLParallelIO);
        else
            acCreator.pushManager<Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::Anime4K09>>(
                pID, dID,
                OpenCLQueueNum,
                OpenCLParallelIO);
        break;
    case GPGPU::CUDA:
#ifdef ENABLE_CUDA
        acCreator.pushManager<Anime4KCPP::Cuda::Manager>(dID);
#else
        if (CNN)
            acCreator.pushManager<Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::ACNet>>(
                pID, dID,
                Anime4KCPP::CNNType::Default,
                OpenCLQueueNum,
                OpenCLParallelIO);
        else
            acCreator.pushManager<Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::Anime4K09>>(
                pID, dID,
                OpenCLQueueNum,
                OpenCLParallelIO);
#endif
        break;
    }
}

BOOL Anime4KCPPDS::IsRGB24(const CMediaType* pMediaType) const
{
    if (IsEqualGUID(*pMediaType->Type(), MEDIATYPE_Video))
    {
        if (IsEqualGUID(*pMediaType->Subtype(), MEDIASUBTYPE_RGB24))
        {
            return TRUE;
        }
    }

    return FALSE;
}

BOOL Anime4KCPPDS::IsRGB32(const CMediaType* pMediaType) const
{
    if (IsEqualGUID(*pMediaType->Type(), MEDIATYPE_Video))
    {
        if (IsEqualGUID(*pMediaType->Subtype(), MEDIASUBTYPE_RGB32))
        {
            return TRUE;
        }
    }

    return FALSE;
}

BOOL Anime4KCPPDS::IsIYUV(const CMediaType* pMediaType) const
{
    if (IsEqualGUID(*pMediaType->Type(), MEDIATYPE_Video))
    {
        if (IsEqualGUID(*pMediaType->Subtype(), MEDIASUBTYPE_IYUV))
        {
            return TRUE;
        }
    }

    return FALSE;
}

BOOL Anime4KCPPDS::IsYV12(const CMediaType* pMediaType) const
{
    if (IsEqualGUID(*pMediaType->Type(), MEDIATYPE_Video))
    {
        if (IsEqualGUID(*pMediaType->Subtype(), MEDIASUBTYPE_YV12))
        {
            return TRUE;
        }
    }

    return FALSE;
}

BOOL Anime4KCPPDS::IsNV12(const CMediaType* pMediaType) const
{
    if (IsEqualGUID(*pMediaType->Type(), MEDIATYPE_Video))
    {
        if (IsEqualGUID(*pMediaType->Subtype(), MEDIASUBTYPE_NV12))
        {
            return TRUE;
        }
    }

    return FALSE;
}

BOOL Anime4KCPPDS::IsP016(const CMediaType* pMediaType) const
{
    if (IsEqualGUID(*pMediaType->Type(), MEDIATYPE_Video))
    {
        if (IsEqualGUID(*pMediaType->Subtype(), MEDIASUBTYPE_P010) ||
            IsEqualGUID(*pMediaType->Subtype(), MEDIASUBTYPE_P016))
        {
            return TRUE;
        }
    }

    return FALSE;
}

CUnknown* Anime4KCPPDS::CreateInstance(LPUNKNOWN punk, HRESULT* phr)
{
    Anime4KCPPDS* pNewObject = new Anime4KCPPDS(NAME("Anime4KCPP for DirectShow"), punk, phr);

    if (pNewObject == nullptr)
    {
        if (phr)
            *phr = E_OUTOFMEMORY;
    }

    return pNewObject;
}

STDMETHODIMP Anime4KCPPDS::NonDelegatingQueryInterface(REFIID riid, void** ppv)
{
    if (riid == IID_IAC)
        return GetInterface((IAC*)this, ppv);
    else if (riid == IID_ISpecifyPropertyPages)
        return GetInterface((ISpecifyPropertyPages*)this, ppv);
    else
        return CTransformFilter::NonDelegatingQueryInterface(riid, ppv);
}

HRESULT Anime4KCPPDS::CheckInputType(const CMediaType* mtIn)
{
    CheckPointer(mtIn, E_POINTER);

    // VIDEOINFOHEADER and VIDEOINFOHEADER2 is supported
    if (!IsEqualGUID(*mtIn->FormatType(), FORMAT_VideoInfo2) && 
        !IsEqualGUID(*mtIn->FormatType(), FORMAT_VideoInfo))
        return VFW_E_TYPE_NOT_ACCEPTED;

    //Resolution check
    if (IsEqualGUID(mtIn->formattype, FORMAT_VideoInfo2))
    {
        VIDEOINFOHEADER2* pVi = (VIDEOINFOHEADER2*)mtIn->pbFormat;
        CheckPointer(pVi, E_INVALIDARG);

        if (H < std::abs(pVi->bmiHeader.biHeight) || W < pVi->bmiHeader.biWidth)
            return VFW_E_TYPE_NOT_ACCEPTED;
    }
    else
    {
        VIDEOINFOHEADER* pVi = (VIDEOINFOHEADER*)mtIn->pbFormat;
        CheckPointer(pVi, E_INVALIDARG);

        if (H < std::abs(pVi->bmiHeader.biHeight) || W < pVi->bmiHeader.biWidth)
            return VFW_E_TYPE_NOT_ACCEPTED;
    }

    // Can we transform this type
    if (IsYV12(mtIn) || IsIYUV(mtIn) || IsNV12(mtIn) || IsRGB24(mtIn) || IsRGB32(mtIn) || IsP016(mtIn))
        return GPUCheckResult;

    return VFW_E_TYPE_NOT_ACCEPTED;
}

HRESULT Anime4KCPPDS::CheckTransform(const CMediaType* mtIn, const CMediaType* mtOut)
{
    CheckPointer(mtIn, E_POINTER);
    CheckPointer(mtOut, E_POINTER);

    if (!IsEqualGUID(*mtOut->FormatType(), FORMAT_VideoInfo2) && 
        !IsEqualGUID(*mtOut->FormatType(), FORMAT_VideoInfo))
        return VFW_E_TYPE_NOT_ACCEPTED;

    acCreator.init();

    if (CNN && IsYV12(mtIn) && IsYV12(mtOut))
    {
        colorFormat = ColorFormat::YV12;
        return S_OK;
    }
    if (CNN && IsIYUV(mtIn) && IsIYUV(mtOut))
    {
        colorFormat = ColorFormat::IYUV;
        return S_OK;
    }
    if (CNN && IsNV12(mtIn) && IsNV12(mtOut))
    {
        colorFormat = ColorFormat::NV12;
        return S_OK;
    }
    if (CNN && IsP016(mtIn) && IsP016(mtOut))
    {
        colorFormat = ColorFormat::P016;
        return S_OK;
    }
    if (IsRGB24(mtIn) && IsRGB24(mtOut))
    {
        colorFormat = ColorFormat::RGB24;
        return S_OK;
    }
    if (IsRGB32(mtIn) && IsRGB32(mtOut))
    {
        colorFormat = ColorFormat::RGB32;
        return S_OK;
    }

    return VFW_E_TYPE_NOT_ACCEPTED;
}

HRESULT Anime4KCPPDS::DecideBufferSize(IMemAllocator* pAlloc, ALLOCATOR_PROPERTIES* pProperties)
{
    // Is the input pin connecte
    if (!m_pInput->IsConnected())
        return E_UNEXPECTED;

    CheckPointer(pAlloc, E_POINTER);
    CheckPointer(pProperties, E_POINTER);

    pProperties->cbBuffer = dstDataLength;

    if (pProperties->cbAlign == 0)
        pProperties->cbAlign = 1;
    if (pProperties->cBuffers == 0)
        pProperties->cBuffers = 1;

    ALLOCATOR_PROPERTIES Actual;
    HRESULT hr = pAlloc->SetProperties(pProperties, &Actual);
    if (FAILED(hr))
        return hr;

    if (pProperties->cBuffers > Actual.cBuffers || pProperties->cbBuffer > Actual.cbBuffer)
        return E_FAIL;

    return S_OK;
}

HRESULT Anime4KCPPDS::GetMediaType(int iPosition, CMediaType* pMediaType)
{
    if (!m_pInput->IsConnected())
        return E_UNEXPECTED;

    if (iPosition < 0)
        return E_INVALIDARG;

    if (iPosition > 0)
        return VFW_S_NO_MORE_ITEMS;

    CheckPointer(pMediaType, E_POINTER);

    HRESULT hr = m_pInput->ConnectionMediaType(pMediaType);
    if (FAILED(hr))
        return hr;

    //resize
    if (IsEqualGUID(pMediaType->formattype, FORMAT_VideoInfo2))
    {
        VIDEOINFOHEADER2* pVi = (VIDEOINFOHEADER2*)pMediaType->pbFormat;
        CheckPointer(pVi, E_INVALIDARG);

        srcH = std::abs(pVi->bmiHeader.biHeight);
        srcW = std::abs(pVi->bmiHeader.biWidth);
        dstH = std::round(srcH * zf);
        dstW = std::round(srcW * zf);
        dstDataLength = std::ceil(pVi->bmiHeader.biSizeImage * zf * zf);
        pVi->bmiHeader.biHeight = dstH;
        pVi->bmiHeader.biWidth = dstW;
        pVi->bmiHeader.biSizeImage = dstDataLength;
        pMediaType->SetSampleSize(dstDataLength);

        SetRect(&pVi->rcSource, 0, 0, dstW, dstH);
        SetRect(&pVi->rcTarget, 0, 0, dstW, dstH);
    }
    else
    {
        VIDEOINFOHEADER* pVi = (VIDEOINFOHEADER*)pMediaType->pbFormat;
        CheckPointer(pVi, E_INVALIDARG);

        srcH = std::abs(pVi->bmiHeader.biHeight);
        srcW = std::abs(pVi->bmiHeader.biWidth);
        dstH = std::round(srcH * zf);
        dstW = std::round(srcW * zf);
        dstDataLength = std::ceil(pVi->bmiHeader.biSizeImage * zf * zf);
        pVi->bmiHeader.biHeight = dstH;
        pVi->bmiHeader.biWidth = dstW;
        pVi->bmiHeader.biSizeImage = dstDataLength;
        pMediaType->SetSampleSize(dstDataLength);

        SetRect(&pVi->rcSource, 0, 0, dstW, dstH);
        SetRect(&pVi->rcTarget, 0, 0, dstW, dstH);
    }

    return S_OK;
}

HRESULT Anime4KCPPDS::Transform(IMediaSample* pIn, IMediaSample* pOut)
{
    HRESULT hr = S_OK;

    BYTE* pBufferIn, * pBufferOut;
    hr = pIn->GetPointer(&pBufferIn);
    if (FAILED(hr))
        return hr;
    hr = pOut->GetPointer(&pBufferOut);
    if (FAILED(hr))
        return hr;

    Anime4KCPP::AC* ac = nullptr;
    switch (GPGPUModel)
    {
    case GPGPU::OpenCL:
        if (CNN)
            ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::OpenCL_ACNet);
        else
            ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::OpenCL_Anime4K09);
        break;
    case GPGPU::CUDA:
#ifdef ENABLE_CUDA
        if (CNN)
            ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::Cuda_ACNet);
        else
            ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::Cuda_Anime4K09);
#else
        if (CNN)
            ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::OpenCL_ACNet);
        else
            ac = acCreator.create(parameters, Anime4KCPP::Processor::Type::OpenCL_Anime4K09);
#endif
        break;
    }

    switch (colorFormat)
    {
    case ColorFormat::IYUV:
    case ColorFormat::YV12:
    {
        size_t srcSize = (pIn->GetActualDataLength() << 1) / 3;
        size_t dstSize = (pOut->GetActualDataLength() << 1) / 3;
        size_t strideY = dstSize / dstH;
        size_t strideUV = strideY >> 1;

        BYTE* pYIn = pBufferIn,
            * pUIn = pYIn + srcSize,
            * pVIn = pUIn + (srcSize >> 2);
        BYTE* pYOut = pBufferOut,
            * pUOut = pYOut + dstSize,
            * pVOut = pUOut + (dstSize >> 2);

        ac->loadImage(
            srcH, srcW, 0, pYIn,
            srcH >> 1, srcW >> 1, 0, pUIn,
            srcH >> 1, srcW >> 1, 0, pVIn);
        ac->process();
        ac->saveImage(pYOut, strideY, pUOut, strideUV, pVOut, strideUV);
    }
    break;
    case ColorFormat::NV12:
    {
        size_t srcSize = (pIn->GetActualDataLength() << 1) / 3;
        size_t dstSize = (pOut->GetActualDataLength() << 1) / 3;
        size_t stride = dstSize / dstH;
        size_t dstHUV = dstH >> 1;

        BYTE* pYIn = pBufferIn,
            * pUVIn = pYIn + srcSize;
        BYTE* pYOut = pBufferOut,
            * pUVOut = pYOut + dstSize;

        cv::Mat dstY;
        cv::Mat dstUV;

        cv::Mat srcY(srcH, srcW, CV_8UC1, pYIn);
        cv::Mat srcUV(srcH >> 1, srcW >> 1, CV_8UC2, pUVIn);
        
        ac->loadImage(srcY);
        ac->process();
        ac->saveImage(dstY);

        cv::resize(srcUV, dstUV, cv::Size(dstW >> 1, dstHUV), 0.0, 0.0, cv::INTER_CUBIC);

        if (stride == dstW)
        {
            memcpy(pYOut, dstY.data, dstSize);
            memcpy(pUVOut, dstUV.data, dstSize >> 1);
        }
        else
        {
            for (size_t y = 0; y < dstH; y++)
            {
                memcpy(pYOut, dstY.data + y * dstW, dstW);
                pYOut += stride;
                if (y < dstHUV)
                {
                    memcpy(pUVOut, dstUV.data + y * dstW, dstW);
                    pUVOut += stride;
                }
            }
        }
    }
    break;
    case ColorFormat::P016:
    {
        size_t srcSize = (pIn->GetActualDataLength() << 1) / 3;
        size_t dstSize = (pOut->GetActualDataLength() << 1) / 3;
        size_t stride = dstSize / dstH;
        size_t dstHUV = dstH >> 1;

        BYTE* pYIn = pBufferIn,
            * pUVIn = pYIn + srcSize;
        BYTE* pYOut = pBufferOut,
            * pUVOut = pYOut + dstSize;

        cv::Mat dstY;
        cv::Mat dstUV;

        cv::Mat srcY(srcH, srcW, CV_16UC1, pYIn);
        cv::Mat srcUV(srcH >> 1, srcW >> 1, CV_16UC2, pUVIn);

        ac->loadImage(srcY);
        ac->process();
        ac->saveImage(dstY);

        cv::resize(srcUV, dstUV, cv::Size(dstW >> 1, dstHUV), 0.0, 0.0, cv::INTER_CUBIC);

        if (stride == dstW)
        {
            memcpy(pYOut, dstY.data, dstSize);
            memcpy(pUVOut, dstUV.data, dstSize >> 1);
        }
        else
        {
            for (size_t y = 0; y < dstH; y++)
            {
                memcpy(pYOut, (WORD*)dstY.data + y * dstW, dstW * sizeof(WORD));
                pYOut += stride;
                if (y < dstHUV)
                {
                    memcpy(pUVOut, (WORD*)dstUV.data + y * dstW, dstW * sizeof(WORD));
                    pUVOut += stride;
                }
            }
        }
    }
    break;
    case ColorFormat::RGB24:
    {
        size_t stride = pOut->GetActualDataLength() / dstH;
        cv::Mat srcTmp(srcH, srcW, CV_8UC3, pBufferIn);
        cv::flip(srcTmp, srcTmp, 0);
        ac->loadImage(srcH, srcW, 0, srcTmp.data);
        ac->process();
        ac->saveImage(pBufferOut, stride);
    }
    break;
    case ColorFormat::RGB32:
    {
        size_t stride = pOut->GetActualDataLength() / dstH;
        cv::Mat srcTmp(srcH, srcW, CV_8UC4, pBufferIn);
        cv::flip(srcTmp, srcTmp, 0);
        ac->loadImage(srcH, srcW, 0, srcTmp.data, false, true);
        ac->process();
        ac->saveImage(pBufferOut, stride);
    }
    break;
    }

    acCreator.release(ac);

    return hr;
}

BOOL Anime4KCPPDS::CheckGPUSupport()
{
    switch (GPGPUModel)
    {
    case GPGPU::OpenCL:
    {
        Anime4KCPP::OpenCL::GPUInfo ret = Anime4KCPP::OpenCL::checkGPUSupport(pID, dID);
        if (!ret)
        {
            MessageBoxExA(nullptr, ret().c_str(), "Anime4KCPPDS Error", MB_APPLMODAL | MB_ICONERROR, LANG_ENGLISH);
            return FALSE;
        }
    }
    break;
    case GPGPU::CUDA:
#ifdef ENABLE_CUDA
    {
        Anime4KCPP::Cuda::GPUInfo ret = Anime4KCPP::Cuda::checkGPUSupport(dID);
        if (!ret)
        {
            MessageBoxExA(nullptr, ret().c_str(), "Anime4KCPPDS Error", MB_APPLMODAL | MB_ICONERROR, LANG_ENGLISH);
            return FALSE;
        }
    }
#else
        MessageBoxExA(nullptr, "CUDA is not supported", "Anime4KCPPDS Error", MB_APPLMODAL | MB_ICONERROR, LANG_ENGLISH);
        return FALSE;
#endif 
    break;
    }

    return TRUE;
}

STDMETHODIMP Anime4KCPPDS::GetParameters(bool* HDN, int* HDNLevel, bool* CNN, unsigned int* pID, unsigned int* dID, double* zoomFactor, int* H, int* W, int* GPGPUModel, int* OpenCLQueueNum, bool* OpenCLParallelIO)
{
    *HDN = parameters.HDN;
    *HDNLevel = parameters.HDNLevel;
    *CNN = this->CNN;
    *pID = this->pID;
    *dID = this->dID;
    *H = this->H;
    *W = this->W;
    *zoomFactor = zf;
    *GPGPUModel = this->GPGPUModel;
    *OpenCLQueueNum = this->OpenCLQueueNum;
    *OpenCLParallelIO = this->OpenCLParallelIO;

    return NOERROR;
}

STDMETHODIMP Anime4KCPPDS::SetParameters(bool HDN, int HDNLevel, bool CNN, unsigned int pID, unsigned int dID, double zoomFactor, int H, int W, int GPGPUModel, int OpenCLQueueNum, bool OpenCLParallelIO)
{
    CAutoLock cAutoLock(&lock);

    TCHAR _pID[10], _dID[10], _OpenCLQueueNum[10], _OpenCLParallelIO[10], _CNN[10], _HDN[10], _HDNLevel[10], _H[10], _W[10], _zoomFactor[10], * _GPGPUModel = nullptr;

    //convert to string
    _itow_s(pID, _pID, 10, 10);
    _itow_s(dID, _dID, 10, 10);
    _itow_s(OpenCLQueueNum, _OpenCLQueueNum, 10, 10);
    _itow_s(OpenCLParallelIO, _OpenCLParallelIO, 10, 10);
    _itow_s(dID, _dID, 10, 10);
    _itow_s(CNN, _CNN, 10, 10);
    _itow_s(H, _H, 10, 10);
    _itow_s(W, _W, 10, 10);
    _itow_s(HDN, _HDN, 10, 10);
    _itow_s(HDNLevel, _HDNLevel, 10, 10);

    swprintf(_zoomFactor, 10, L"%f", zoomFactor);
    switch (GPGPUModel)
    {
    case GPGPU::OpenCL:
        _GPGPUModel = L"OpenCL";
        break;
    case GPGPU::CUDA:
        _GPGPUModel = L"CUDA";
        break;
    default:
        _GPGPUModel = L"OpenCL";
    }
    
    //write config
    WritePrivateProfileString(L"Anime4KCPP for DirectShow Config", L"pID", _pID, lpPath);
    WritePrivateProfileString(L"Anime4KCPP for DirectShow Config", L"dID", _dID, lpPath);
    WritePrivateProfileString(L"Anime4KCPP for DirectShow Config", L"OpenCLQueueNum", _OpenCLQueueNum, lpPath);
    WritePrivateProfileString(L"Anime4KCPP for DirectShow Config", L"OpenCLParallelIO", _OpenCLParallelIO, lpPath);
    WritePrivateProfileString(L"Anime4KCPP for DirectShow Config", L"ACNet", _CNN, lpPath);
    WritePrivateProfileString(L"Anime4KCPP for DirectShow Config", L"HDN", _HDN, lpPath);
    WritePrivateProfileString(L"Anime4KCPP for DirectShow Config", L"HDNLevel", _HDNLevel, lpPath);
    WritePrivateProfileString(L"Anime4KCPP for DirectShow Config", L"H", _H, lpPath);
    WritePrivateProfileString(L"Anime4KCPP for DirectShow Config", L"W", _W, lpPath);
    WritePrivateProfileString(L"Anime4KCPP for DirectShow Config", L"zoomFactor", _zoomFactor, lpPath);
    WritePrivateProfileString(L"Anime4KCPP for DirectShow Config", L"GPGPUModel", _GPGPUModel, lpPath);

    return NOERROR;
}

STDMETHODIMP Anime4KCPPDS::GetGPUInfo(std::string& info)
{
    std::string tmpStr;
    switch (GPGPUModel)
    {
    case GPGPU::OpenCL:
    {
        Anime4KCPP::OpenCL::GPUList GPUInfo = Anime4KCPP::OpenCL::listGPUs();
        tmpStr = GPUInfo();
    }
    break;
    case GPGPU::CUDA:
#ifdef ENABLE_CUDA
    {
        Anime4KCPP::Cuda::GPUList GPUInfo = Anime4KCPP::Cuda::listGPUs();
        tmpStr = GPUInfo();
    }
#else
    {
        Anime4KCPP::OpenCL::GPUList GPUInfo = Anime4KCPP::OpenCL::listGPUs();
        tmpStr = GPUInfo();
    }
#endif
    break;
    }


    size_t tmp = 0;
    std::vector<std::string> subInfo(4);
    for (size_t i = 0; i < tmpStr.size(); i++)
    {
        if (tmpStr[i] == '\n')
        {
            subInfo.emplace_back(tmpStr.substr(tmp, i - tmp));
            tmp = i;
        }
    }
    for (auto& s : subInfo)
    {
        info += (s + "\r");
    }

    return NOERROR;
}

STDMETHODIMP Anime4KCPPDS::GetPages(CAUUID* pPages)
{
    CheckPointer(pPages, E_POINTER);

    pPages->cElems = 1;
    pPages->pElems = (GUID*)CoTaskMemAlloc(sizeof(GUID));
    if (pPages->pElems == NULL)
        return E_OUTOFMEMORY;

    *(pPages->pElems) = CLSID_ACProp;

    return NOERROR;
}
