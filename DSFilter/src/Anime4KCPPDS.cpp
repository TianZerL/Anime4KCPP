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
    acCreator(false, false),
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
    //read config
    pID = GetPrivateProfileInt(L"Anime4KCPP for DirectShow Config", L"pID", 0, lpPath);
    dID = GetPrivateProfileInt(L"Anime4KCPP for DirectShow Config", L"dID", 0, lpPath);
    CNN = GetPrivateProfileInt(L"Anime4KCPP for DirectShow Config", L"ACNet", 1, lpPath);
    H = GetPrivateProfileInt(L"Anime4KCPP for DirectShow Config", L"H", 1080, lpPath);
    W = GetPrivateProfileInt(L"Anime4KCPP for DirectShow Config", L"W", 1920, lpPath);
    parameters.HDN = GetPrivateProfileInt(L"Anime4KCPP for DirectShow Config", L"HDN", 0, lpPath);

    GetPrivateProfileStringW(L"Anime4KCPP for DirectShow Config", L"zoomFactor", L"2.0", _zoomFactor, 10, lpPath);
    zf =  _wtof(_zoomFactor);
    zf = parameters.zoomFactor = zf >= 1.0F ? zf : 1.0F;
}

inline BOOL Anime4KCPPDS::IsRGB24(const CMediaType* pMediaType) const
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

inline BOOL Anime4KCPPDS::IsRGB32(const CMediaType* pMediaType) const
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

inline BOOL Anime4KCPPDS::IsIYUV(const CMediaType* pMediaType) const
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

inline BOOL Anime4KCPPDS::IsYV12(const CMediaType* pMediaType) const
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

inline BOOL Anime4KCPPDS::IsNV12(const CMediaType* pMediaType) const
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

CUnknown* Anime4KCPPDS::CreateInstance(LPUNKNOWN punk, HRESULT* phr)
{
    Anime4KCPPDS* pNewObject = new Anime4KCPPDS(NAME("Anime4KCPP for DirectShow"), punk, phr);

    if (pNewObject == NULL)
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
    if (*mtIn->FormatType() != FORMAT_VideoInfo && *mtIn->FormatType() != FORMAT_VideoInfo2)
        return E_INVALIDARG;

    //Resolution check
    if (mtIn->formattype == FORMAT_VideoInfo2)
    {
        VIDEOINFOHEADER2* pVi = (VIDEOINFOHEADER2*)mtIn->pbFormat;
        if (H < abs(pVi->bmiHeader.biHeight) || W < abs(pVi->bmiHeader.biWidth))
            return E_INVALIDARG;
    }
    else
    {
        VIDEOINFOHEADER* pVi = (VIDEOINFOHEADER*)mtIn->pbFormat;
        if (H < abs(pVi->bmiHeader.biHeight) || W < abs(pVi->bmiHeader.biWidth))
            return E_INVALIDARG;
    }

    // Can we transform this type
    if (IsYV12(mtIn) || IsIYUV(mtIn) || IsNV12(mtIn) || IsRGB24(mtIn) || IsRGB32(mtIn))
        return NOERROR;

    return E_FAIL;
}

HRESULT Anime4KCPPDS::CheckTransform(const CMediaType* mtIn, const CMediaType* mtOut)
{
    CheckPointer(mtIn, E_POINTER);
    CheckPointer(mtOut, E_POINTER);

    // init GPU
    if (CNN)
    {
        if (!Anime4KCPP::Anime4KGPUCNN::isInitializedGPU())
            Anime4KCPP::Anime4KGPUCNN::initGPU(pID, dID);
    }
    else
    {
        if (!Anime4KCPP::Anime4KGPU::isInitializedGPU())
            Anime4KCPP::Anime4KGPU::initGPU(pID, dID);
    }

    if (CNN && IsYV12(mtIn) && IsYV12(mtOut))
    {
        colorFormat = ColorFormat::YV12;
        return NOERROR;
    }
    if (CNN && IsIYUV(mtIn) && IsIYUV(mtOut))
    {
        colorFormat = ColorFormat::IYUV;
        return NOERROR;
    }
    if (CNN && IsNV12(mtIn) && IsNV12(mtOut))
    {
        colorFormat = ColorFormat::NV12;
        return NOERROR;
    }
    if (IsRGB24(mtIn) && IsRGB24(mtOut))
    {
        colorFormat = ColorFormat::RGB24;
        return NOERROR;
    }
    if (IsRGB32(mtIn) && IsRGB32(mtOut))
    {
        colorFormat = ColorFormat::RGB32;
        return NOERROR;
    }

    return E_FAIL;
}

HRESULT Anime4KCPPDS::DecideBufferSize(IMemAllocator* pAlloc, ALLOCATOR_PROPERTIES* pProperties)
{
    // Is the input pin connecte
    if (m_pInput->IsConnected() == FALSE)
        return E_UNEXPECTED;

    CheckPointer(pAlloc, E_POINTER);
    CheckPointer(pProperties, E_POINTER);
    HRESULT hr = NOERROR;

    pProperties->cBuffers = 1;
    pProperties->cbBuffer = dstDataLength;

    ALLOCATOR_PROPERTIES Actual;
    hr = pAlloc->SetProperties(pProperties, &Actual);
    if (FAILED(hr))
        return hr;

    if (pProperties->cBuffers > Actual.cBuffers || pProperties->cbBuffer > Actual.cbBuffer)
        return E_FAIL;

    return NOERROR;
}

HRESULT Anime4KCPPDS::GetMediaType(int iPosition, CMediaType* pMediaType)
{
    if (m_pInput->IsConnected() == FALSE)
        return E_UNEXPECTED;

    if (iPosition < 0)
        return E_INVALIDARG;

    if (iPosition > 0)
        return VFW_S_NO_MORE_ITEMS;

    CheckPointer(pMediaType, E_POINTER);
    *pMediaType = m_pInput->CurrentMediaType();
    //resize
    if (pMediaType->formattype == FORMAT_VideoInfo2)
    {
        VIDEOINFOHEADER2* pVi = (VIDEOINFOHEADER2*)pMediaType->pbFormat;
        srcH = pVi->bmiHeader.biHeight;
        srcW = pVi->bmiHeader.biWidth;
        dstH = srcH * zf;
        dstW = srcW * zf;
        dstDataLength = static_cast<size_t>(pVi->bmiHeader.biSizeImage) * static_cast<size_t>(zf) * static_cast<size_t>(zf);
        pVi->bmiHeader.biHeight = dstH;
        pVi->bmiHeader.biWidth = dstW;
        pVi->bmiHeader.biSizeImage = dstDataLength;
        pMediaType->SetSampleSize(dstDataLength);
        SetRectEmpty(&pVi->rcSource);
        SetRectEmpty(&pVi->rcTarget);
    }
    else
    {
        VIDEOINFOHEADER* pVi = (VIDEOINFOHEADER*)pMediaType->pbFormat;
        srcH = pVi->bmiHeader.biHeight;
        srcW = pVi->bmiHeader.biWidth;
        dstH = srcH * zf;
        dstW = srcW * zf;
        dstDataLength = static_cast<size_t>(pVi->bmiHeader.biSizeImage) * static_cast<size_t>(zf) * static_cast<size_t>(zf);
        pVi->bmiHeader.biHeight = dstH;
        pVi->bmiHeader.biWidth = dstW;
        pVi->bmiHeader.biSizeImage = dstDataLength;
        pMediaType->SetSampleSize(dstDataLength);
        SetRectEmpty(&pVi->rcSource);
        SetRectEmpty(&pVi->rcTarget);
    }

    return NOERROR;
}

HRESULT Anime4KCPPDS::Transform(IMediaSample* pIn, IMediaSample* pOut)
{
    HRESULT hr = NOERROR;

    BYTE* pBufferIn, * pBufferOut;
    hr = pIn->GetPointer(&pBufferIn);
    if (FAILED(hr))
        return hr;
    hr = pOut->GetPointer(&pBufferOut);
    if (FAILED(hr))
        return hr;

    Anime4KCPP::Anime4K* ac = nullptr;
    if (CNN)
        ac = acCreator.create(parameters, Anime4KCPP::ProcessorType::GPUCNN);
    else
        ac = acCreator.create(parameters, Anime4KCPP::ProcessorType::GPU);

    switch (colorFormat)
    {
    case ColorFormat::YV12:
    case ColorFormat::IYUV:
    {
        LONG srcSize = (pIn->GetActualDataLength() << 1) / 3;
        LONG dstSize = (pOut->GetActualDataLength() << 1) / 3;
        LONG stride = dstSize / dstH;

        BYTE* pYIn = pBufferIn,
            * pUIn = pYIn + srcSize,
            * pVIn = pUIn + (srcSize >> 2);
        BYTE* pYOut = pBufferOut,
            * pUOut = pYOut + dstSize,
            * pVOut = pUOut + (dstSize >> 2);

        if (stride == dstW)
        {
            ac->loadImage(srcH, srcW, pYIn, srcH >> 1, srcW >> 1, pUIn, srcH >> 1, srcW >> 1, pVIn);
            ac->process();
            ac->saveImage(pYOut, pUOut, pVOut);
        }
        else
        {
            LONG dstHUV = dstH >> 1;
            LONG dstWUV = dstW >> 1;
            LONG strideUV = stride >> 1;
            cv::Mat dstTmpY, dstTmpU, dstTmpV;

            ac->loadImage(srcH, srcW, pYIn, srcH >> 1, srcW >> 1, pUIn, srcH >> 1, srcW >> 1, pVIn);
            ac->process();
            ac->saveImage(dstTmpY, dstTmpU, dstTmpV);

            for (LONG y = 0; y < dstH; y++)
            {
                memcpy(pYOut, dstTmpY.data + static_cast<size_t>(y) * static_cast<size_t>(dstW), dstW);
                pYOut += stride;
                if (y < dstHUV)
                {
                    memcpy(pUOut, dstTmpU.data + static_cast<size_t>(y) * static_cast<size_t>(dstWUV), dstWUV);
                    pUOut += strideUV;
                    memcpy(pVOut, dstTmpV.data + static_cast<size_t>(y) * static_cast<size_t>(dstWUV), dstWUV);
                    pVOut += strideUV;
                }
            }
        }
    }
    break;
    case ColorFormat::NV12:
    {
        LONG srcSize = (pIn->GetActualDataLength() << 1) / 3;
        LONG dstSize = (pOut->GetActualDataLength() << 1) / 3;
        LONG stride = dstSize / dstH;

        BYTE* pYIn = pBufferIn,
            * pUVIn = pYIn + srcSize;
        BYTE* pYOut = pBufferOut,
            * pUVOut = pYOut + dstSize;

        cv::Mat dstTmpY, dstTmpU, dstTmpV;
        cv::Mat dstUV;

        cv::Mat srcUV(srcH >> 1, srcW >> 1, CV_8UC2, pUVIn);
        std::vector<cv::Mat> uv(2);
        cv::split(srcUV, uv);
        BYTE* pUIn = uv[0].data;
        BYTE* pVIn = uv[1].data;

        if (stride == dstW)
        {
            ac->loadImage(srcH, srcW, pYIn, srcH >> 1, srcW >> 1, pUIn, srcH >> 1, srcW >> 1, pVIn);
            ac->process();
            ac->saveImage(dstTmpY, dstTmpU, dstTmpV);
            cv::merge(std::vector<cv::Mat>{dstTmpU, dstTmpV}, dstUV);
            memcpy(pYOut, dstTmpY.data, dstSize);
            memcpy(pUVOut, dstUV.data, dstSize >> 1);
        }
        else
        {
            LONG dstHUV = dstH >> 1;

            ac->loadImage(srcH, srcW, pYIn, srcH >> 1, srcW >> 1, pUIn, srcH >> 1, srcW >> 1, pVIn);
            ac->process();
            ac->saveImage(dstTmpY, dstTmpU, dstTmpV);

            cv::merge(std::vector<cv::Mat>{dstTmpU, dstTmpV}, dstUV);

            for (LONG y = 0; y < dstH; y++)
            {
                memcpy(pYOut, dstTmpY.data + static_cast<size_t>(y) * static_cast<size_t>(dstW), dstW);
                pYOut += stride;
                if (y < dstHUV)
                {
                    memcpy(pUVOut, dstUV.data + static_cast<size_t>(y) * static_cast<size_t>(dstW), dstW);
                    pUVOut += stride;
                }
            }
        }
    }
    break;
    case ColorFormat::RGB24:
    {
        LONG stride = pOut->GetActualDataLength() / dstH;
        LONG dataPerLine = dstW * 3;
        if (stride == dataPerLine)
        {
            ac->loadImage(srcH, srcW, pBufferIn);
            ac->process();
            ac->saveImage(pBufferOut);
        }
        else
        {
            BYTE* dstRGB = pBufferOut;
            cv::Mat dstTmp;
            ac->loadImage(srcH, srcW, pBufferIn);
            ac->process();
            ac->saveImage(dstTmp);
            for (size_t y = 0; y < dstH; y++)
            {
                memcpy(dstRGB, dstTmp.data + y * dataPerLine, dataPerLine);
                dstRGB += stride;
            }
        }
    }
    break;
    case ColorFormat::RGB32:
    {
        LONG stride = pOut->GetActualDataLength() / dstH;
        LONG dataPerLine = dstW << 2;
        if (stride == dataPerLine)
        {
            cv::Mat srcTmp(srcH, srcW, CV_8UC4, pBufferIn);
            cv::flip(srcTmp, srcTmp, 0);
            ac->loadImage(srcH, srcW, srcTmp.data, 0Ui64, false, true);
            ac->process();
            ac->saveImage(pBufferOut);
        }
        else
        {
            BYTE* dstRGBA = pBufferOut;
            cv::Mat dstTmp;
            cv::Mat srcTmp(srcH, srcW, CV_8UC4, pBufferIn);
            cv::flip(srcTmp, srcTmp, 0); // upside down ???
            ac->loadImage(srcH, srcW, srcTmp.data, 0Ui64, false, true);
            ac->process();
            ac->saveImage(dstTmp);
            for (size_t y = 0; y < dstH; y++)
            {
                memcpy(dstRGBA, dstTmp.data + y * dataPerLine, dataPerLine);
                dstRGBA += stride;
            }
        }
    }
    break;
    }

    acCreator.release(ac);

    return hr;
}

STDMETHODIMP Anime4KCPPDS::GetParameters(bool* HDN, bool* CNN, unsigned int* pID, unsigned int* dID, float* zoomFactor, int* H, int* W)
{
    *HDN = parameters.HDN;
    *CNN = this->CNN;
    *pID = this->pID;
    *dID = this->dID;
    *H = this->H;
    *W = this->W;
    *zoomFactor = zf;

    return NOERROR;
}

STDMETHODIMP Anime4KCPPDS::SetParameters(bool HDN, bool CNN, unsigned int pID, unsigned int dID, float zoomFactor, int H, int W)
{
    CAutoLock cAutoLock(&lock);

    TCHAR _pID[10], _dID[10], _CNN[10], _HDN[10], _H[10], _W[10], _zoomFactor[10];

    //convert to string
    _itow_s(pID, _pID, 10, 10);
    _itow_s(dID, _dID, 10, 10);
    _itow_s(CNN, _CNN, 10, 10);
    _itow_s(H, _H, 10, 10);
    _itow_s(W, _W, 10, 10);
    _itow_s(HDN, _HDN, 10, 10);
    swprintf(_zoomFactor, 10, L"%f", zoomFactor);

    //write config
    WritePrivateProfileString(L"Anime4KCPP for DirectShow Config", L"pID", _pID, lpPath);
    WritePrivateProfileString(L"Anime4KCPP for DirectShow Config", L"dID", _dID, lpPath);
    WritePrivateProfileString(L"Anime4KCPP for DirectShow Config", L"ACNet", _CNN, lpPath);
    WritePrivateProfileString(L"Anime4KCPP for DirectShow Config", L"HDN", _HDN, lpPath);
    WritePrivateProfileString(L"Anime4KCPP for DirectShow Config", L"H", _H, lpPath);
    WritePrivateProfileString(L"Anime4KCPP for DirectShow Config", L"W", _W, lpPath);
    WritePrivateProfileString(L"Anime4KCPP for DirectShow Config", L"zoomFactor", _zoomFactor, lpPath);

    return NOERROR;
}

STDMETHODIMP Anime4KCPPDS::GetGPUInfo(std::string& info)
{
    auto GPUInfo = Anime4KCPP::Anime4KGPU::listGPUs();

    std::string tmpStr = GPUInfo.second;
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
