#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <unordered_map>

#include <malloc.h>

#include "SideData.hpp"

class CSideDataMediaSample : public CMediaSample, public IMediaSideData
{
public:
    using CMediaSample::CMediaSample;
    ~CSideDataMediaSample() noexcept override;

    STDMETHODIMP QueryInterface(REFIID riid, void** ppv) override;
    STDMETHODIMP_(ULONG) AddRef() override;
    STDMETHODIMP_(ULONG) Release() override;

    STDMETHODIMP SetSideData(GUID guidType, const BYTE* data, std::size_t size) override;
    STDMETHODIMP GetSideData(GUID guidType, const BYTE** data, std::size_t* size) override;

private:
    void ReleaseSideData() noexcept;

private:
    struct SideData
    {
        BYTE* data;
        std::size_t size;
    };

    struct GUIDHasher
    {
        std::size_t operator()(const GUID& guid) const noexcept
        {
            std::uint64_t lo{};
            std::uint64_t hi{};

            std::memcpy(&lo, &guid, 8);
            std::memcpy(&hi, guid.Data4, 8);

            std::hash<std::uint64_t> hash{};
            auto h1 = hash(hi);
            auto h2 = hash(lo);

            return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
        }
    };

private:
    CCritSec mtx{};
    std::unordered_map<GUID, SideData, GUIDHasher> sideDataMap{};
};

class CSideDataAllocator : public CMemAllocator
{
public:
    using CMemAllocator::CMemAllocator;
protected:
    HRESULT Alloc() override;
};

CSideDataMediaSample::~CSideDataMediaSample() noexcept
{
    ReleaseSideData();
}
STDMETHODIMP CSideDataMediaSample::QueryInterface(REFIID riid, void** const ppv)
{
    CheckPointer(ppv, E_POINTER);

    if (riid == CLSID_IMediaSideData)
        return GetInterface(static_cast<IMediaSideData*>(this), ppv);
    else
        return CMediaSample::QueryInterface(riid, ppv);
}
STDMETHODIMP_(ULONG) CSideDataMediaSample::AddRef()
{
    return CMediaSample::AddRef();
}
STDMETHODIMP_(ULONG) CSideDataMediaSample::Release()
{
    ULONG refCount = InterlockedDecrement(&m_cRef);

    if (refCount == 0)
    {
        if (m_dwFlags & Sample_TypeChanged) SetMediaType(nullptr);

        m_dwFlags = 0;
        m_dwTypeSpecificFlags = 0;
        m_dwStreamId = AM_STREAM_MEDIA;

        ReleaseSideData();

        m_pAllocator->ReleaseBuffer(this);
    }

    return refCount;
}
STDMETHODIMP CSideDataMediaSample::SetSideData(const GUID guidType, const BYTE* const data, const std::size_t size)
{
    if (!data || !size) return E_POINTER;

    CAutoLock lock(&mtx);

    auto it = sideDataMap.find(guidType);
    if (it == sideDataMap.end())
    {
        SideData sideData{};
        sideData.data = static_cast<BYTE*>(_aligned_malloc(size, 16));
        if (sideData.data) sideData.size = size;
        else return E_OUTOFMEMORY;

        it = sideDataMap.emplace(guidType, sideData).first;
    }
    else
    {
        SideData& sd = it->second;
        if (size != sd.size)
        {
            auto buffer = static_cast<BYTE*>(_aligned_realloc(sd.data, size, 16));
            if (buffer)
            {
                sd.data = buffer;
                sd.size = size;
            }
            else return E_OUTOFMEMORY;
        }
    }
    std::memcpy(it->second.data, data, size);

    return S_OK;
}
STDMETHODIMP CSideDataMediaSample::GetSideData(const GUID guidType, const BYTE** const data, std::size_t* const size)
{
    if (!data || !size) return E_POINTER;

    CAutoLock lock(&mtx);

    auto it = sideDataMap.find(guidType);
    if (it == sideDataMap.end()) return E_FAIL;

    *data = it->second.data;
    *size = it->second.size;

    return S_OK;
}
void CSideDataMediaSample::ReleaseSideData() noexcept
{
    CAutoLock lock(&mtx);

    for (auto&& [guid, sd] : sideDataMap) _aligned_free(sd.data);

    sideDataMap.clear();
}

HRESULT CSideDataAllocator::Alloc()
{
    CAutoLock lock(this);

    HRESULT hr = S_OK;

    hr = CBaseAllocator::Alloc();
    if (FAILED(hr)) return hr;
    if (hr == S_FALSE) return S_OK;

    if (m_pBuffer) ReallyFree();
    if (m_lSize < 0 || m_lPrefix < 0 || m_lCount < 0) return E_OUTOFMEMORY;

    auto alignedSize = (m_lSize + m_lPrefix + m_lAlignment - 1) & -m_lAlignment;
    auto totalSize = m_lCount * static_cast<std::size_t>(alignedSize);
    if (totalSize > MAXLONG) return E_OUTOFMEMORY;

    m_pBuffer = static_cast<decltype(m_pBuffer)>(VirtualAlloc(nullptr, totalSize, MEM_COMMIT, PAGE_READWRITE));
    if (!m_pBuffer) return E_OUTOFMEMORY;

    for (auto next = m_pBuffer; m_lAllocated < m_lCount; m_lAllocated++, next += alignedSize)
    {
        auto sample = std::unique_ptr<CSideDataMediaSample>{ new(std::nothrow) CSideDataMediaSample{ NAME("Sidedata media sample"), this, &hr, next + m_lPrefix, m_lSize} };
        if (!sample) return E_OUTOFMEMORY;
        if (FAILED(hr)) return E_UNEXPECTED;

        m_lFree.Add(sample.release());
    }

    m_bChanged = FALSE;
    return S_OK;
}

STDMETHODIMP CSideDataInputPin::ReceiveConnection(IPin* const connector, const AM_MEDIA_TYPE* const amt)
{
    CheckPointer(connector, E_POINTER);
    CheckPointer(amt, E_POINTER);

    if (m_Connected)
    {
        CAutoLock lock{ m_pLock };

        HRESULT hr = S_OK;

        CMediaType mtIn{ *amt };
        hr = CheckMediaType(&mtIn); if (FAILED(hr)) return VFW_E_TYPE_NOT_ACCEPTED;
        hr = SetMediaType(&mtIn); if (FAILED(hr)) return VFW_E_TYPE_NOT_ACCEPTED;

        auto outputPin = static_cast<CBaseOutputPin*>(m_pTransformFilter->GetPin(1));
        if (outputPin && outputPin->IsConnected())
        {
            int timeout = 100;
            CMediaType mtOut{};
            hr = outputPin->GetMediaType(0, &mtOut); if (FAILED(hr)) return VFW_E_TYPE_NOT_ACCEPTED;
            for (;;)
            {
                hr = outputPin->GetConnected()->ReceiveConnection(outputPin, &mtOut);
                if (hr != VFW_E_BUFFERS_OUTSTANDING || timeout < 0) break;
                if (timeout)
                {
                    Sleep(10);
                    timeout -= 10;
                }
                else
                {
                    outputPin->DeliverBeginFlush();
                    outputPin->DeliverEndFlush();
                    timeout -= 1;
                }
            }
            if (SUCCEEDED(hr)) hr = outputPin->SetMediaType(&mtOut);
        }
        return hr;
    }

    return CTransformInputPin::ReceiveConnection(connector, amt);
}
STDMETHODIMP CSideDataInputPin::GetAllocator(IMemAllocator** const allocator)
{
    CheckPointer(allocator, E_POINTER);

    CAutoLock lock(m_pLock);

    if (!m_pAllocator)
    {
        HRESULT hr = S_OK;
        auto allocatorPtr = std::unique_ptr<CSideDataAllocator>{ new(std::nothrow) CSideDataAllocator{NAME("CSideDataAllocator"), nullptr, &hr} };
        if (!allocatorPtr) return E_OUTOFMEMORY;
        if (FAILED(hr)) return hr;

        allocatorPtr->AddRef();
        m_pAllocator = allocatorPtr.release();
    }

    *allocator = m_pAllocator;
    m_pAllocator->AddRef();

    return S_OK;
}
