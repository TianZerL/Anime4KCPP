#pragma once

#include <string>
#include <streams.h>
#include <strsafe.h>

#include "Anime4KCPP.hpp"
#include "resource.h"

namespace GPGPU
{
    static constexpr int CPU = 0, OpenCL = 1, CUDA = 2;
};

struct ACPropData
{
    bool CNN = true;
    bool HDN = false;
    int HDNLevel = 1;
    int pID = 0;
    int dID = 0;
    double zoomFactor = 2.0;
    int H = 1080;
    int W = 1920;
    int GPGPUModel = GPGPU::OpenCL;
    int OpenCLQueueNum = 4;
    bool OpenCLParallelIO = false;
};

extern "C"
{
    // {82F56FD7-717C-46CA-94C4-C1DBD380BCA4}
    DEFINE_GUID(IID_IAC,
        0x82f56fd7, 0x717c, 0x46ca, 0x94, 0xc4, 0xc1, 0xdb, 0xd3, 0x80, 0xbc, 0xa4);

    DECLARE_INTERFACE_(IAC, IUnknown)
    {
        STDMETHOD(GetParameters) (THIS_
            ACPropData& data
            ) PURE;

        STDMETHOD(SetParameters) (THIS_
            const ACPropData & data
            ) PURE;

        STDMETHOD(GetProcessorInfo) (THIS_
            std::string& info
            ) PURE;
    };
}

class ACProp :
    public CBasePropertyPage
{
public:
    static CUnknown* WINAPI CreateInstance(LPUNKNOWN lpunk, HRESULT* phr);
private:
    virtual INT_PTR OnReceiveMessage(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    virtual HRESULT OnConnect(IUnknown* pUnk);
    virtual HRESULT OnDisconnect();
    virtual HRESULT OnActivate();
    virtual HRESULT OnDeactivate();
    virtual HRESULT OnApplyChanges();

    void GetValues();

private:
    ACProp(LPUNKNOWN lpunk, HRESULT* phr);
private:
    ACPropData data;
    std::string ProcessorInfo;

    IAC* pIAC;
    BOOL bInit;
};
