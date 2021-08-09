#pragma once

#include <string>
#include <streams.h>
#include <strsafe.h>

#include "Anime4KCPP.hpp"
#include "resource.h"

extern "C"
{
    // {82F56FD7-717C-46CA-94C4-C1DBD380BCA4}
    DEFINE_GUID(IID_IAC,
        0x82f56fd7, 0x717c, 0x46ca, 0x94, 0xc4, 0xc1, 0xdb, 0xd3, 0x80, 0xbc, 0xa4);

    DECLARE_INTERFACE_(IAC, IUnknown)
    {
        STDMETHOD(GetParameters) (THIS_
            bool* HDN,
            int* HDNLevel,
            bool* CNN,
            unsigned int* pID,
            unsigned int* dID,
            double* zoomFactor,
            int* H,
            int* W,
            int* GPGPUModel,
            int* OpenCLQueueNum,
            bool* OpenCLParallelIO
            ) PURE;

        STDMETHOD(SetParameters) (THIS_
            bool HDN,
            int HDNLevel,
            bool CNN,
            unsigned int pID,
            unsigned int dID,
            double zoomFactor,
            int H,
            int W,
            int GPGPUModel,
            int OpenCLQueueNum,
            bool OpenCLParallelIO
            ) PURE;

        STDMETHOD(GetGPUInfo) (THIS_
            std::string & info
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
    bool HDN, CNN;
    int HDNLevel;
    unsigned int pID, dID;
    double zoomFactor;
    int H, W;
    std::string GPUInfo;
    int GPGPUModelIdx;
    int OpenCLQueueNum;
    bool OpenCLParallelIO;

    IAC* pIAC;
    BOOL bInit;
};
