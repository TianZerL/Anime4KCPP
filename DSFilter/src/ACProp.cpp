#include "ACProp.h"

CUnknown* ACProp::CreateInstance(LPUNKNOWN lpunk, HRESULT* phr)
{
    CUnknown* punk = new ACProp(lpunk, phr);

    if (punk == NULL)
    {
        if (phr)
            *phr = E_OUTOFMEMORY;
    }

    return punk;
}

INT_PTR ACProp::OnReceiveMessage(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
    case WM_COMMAND:
    {
        if (bInit)
        {
            m_bDirty = TRUE;
            if (m_pPageSite)
                m_pPageSite->OnStatusChange(PROPPAGESTATUS_DIRTY);
        }
        return (LRESULT)1;
    }
    }

    return CBasePropertyPage::OnReceiveMessage(hwnd, uMsg, wParam, lParam);
}

HRESULT ACProp::OnConnect(IUnknown* pUnk)
{
    CheckPointer(pUnk, E_POINTER);

    HRESULT hr = pUnk->QueryInterface(IID_IAC, (void**)&pIAC);
    if (FAILED(hr))
        return E_NOINTERFACE;

    CheckPointer(pIAC, E_FAIL);
    pIAC->GetParameters(&HDN, &CNN, &pID, &dID, &zoomFactor, &H, &W);

    pIAC->GetGPUInfo(GPUInfo);

    bInit = FALSE;

    return NOERROR;
}

HRESULT ACProp::OnDisconnect()
{
    if (pIAC)
    {
        pIAC->Release();
        pIAC = NULL;
    }

    return NOERROR;
}

HRESULT ACProp::OnActivate()
{
    Button_SetCheck(GetDlgItem(m_Dlg, IDC_CHECK_HDN), HDN);
    Button_SetCheck(GetDlgItem(m_Dlg, IDC_CHECK_CNN), CNN);

    TCHAR sz[STR_MAX_LENGTH];

    StringCchPrintf(sz, NUMELMS(sz), TEXT("%f\0"), zoomFactor);
    Edit_SetText(GetDlgItem(m_Dlg, IDC_EDIT_ZF), sz);

    StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), pID);
    Edit_SetText(GetDlgItem(m_Dlg, IDC_EDIT_PID), sz);

    StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), dID);
    Edit_SetText(GetDlgItem(m_Dlg, IDC_EDIT_DID), sz);

    StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), H);
    Edit_SetText(GetDlgItem(m_Dlg, IDC_EDIT_H), sz);

    StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), W);
    Edit_SetText(GetDlgItem(m_Dlg, IDC_EDIT_W), sz);

    MultiByteToWideChar(CP_ACP, 0, GPUInfo.c_str(), -1, sz, STR_MAX_LENGTH);
    Edit_SetText(GetDlgItem(m_Dlg, IDC_EDIT_GPUINFO), sz);

    MultiByteToWideChar(CP_ACP, 0, ANIME4KCPP_CORE_VERSION, -1, sz, STR_MAX_LENGTH);
    Edit_SetText(GetDlgItem(m_Dlg, IDC_EDIT_VERSION), sz);

    bInit = TRUE;

    return NOERROR;
}

HRESULT ACProp::OnDeactivate()
{
    bInit = FALSE;
    GetValues();

    return NOERROR;
}

HRESULT ACProp::OnApplyChanges()
{
    GetValues();
    CheckPointer(pIAC, E_POINTER);
    pIAC->SetParameters(HDN, CNN, pID, dID, zoomFactor, H, W);

    return NOERROR;
}

void ACProp::GetValues()
{
    HDN = Button_GetCheck(GetDlgItem(m_Dlg, IDC_CHECK_HDN)) == BST_CHECKED;

    CNN = Button_GetCheck(GetDlgItem(m_Dlg, IDC_CHECK_CNN)) == BST_CHECKED;

    TCHAR sz[STR_MAX_LENGTH];
    Edit_GetText(GetDlgItem(m_Dlg, IDC_EDIT_ZF), sz, STR_MAX_LENGTH);
    zoomFactor = _wtof(sz);
    zoomFactor = zoomFactor >= 1.0F ? zoomFactor : 1.0F;

    Edit_GetText(GetDlgItem(m_Dlg, IDC_EDIT_PID), sz, STR_MAX_LENGTH);
    pID = _wtoi(sz);
    pID = pID < 0 ? 0 : pID;

    Edit_GetText(GetDlgItem(m_Dlg, IDC_EDIT_DID), sz, STR_MAX_LENGTH);
    dID = _wtoi(sz);
    dID = dID < 0 ? 0 : dID;

    Edit_GetText(GetDlgItem(m_Dlg, IDC_EDIT_H), sz, STR_MAX_LENGTH);
    H = _wtoi(sz);
    H = H < 0 ? 0 : H;

    Edit_GetText(GetDlgItem(m_Dlg, IDC_EDIT_W), sz, STR_MAX_LENGTH);
    W = _wtoi(sz);
    W = W < 0 ? 0 : W;
}

ACProp::ACProp(LPUNKNOWN lpunk, HRESULT* phr) :
    CBasePropertyPage(NAME("Anime4KCPP for DirectShow Property Page"), lpunk,
        IDD_ACPROP, IDS_TITLE),
    HDN(false), CNN(false), pID(0), dID(0), zoomFactor(2.0), H(1080), W(1920),
    pIAC(NULL), bInit(FALSE) {}
