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
    pIAC->GetParameters(data);

    pIAC->GetProcessorInfo(ProcessorInfo);

    bInit = FALSE;

    return S_OK;
}

HRESULT ACProp::OnDisconnect()
{
    if (pIAC)
    {
        pIAC->Release();
        pIAC = NULL;
    }

    return S_OK;
}

HRESULT ACProp::OnActivate()
{
    Button_SetCheck(GetDlgItem(m_Dlg, IDC_CHECK_HDN), data.HDN);
    Button_SetCheck(GetDlgItem(m_Dlg, IDC_CHECK_CNN), data.CNN);
    Button_SetCheck(GetDlgItem(m_Dlg, IDC_CHECK_OpenCLParallelIO), data.OpenCLParallelIO);

    ComboBox_AddString(GetDlgItem(m_Dlg, IDC_COMBO_GPGPU), L"CPU");
    ComboBox_AddString(GetDlgItem(m_Dlg, IDC_COMBO_GPGPU), L"OpenCL");
    ComboBox_AddString(GetDlgItem(m_Dlg, IDC_COMBO_GPGPU), L"CUDA");
    ComboBox_SetCurSel(GetDlgItem(m_Dlg, IDC_COMBO_GPGPU), data.GPGPUModel);

    TCHAR sz[STR_MAX_LENGTH];

    StringCchPrintf(sz, NUMELMS(sz), TEXT("%lf\0"), data.zoomFactor);
    Edit_SetText(GetDlgItem(m_Dlg, IDC_EDIT_ZF), sz);

    StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), data.HDNLevel);
    Edit_SetText(GetDlgItem(m_Dlg, IDC_EDIT_HDNLevel), sz);

    StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), data.pID);
    Edit_SetText(GetDlgItem(m_Dlg, IDC_EDIT_PID), sz);

    StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), data.dID);
    Edit_SetText(GetDlgItem(m_Dlg, IDC_EDIT_DID), sz);

    StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), data.OpenCLQueueNum);
    Edit_SetText(GetDlgItem(m_Dlg, IDC_EDIT_OpenCLQueueNum), sz);

    StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), data.H);
    Edit_SetText(GetDlgItem(m_Dlg, IDC_EDIT_H), sz);

    StringCchPrintf(sz, NUMELMS(sz), TEXT("%d\0"), data.W);
    Edit_SetText(GetDlgItem(m_Dlg, IDC_EDIT_W), sz);

    MultiByteToWideChar(CP_ACP, 0, ProcessorInfo.c_str(), -1, sz, STR_MAX_LENGTH);
    Edit_SetText(GetDlgItem(m_Dlg, IDC_EDIT_GPUINFO), sz);

    MultiByteToWideChar(CP_ACP, 0, Anime4KCPP::CoreInfo::version(), -1, sz, STR_MAX_LENGTH);
    Edit_SetText(GetDlgItem(m_Dlg, IDC_EDIT_VERSION), sz);

    bInit = TRUE;

    return S_OK;
}

HRESULT ACProp::OnDeactivate()
{
    bInit = FALSE;
    GetValues();

    return S_OK;
}

HRESULT ACProp::OnApplyChanges()
{
    GetValues();
    CheckPointer(pIAC, E_POINTER);
    pIAC->SetParameters(data);

    return S_OK;
}

void ACProp::GetValues()
{
    data.HDN = Button_GetCheck(GetDlgItem(m_Dlg, IDC_CHECK_HDN)) == BST_CHECKED;

    data.CNN = Button_GetCheck(GetDlgItem(m_Dlg, IDC_CHECK_CNN)) == BST_CHECKED;

    data.OpenCLParallelIO = Button_GetCheck(GetDlgItem(m_Dlg, IDC_CHECK_OpenCLParallelIO)) == BST_CHECKED;

    data.GPGPUModel = ComboBox_GetCurSel(GetDlgItem(m_Dlg, IDC_COMBO_GPGPU));

    TCHAR sz[STR_MAX_LENGTH];
    Edit_GetText(GetDlgItem(m_Dlg, IDC_EDIT_ZF), sz, STR_MAX_LENGTH);
    data.zoomFactor = _wtof(sz);
    data.zoomFactor = data.zoomFactor >= 1.0 ? data.zoomFactor : 1.0;

    Edit_GetText(GetDlgItem(m_Dlg, IDC_EDIT_HDNLevel), sz, STR_MAX_LENGTH);
    data.HDNLevel = _wtoi(sz);
    data.HDNLevel = (data.HDNLevel < 0 || data.HDNLevel > 3) ? 1 : data.HDNLevel;

    Edit_GetText(GetDlgItem(m_Dlg, IDC_EDIT_PID), sz, STR_MAX_LENGTH);
    data.pID = _wtoi(sz);
    data.pID = data.pID < 0 ? 0 : data.pID;

    Edit_GetText(GetDlgItem(m_Dlg, IDC_EDIT_DID), sz, STR_MAX_LENGTH);
    data.dID = _wtoi(sz);
    data.dID = data.dID < 0 ? 0 : data.dID;

    Edit_GetText(GetDlgItem(m_Dlg, IDC_EDIT_OpenCLQueueNum), sz, STR_MAX_LENGTH);
    data.OpenCLQueueNum = _wtoi(sz);
    data.OpenCLQueueNum = data.OpenCLQueueNum < 1 ? 1 : data.OpenCLQueueNum;

    Edit_GetText(GetDlgItem(m_Dlg, IDC_EDIT_H), sz, STR_MAX_LENGTH);
    data.H = _wtoi(sz);
    data.H = data.H < 0 ? 0 : data.H;

    Edit_GetText(GetDlgItem(m_Dlg, IDC_EDIT_W), sz, STR_MAX_LENGTH);
    data.W = _wtoi(sz);
    data.W = data.W < 0 ? 0 : data.W;
}

ACProp::ACProp(LPUNKNOWN lpunk, HRESULT* phr) :
    CBasePropertyPage(NAME("Anime4KCPP for DirectShow Property Page"), lpunk,
        IDD_ACPROP, IDS_TITLE), data{}, pIAC(NULL), bInit(FALSE) {}
