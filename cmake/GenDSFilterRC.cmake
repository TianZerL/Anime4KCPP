# Generate some files required for VS
file(WRITE "${FILTER_DIRECTSHOW_BINARY_DIR}/include/resource.hpp" [[
//{{NO_DEPENDENCIES}}
#define IDD_PROPPAGE                    102
#define IDS_TITLE                       103
#define IDC_COMBO_PROCESSOR             1002
#define IDC_COMBO_MODEL                 1003
#define IDC_EDIT_DEVICE                 1004
#define IDC_EDIT_FACTOR                 1005
#define IDC_EDIT_INFO                   1006
#define IDC_EDIT_LIMIT_WIDTH            1007
#define IDC_EDIT_LIMIT_HEIGHT           1008
#define IDC_STATIC_COPYRIGHT            1009
#define IDC_STATIC_DEVICE               1010
#define IDC_STATIC_FACTOR               1011
#define IDC_STATIC_LIMIT                1012
#define IDC_STATIC_LIMIT_X              1013
#define IDC_STATIC_MODEL                1014
#define IDC_STATIC_PROCRSSOR            1015
#define IDC_STATIC_TITLE                1016
#define IDC_STATIC_VERSION              1017
#ifdef APSTUDIO_INVOKED
#ifndef APSTUDIO_READONLY_SYMBOLS
#define _APS_NEXT_RESOURCE_VALUE        104
#define _APS_NEXT_COMMAND_VALUE         40001
#define _APS_NEXT_CONTROL_VALUE         1018
#define _APS_NEXT_SYMED_VALUE           101
#endif
#endif
]])

file(WRITE "${FILTER_DIRECTSHOW_BINARY_DIR}/rc/Filter.rc" [[
#include "resource.h"
#define APSTUDIO_READONLY_SYMBOLS
#include "winres.h"
#undef APSTUDIO_READONLY_SYMBOLS
#if !defined(AFX_RESOURCE_DLL) || defined(AFX_TARG_ENU)
LANGUAGE LANG_ENGLISH, SUBLANG_ENGLISH_US
#ifdef APSTUDIO_INVOKED
1 TEXTINCLUDE
BEGIN
    "resource.h\0"
END
2 TEXTINCLUDE
BEGIN
    "#include ""winres.h""\r\n"
    "\0"
END
3 TEXTINCLUDE
BEGIN
    "\r\n"
    "\0"
END
#endif
IDD_PROPPAGE DIALOGEX 0, 0, 215, 190
STYLE DS_SETFONT | DS_FIXEDSYS | WS_CHILD
FONT 9, "MS Shell Dlg", 400, 0, 0x0
BEGIN
    LTEXT           "Anime4KCPP for DirectShow",IDC_STATIC_TITLE,10,8,113,8
    LTEXT           "factor",IDC_STATIC_FACTOR,10,25,30,8
    EDITTEXT        IDC_EDIT_FACTOR,45,23,65,14,ES_AUTOHSCROLL | ES_NUMBER
    LTEXT           "model",IDC_STATIC_MODEL,115,25,20,8
    COMBOBOX        IDC_COMBO_MODEL,140,23,65,14,CBS_DROPDOWNLIST | WS_VSCROLL | WS_TABSTOP
    LTEXT           "processor",IDC_STATIC_PROCRSSOR,10,45,30,8
    COMBOBOX        IDC_COMBO_PROCESSOR,45,44,65,30,CBS_DROPDOWNLIST | WS_VSCROLL | WS_TABSTOP
    LTEXT           "device",IDC_STATIC_DEVICE,115,45,20,8
    EDITTEXT        IDC_EDIT_DEVICE,140,43,65,14,ES_AUTOHSCROLL | ES_NUMBER
    LTEXT           "disable, if the input resolution is higher than",IDC_STATIC_LIMIT,10,65,150,8
    EDITTEXT        IDC_EDIT_LIMIT_WIDTH,10,76,40,14,ES_AUTOHSCROLL | ES_NUMBER
    LTEXT           "X",IDC_STATIC_LIMIT_X,56,79,8,8
    EDITTEXT        IDC_EDIT_LIMIT_HEIGHT,65,76,40,14,ES_AUTOHSCROLL | ES_NUMBER
    EDITTEXT        IDC_EDIT_INFO,10,94,195,66,ES_MULTILINE | ES_AUTOHSCROLL | ES_READONLY | WS_VSCROLL | WS_HSCROLL
    LTEXT           "",IDC_STATIC_COPYRIGHT,12,170,163,18
    LTEXT           "",IDC_STATIC_VERSION,104,8,71,8
END
#ifdef APSTUDIO_INVOKED
GUIDELINES DESIGNINFO
BEGIN
    IDD_PROPPAGE, DIALOG
    BEGIN
        BOTTOMMARGIN, 169
    END
END
#endif
IDD_PROPPAGE AFX_DIALOG_LAYOUT
BEGIN
    0
END
STRINGTABLE
BEGIN
    IDS_TITLE               "Settings"
END
#endif
#ifndef APSTUDIO_INVOKED
#endif
]])

file(WRITE "${FILTER_DIRECTSHOW_BINARY_DIR}/rc/Filter.def" [[
LIBRARY     ac_filter_ds.dll

EXPORTS
            DllMain                 PRIVATE
            DllGetClassObject       PRIVATE
            DllCanUnloadNow         PRIVATE
            DllRegisterServer       PRIVATE
            DllUnregisterServer     PRIVATE
]])
