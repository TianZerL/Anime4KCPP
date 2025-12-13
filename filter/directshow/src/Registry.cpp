#include "Registry.hpp"

RegArgument::RegArgument() noexcept
{
    RegCreateKeyEx(HKEY_CURRENT_USER, TEXT("Software\\Anime4KCPP\\DSFilter"), 0, nullptr,
        REG_OPTION_NON_VOLATILE, KEY_WRITE | KEY_READ, nullptr, &key, nullptr);
}
RegArgument::~RegArgument() noexcept
{
    RegCloseKey(key);
}
double RegArgument::getFactor() noexcept
{
    DWORD size = sizeof(factor);
    auto data = reinterpret_cast<LPBYTE>(&factor);
    if (ERROR_SUCCESS == RegQueryValueEx(key, FactorValueName, nullptr, nullptr, data, &size))
        return factor;
    else return FactorDefault;
}
int RegArgument::getDevice() noexcept
{
    DWORD size = sizeof(device);
    auto data = reinterpret_cast<LPBYTE>(&device);
    if (ERROR_SUCCESS == RegQueryValueEx(key, DeviceValueName, nullptr, nullptr, data, &size))
        return device;
    else return DeviceDefault;
}
int RegArgument::getLimitWidth() noexcept
{
    DWORD size = sizeof(limitWidth);
    auto data = reinterpret_cast<LPBYTE>(&limitWidth);
    if (ERROR_SUCCESS == RegQueryValueEx(key, LimitWidthValueName, nullptr, nullptr, data, &size))
        return limitWidth;
    else return LimitWidthDefault;
}
int RegArgument::getLimitHeight() noexcept
{
    DWORD size = sizeof(limitHeight);
    auto data = reinterpret_cast<LPBYTE>(&limitHeight);
    if (ERROR_SUCCESS == RegQueryValueEx(key, LimitHeightValueName, nullptr, nullptr, data, &size))
        return limitHeight;
    else return LimitHeightDefault;
}
const TCHAR* RegArgument::getProcessorType() noexcept
{
    DWORD size = ProcessorTypeMaxSize;
    auto data = reinterpret_cast<LPBYTE>(ProcessorType);
    if (ERROR_SUCCESS == RegQueryValueEx(key, ProcessorTypeValueName, nullptr, nullptr, data, &size))
    {
        ProcessorType[size - 1] = TEXT('\0');
        return ProcessorType;
    }
    else return ProcessorTypeDefault;
}
const TCHAR* RegArgument::getModelName() noexcept
{
    DWORD size = ModelNameMaxSize;
    auto data = reinterpret_cast<LPBYTE>(modelName);
    if (ERROR_SUCCESS == RegQueryValueEx(key, ModelNameValueName, nullptr, nullptr, data, &size))
    {
        modelName[size - 1] = TEXT('\0');
        return modelName;
    }
    else return ModelNameDefault;
}
void RegArgument::setFactor(const double v) const noexcept
{
    DWORD size = sizeof(factor);
    auto data = reinterpret_cast<const BYTE*>(&v);
    RegSetValueEx(key, FactorValueName, 0, REG_BINARY, data, size);
}
void RegArgument::setDevice(const int v) const noexcept
{
    DWORD size = sizeof(device);
    auto data = reinterpret_cast<const BYTE*>(&v);
    RegSetValueEx(key, DeviceValueName, 0, REG_DWORD, data, size);
}
void RegArgument::setLimitWidth(const int v) const noexcept
{
    DWORD size = sizeof(limitWidth);
    auto data = reinterpret_cast<const BYTE*>(&v);
    RegSetValueEx(key, LimitWidthValueName, 0, REG_DWORD, data, size);
}
void RegArgument::setLimitHeight(const int v) const noexcept
{
    DWORD size = sizeof(limitHeight);
    auto data = reinterpret_cast<const BYTE*>(&v);
    RegSetValueEx(key, LimitHeightValueName, 0, REG_DWORD, data, size);
}
void RegArgument::setProcessorType(const TCHAR* const v) const noexcept
{
    DWORD size = ProcessorTypeMaxSize;
    auto data = reinterpret_cast<const BYTE*>(v);
    RegSetValueEx(key, ProcessorTypeValueName, 0, REG_SZ, data, size);
}
void RegArgument::setModelName(const TCHAR* const v) const noexcept
{
    DWORD size = ModelNameMaxSize;
    auto data = reinterpret_cast<const BYTE*>(v);
    RegSetValueEx(key, ModelNameValueName, 0, REG_SZ, data, size);
}

RegArgument& RegArgument::instance() noexcept
{
    static RegArgument object{};
    return object;
}
