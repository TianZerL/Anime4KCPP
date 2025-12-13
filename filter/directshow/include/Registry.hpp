#ifndef AC_FILTER_DIRECTSHOW_REGISTRY_HPP
#define AC_FILTER_DIRECTSHOW_REGISTRY_HPP

#include <windows.h>

#define gRegArgument (RegArgument::instance())

class RegArgument
{
private:
    RegArgument() noexcept;
    ~RegArgument() noexcept;

public:
    double getFactor() noexcept;
    int getDevice() noexcept;
    int getLimitWidth() noexcept;
    int getLimitHeight() noexcept;
    const TCHAR* getProcessorType() noexcept;
    const TCHAR* getModelName() noexcept;

    void setFactor(double v) const noexcept;
    void setDevice(int v) const noexcept;
    void setLimitWidth(int v) const noexcept;
    void setLimitHeight(int v) const noexcept;
    void setProcessorType(const TCHAR* v) const noexcept;
    void setModelName(const TCHAR* v) const noexcept;

public:
    static RegArgument& instance() noexcept;

public:
    static constexpr int ProcessorTypeMaxSize = 32;
    static constexpr int ModelNameMaxSize = 32;

    static constexpr double FactorDefault = 2.0;
    static constexpr int DeviceDefault = 0;
    static constexpr int LimitWidthDefault = 1280;
    static constexpr int LimitHeightDefault = 720;
private:
    static constexpr const TCHAR* ProcessorTypeDefault = TEXT("cpu");
    static constexpr const TCHAR* ModelNameDefault = TEXT("acnet-hdn0");
    static constexpr const TCHAR* FactorValueName = TEXT("Factor");
    static constexpr const TCHAR* DeviceValueName = TEXT("Device");
    static constexpr const TCHAR* LimitWidthValueName = TEXT("LimitWidth");
    static constexpr const TCHAR* LimitHeightValueName = TEXT("LimitHeight");
    static constexpr const TCHAR* ProcessorTypeValueName = TEXT("ProcessorType");
    static constexpr const TCHAR* ModelNameValueName = TEXT("ModelName");
private:
    HKEY key{};
    double factor{};
    int device{};
    int limitWidth{};
    int limitHeight{};
    TCHAR ProcessorType[ProcessorTypeMaxSize]{};
    TCHAR modelName[ModelNameMaxSize]{};
};

#endif
