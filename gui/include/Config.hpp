#ifndef AC_GUI_CONFIG_HPP
#define AC_GUI_CONFIG_HPP

#include <QDir>
#include <QString>
#include <QStringList>

#define gConfig (Config::instance())

class Config
{
public:
    struct {
        bool exitConfirmation = false;
        QString styleName{"Fusion"};
    } gui{};

    struct {
        QString imageSuffix{".jpg"};
        QString videoSuffix{".mp4"};
        QString imagePrefix{"ac_"};
        QString videoPrefix{"ac_"};
        QString imageOutputPath{};
        QString videoOutputPath{};
    } io{};

    struct {
        const QStringList modelList{ "acnet-hdn0", "acnet-hdn1", "acnet-hdn2", "acnet-hdn3" };
        const QStringList processorList{
            "cpu",
#           ifdef AC_CORE_WITH_OPENCL
            "opencl",
#           endif
#           ifdef AC_CORE_WITH_CUDA
            "cuda",
#           endif
        };

        int device = 0;
        double factor = 2.0;
        QString processor{processorList[0]};
        QString model{modelList[0]};
    } upscaler{};

    struct {
        // decoder hints
        QString decoder{};
        QString format{};
        // encoder hints
        QString encoder{};
        int bitrate = 0;
    } video{};

public:
    static Config& instance() noexcept;

private:
    Config() noexcept;
    ~Config() noexcept;

    void save() noexcept;
    void load() noexcept;

public:
    static QDir userDataDir() noexcept;
};

#endif
