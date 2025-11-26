#ifndef AC_GUI_CONFIG_HPP
#define AC_GUI_CONFIG_HPP

#include <QDir>
#include <QString>

#define gConfig (Config::instance())

class Config
{
public:
    struct {
        bool exitConfirmation = false;
        QString styleName{ "Fusion" };
    } gui{};

    struct {
        QString imageSuffix{ ".jpg" };
        QString videoSuffix{ ".mkv" };
        QString imagePrefix{ "ac_" };
        QString videoPrefix{ "ac_" };
        QString imageOutputPath{};
        QString videoOutputPath{};
    } io{};

    struct {
        int device = 0;
        double factor = 2.0;
        QString processor{};
        QString model{};
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
