#include <QCoreApplication>
#include <QSettings>

#include "AC/Specs.hpp"

#include "Config.hpp"

#define AC_GUI_USER_DATA_FOLDER "user"
#define AC_GUI_DEFAULT_OUTPUT_FOLDER "output"
#define AC_GUI_SETTINGS_FILE "ac_gui.ini"

Config& Config::instance() noexcept
{
    static Config config{};
    return config;
}

Config::Config() noexcept
{
    QDir dir{ QCoreApplication::applicationDirPath() };
    dir.mkpath(AC_GUI_USER_DATA_FOLDER);
    dir.mkpath(AC_GUI_DEFAULT_OUTPUT_FOLDER);
    dir.cd(AC_GUI_DEFAULT_OUTPUT_FOLDER);
    io.imageOutputPath = dir.canonicalPath();
    io.videoOutputPath = dir.canonicalPath();

    upscaler.model = ac::specs::ModelList[0];
    upscaler.processor = ac::specs::ProcessorList[0];

    load();
}
Config::~Config() noexcept
{
    save();
}

void Config::save() noexcept
{
    QSettings settings{ userDataDir().filePath(AC_GUI_SETTINGS_FILE), QSettings::IniFormat };

    settings.beginGroup("GUI");
    settings.setValue("ExitConfirmation", gui.exitConfirmation);
    settings.setValue("StyleName", gui.styleName);
    settings.endGroup();

    settings.beginGroup("IO");
    settings.setValue("ImageSuffix", io.imageSuffix);
    settings.setValue("VideoSuffix", io.videoSuffix);
    settings.setValue("ImagePrefix", io.imagePrefix);
    settings.setValue("VideoPrefix", io.videoPrefix);
    settings.setValue("ImageOutputPath", io.imageOutputPath);
    settings.setValue("VideoOutputPath", io.videoOutputPath);
    settings.endGroup();

    settings.beginGroup("UPSCALER");
    settings.setValue("Processor", upscaler.processor);
    settings.setValue("Device", upscaler.device);
    settings.setValue("Factor", upscaler.factor);
    settings.setValue("Model", upscaler.model);
    settings.endGroup();

    settings.beginGroup("VIDEO");
    settings.setValue("Decoder", video.decoder);
    settings.setValue("Format", video.format);
    settings.setValue("Encoder", video.encoder);
    settings.setValue("Bitrate", video.bitrate);
    settings.endGroup();
}
void Config::load() noexcept
{
    QSettings settings{ userDataDir().filePath(AC_GUI_SETTINGS_FILE), QSettings::IniFormat };

    settings.beginGroup("GUI");
    gui.exitConfirmation = settings.value("ExitConfirmation", gui.exitConfirmation).toBool();
    gui.styleName = settings.value("StyleName", gui.styleName).toString();
    settings.endGroup();

    settings.beginGroup("IO");
    io.imageSuffix = settings.value("ImageSuffix", io.imageSuffix).toString();
    io.videoSuffix = settings.value("VideoSuffix", io.videoSuffix).toString();
    io.imagePrefix = settings.value("ImagePrefix", io.imagePrefix).toString();
    io.videoPrefix = settings.value("VideoPrefix", io.videoPrefix).toString();
    io.imageOutputPath = settings.value("ImageOutputPath", io.imageOutputPath).toString();
    io.videoOutputPath = settings.value("VideoOutputPath", io.videoOutputPath).toString();
    settings.endGroup();

    settings.beginGroup("UPSCALER");
    upscaler.processor = settings.value("Processor", upscaler.processor).toString();
    upscaler.device = settings.value("Device", upscaler.device).toInt();
    upscaler.factor = settings.value("Factor", upscaler.factor).toDouble();
    upscaler.model = settings.value("Model", upscaler.model).toString();
    settings.endGroup();

    settings.beginGroup("VIDEO");
    video.decoder = settings.value("Decoder", video.decoder).toString();
    video.format = settings.value("Format", video.format).toString();
    video.encoder = settings.value("Encoder", video.encoder).toString();
    video.bitrate = settings.value("Bitrate", video.bitrate).toInt();
    settings.endGroup();
}

QDir Config::userDataDir() noexcept
{
    static auto path = QCoreApplication::applicationDirPath();
    QDir dir{ path };
    dir.cd(AC_GUI_USER_DATA_FOLDER);
    return dir;
}
