#include <QtLogging>
#include <QMessageLogger>
#include <QFile>

#include "Config.hpp"
#include "Logger.hpp"

#define AC_GUI_SETTINGS_FILE "ac_gui.log.md"

Logger& Logger::instance() noexcept
{
    static Logger logger{};
    return logger;
}

Logger::Logger() noexcept
{
    static QFile log {logFilePath()};
    log.open(QIODevice::Text | QIODevice::WriteOnly);
    qInstallMessageHandler([](QtMsgType type, const QMessageLogContext& context, const QString& msg) {
        QString color{ [=]() {
            if (type == QtMsgType::QtInfoMsg) return "DeepSkyBlue";
            if (type == QtMsgType::QtWarningMsg) return "Orange";
            if (type == QtMsgType::QtCriticalMsg) return "Red";
            return "Blue";
        }() };
        QString buffer = msg.contains('\n') ? QString("\n```\n%1\n```  ").arg(msg) : QString("<span style='color:%1'>%2</span>  ").arg(color, msg);
        log.write(qFormatLogMessage(type, context, buffer).toUtf8());
        log.putChar('\n');
        log.flush();
        emit gLogger.logged();
    });
    qSetMessagePattern("[%{time yyyy-MM-dd h:mm:ss} %{if-debug}D%{endif}%{if-info}I%{endif}%{if-warning}W%{endif}%{if-critical}C%{endif}%{if-fatal}F%{endif}] - %{message}");

    log.write("### Anime4KCPP GUI v" AC_CORE_VERSION_STR "\n\n");
    log.flush();
}

Logger::~Logger() noexcept
{
    qInstallMessageHandler(nullptr);
}

QDebug Logger::info() const noexcept
{
    return QMessageLogger().info().noquote();
}
QDebug Logger::warning() const noexcept
{
    return QMessageLogger().warning().noquote();
}
QDebug Logger::error() const noexcept
{
    return QMessageLogger().critical().noquote();
}

QString Logger::logFilePath()
{
    return gConfig.userDataDir().filePath(AC_GUI_SETTINGS_FILE);
}
