#include <mutex>

#include <QFile>
#include <QMessageLogger>
#include <QtGlobal>

#include "Config.hpp"
#include "Logger.hpp"

#define AC_GUI_SETTINGS_FILE "ac_gui.log"

Logger& Logger::instance() noexcept
{
    static Logger logger{};
    return logger;
}

Logger::Logger() noexcept
{
    static std::mutex mtx{};
    static QFile log { logFilePath() };
    if (log.open(QIODevice::Text | QIODevice::WriteOnly))
    {
        qInstallMessageHandler([](QtMsgType type, const QMessageLogContext& context, const QString& msg) {
            QString color{ [=]() {
                if (type == QtMsgType::QtInfoMsg) return "DeepSkyBlue";
                if (type == QtMsgType::QtWarningMsg) return "Orange";
                if (type == QtMsgType::QtCriticalMsg) return "Red";
                return "Blue";
            }() };
            {
                const std::lock_guard<std::mutex> lock(mtx);
                log.write(qFormatLogMessage(type, context, msg + '\n').toUtf8());
                log.flush();
            }
            emit gLogger.logged(qFormatLogMessage(type, context, QString{ "<span style='color:%1;white-space: pre'>%2</span>\n" }.arg(color, msg)));
        });
        qSetMessagePattern("[%{time yyyy-MM-dd h:mm:ss} %{if-debug}D%{endif}%{if-info}I%{endif}%{if-warning}W%{endif}%{if-critical}C%{endif}%{if-fatal}F%{endif}] - %{message}");
    }
}

Logger::~Logger() noexcept
{
    qInstallMessageHandler(nullptr);
}

QDebug Logger::info() const noexcept
{
    return QMessageLogger().info().noquote().nospace();
}
QDebug Logger::warning() const noexcept
{
    return QMessageLogger().warning().noquote().nospace();
}
QDebug Logger::error() const noexcept
{
    return QMessageLogger().critical().noquote().nospace();
}

QString Logger::logFilePath()
{
    return gConfig.userDataDir().filePath(AC_GUI_SETTINGS_FILE);
}
