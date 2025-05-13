#ifndef AC_GUI_LOGGER_HPP
#define AC_GUI_LOGGER_HPP

#include <QObject>
#include <QDebug>
#include <QString>

#define gLogger (Logger::instance())

class Logger : public QObject
{
    Q_OBJECT

public:
    static Logger& instance() noexcept;

private:
    Logger() noexcept;
    ~Logger() noexcept;

public:
    QDebug info() const noexcept;
    QDebug warning() const noexcept;
    QDebug error() const noexcept;

signals:
    void logged(const QString& msg);

public:
    static QString logFilePath();

};

#endif
