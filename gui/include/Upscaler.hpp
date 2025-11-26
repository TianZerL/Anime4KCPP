#ifndef AC_GUI_UPSCALER_HPP
#define AC_GUI_UPSCALER_HPP

#include <memory>

#include <QObject>
#include <QList>
#include <QString>

#define gUpscaler (Upscaler::instance())

class TaskData : public QObject
{
    Q_OBJECT

signals:
    void finished(bool success);

public:
    enum {
        TYPE_IMAGE,
        TYPE_VIDEO,
    } type;

    struct {
        QString input;
        QString output;
    } path;
};

class Upscaler : public QObject
{
    Q_OBJECT

private:
    struct UpscalerData;

public:
    static Upscaler& instance() noexcept;
    static QString& listProcessorInfo();

public:
    Upscaler();
    ~Upscaler() noexcept override;

    void start(const QList<QSharedPointer<TaskData>>& taskList);
    void stop();

signals:
    void progress(int value);
    void started();
    void stopped();

private:
    const std::unique_ptr<UpscalerData> dptr;
};

#endif
