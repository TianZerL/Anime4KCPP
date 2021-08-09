#ifndef COMMUNICATOR_H
#define COMMUNICATOR_H

#include <QObject>

class Communicator : public QObject
{
    Q_OBJECT
public:
    explicit Communicator(QObject* parent = nullptr);

signals:
    void error(int row, QString err);
    void done(int row, double pro, quint64 time);
    void allDone(quint64 totalTime);
    void showInfo(std::string info);
    void updateProgress(double v, double elpsed, double remaining);
};

#endif // COMMUNICATOR_H
