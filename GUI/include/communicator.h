#ifndef COMMUNICATOR_H
#define COMMUNICATOR_H

#include <QObject>

class Communicator : public QObject
{
    Q_OBJECT
public:
    explicit Communicator(QObject *parent = nullptr);

signals:
    void error(int row, QString err);
    void done(int row, double pro, quint64 time);
    void allDone();
    void showInfo(std::string info);
};

#endif // COMMUNICATOR_H
