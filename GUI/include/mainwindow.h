#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "communicator.h"
#include "Anime4K.h"

#include <QMainWindow>
#include <QTranslator>
#include <QCloseEvent>
#include <QMessageBox>
#include <QStandardItemModel>
#include <QFileDialog>
#include <QFileInfo>
#include <QMimeData>
#include <QtConcurrent/QtConcurrent>
#include <QMutex>
#include <QMetaType>

#define CORE_VERSION "1.3"
#define VERSION "0.9"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

enum FileType
{
    IMAGE = 0, VIDEO = 1, ERROR=2
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    void closeEvent(QCloseEvent *event);
    void dragEnterEvent(QDragEnterEvent *event);
    void dropEvent(QDropEvent *event);

private:
    void initTextBrowser();
    bool checkFFmpeg();
    QString formatSuffixList(const QString &&type, QString str);
    void initAnime4K(Anime4K *&anime4K);
    void releaseAnime4K(Anime4K *&anime4K);
    FileType fileType(QFileInfo &file);

private slots:
    void solt_done_renewState(int row, double pro, quint64 time);
    void solt_error_renewState(int row, QString err);
    void solt_allDone_remindUser();
    void solt_showInfo_renewTextBrowser(std::string info);

private slots:
    void on_actionQuit_triggered();

    void on_pushButtonInputPath_clicked();

    void on_pushButtonOutputPath_clicked();

    void on_pushButtonClear_clicked();

    void on_pushButtonDelete_clicked();

    void on_radioButtonFast_clicked();

    void on_radioButtonBalance_clicked();

    void on_radioButtonQuality_clicked();

    void on_checkBoxEnablePreprocessing_stateChanged(int arg1);

    void on_checkBoxEnablePostprocessing_stateChanged(int arg1);

    void on_pushButtonPreview_clicked();

    void on_pushButtonPreviewPick_clicked();

    void on_pushButtonStart_clicked();

    void on_actionAbout_triggered();

    void on_tabWidgetMain_tabBarClicked(int index);

    void on_actionChinese_triggered();

    void on_actionEnglish_triggered();

private:
    Ui::MainWindow *ui;
    QTranslator *translator;
    QStandardItemModel *tableModel;
    QMutex *mutex;
    quint64 totalTime;
    int imageCount;
    int videoCount;
    bool ffmpeg;
};
#endif // MAINWINDOW_H
