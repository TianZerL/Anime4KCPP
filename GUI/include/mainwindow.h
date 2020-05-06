#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "communicator.h"
#include "Anime4K.h"
#include "Anime4KGPU.h"

#include <QMainWindow>
#include <QTranslator>
#include <QCloseEvent>
#include <QMessageBox>
#include <QStandardItemModel>
#include <QFileDialog>
#include <QFileInfo>
#include <QMimeData>
#include <QtConcurrent/QtConcurrent>
#include <QMetaType>
#include <QClipboard>
#include <QSettings>
#include <QDesktopServices>

#include <opencv2/opencv.hpp>

#define ANIME4KCPP_GUI_VERSION "1.4.0"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

enum Language
{
    en = 0, zh_cn = 1
};

enum ErrorType
{
    INPUT_NONASCII = 0, PROCESSING_LIST_EMPTY=1,
    FILE_NOT_EXIST = 2, DIR_NOT_EXIST=3,
    TYPE_NOT_IMAGE = 4, TYPE_NOT_ADD = 5
};

enum FileType
{
    IMAGE = 0, VIDEO = 1, ERROR_TYPE=2
};

enum GPUMode
{
    GPUMODE_INITIALZED = 0, GPUMODE_UNINITIALZED = 1, GPUMODE_UNSUPPORT = 3
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
    void readConfig(const QSettings *conf);
    void writeConfig(QSettings *conf);
    Language getLanguage(const QString &lang);
    QString getLanguage(const Language lang);
    void errorHandler(const ErrorType err);
    void initTextBrowser();
    bool checkFFmpeg();
    QString formatSuffixList(const QString &&type, QString str);
    void initAnime4K(Anime4K *&anime4K);
    void releaseMainAnime4K();
    FileType fileType(const QFileInfo &file);
    QString getOutputPrefix();
    CODEC getCodec(const QString &codec);

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

    void on_pushButtonClearText_clicked();

    void on_spinBoxFontSize_valueChanged(int arg1);

    void on_fontComboBox_currentFontChanged(const QFont &f);

    void on_pushButtonCopyText_clicked();

    void on_pushButtonPreviewOrgin_clicked();

    void on_pushButtonPreviewOnlyResize_clicked();

    void on_pushButtonPickFolder_clicked();

    void on_checkBoxGPUMode_stateChanged(int arg1);

    void on_actionList_GPUs_triggered();

    void on_pushButtonListGPUs_clicked();

    void on_spinBoxPlatformID_valueChanged(int arg1);

    void on_pushButton_clicked();

    void on_pushButtonReleaseGPU_clicked();

private:
    Ui::MainWindow *ui;
    QTranslator *translator;
    QStandardItemModel *tableModel;
    QSettings *config;
    quint64 totalTime;
    int imageCount;
    int videoCount;
    bool ffmpeg;
    unsigned int totalTaskCount;
    Language currLanguage;

    GPUMode GPU;
    Anime4K *mainAnime4kCPU;
    Anime4KGPU *mainAnime4kGPU;

    std::vector<int> devices;
    int platforms;

    QHash<QString,Language> languageSelector;
    QHash<QString,CODEC> codecSelector;
};
#endif // MAINWINDOW_H
