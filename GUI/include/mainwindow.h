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
#include <QMetaType>
#include <QClipboard>
#include <QSettings>

#include <opencv2/opencv.hpp>

#define CORE_VERSION "1.3.2"
#define VERSION "1.1"

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
    FILE_NOT_EXIST = 2, TYPE_NOT_IMAGE = 3,
    TYPE_NOT_ADD = 4
};

enum FileType
{
    IMAGE = 0, VIDEO = 1, ERROR_TYPE=2
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
    void readConfig(QSettings *conf);
    void writeConfig(QSettings *conf);
    Language getLanguage(QString &lang);
    QString getLanguage(Language lang);
    void errorHandler(ErrorType err);
    void initTextBrowser();
    bool checkFFmpeg();
    QString formatSuffixList(const QString &&type, QString str);
    void initAnime4K(Anime4K *&anime4K);
    void releaseAnime4K(Anime4K *&anime4K);
    FileType fileType(QFileInfo &file);
    QString getOutputPrefix();

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
};
#endif // MAINWINDOW_H
