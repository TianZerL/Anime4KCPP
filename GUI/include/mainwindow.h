#ifndef ANIME4KCPP_GUI_MAINWINDOW_H
#define ANIME4KCPP_GUI_MAINWINDOW_H

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
#include <QInputDialog>
#include <QStyleFactory>

#include "Anime4KCPP.hpp"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

enum class Language
{
    en, zh_CN, zh_TW, ja_JP, fr_FR
};

enum class Style
{
    DEFAULT, FUSION, FUSION_DARK
};

enum class ErrorType
{
    PROCESSING_LIST_EMPTY,
    FILE_NOT_EXIST, DIR_NOT_EXIST,
    TYPE_NOT_IMAGE, BAD_TYPE,
    URL_INVALID, IMAGE_FORMAT_INVALID,
    OPENCL_NOT_SUPPORT,
    CUDA_NOT_SUPPORT
};

enum class FileType
{
    IMAGE, VIDEO, GIF, BAD_TYPE
};

enum class GPUMode
{
    INITIALZED, UNINITIALZED, UNSUPPORT
};

enum class ProcessingState
{
    STOP, PROCESSING, PAUSE, PAUSED, CONTINUE
};

enum class GPGPU
{
    OpenCL = 0, CUDA = 1
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

protected:
    void closeEvent(QCloseEvent* event);
    void dragEnterEvent(QDragEnterEvent* event);
    void dropEvent(QDropEvent* event);

private:
    bool addTask(const QFileInfo& fileInfo, const QString& path, bool perfix = true);

    void readConfig(const QSettings* conf);
    void writeConfig(QSettings* conf);

    QString getOutputpath();

    Language getLanguageValue(const QString& lang);
    QString getLanguageString(const Language lang);

    QString getStyleString(const Style style);

    void errorHandler(const ErrorType err);
    void errorHandler(const QString& err);

    void initTextBrowser();
    bool checkFFmpegPath(const QString& path);
    QString formatSuffixList(const QString&& type, QString str);
    FileType fileType(const QFileInfo& file);
    QString getOutputPrefix();
    QString getImageOutputSuffix();
    QString getVideoOutputSuffix();
    bool checkGIF(const QString& file);
    bool mergeAudio2Video(const QString& dstFile, const QString& srcFile, const QString& tmpFile);
    bool video2GIF(const QString& srcFile, const QString& dstFile);

    std::unique_ptr<Anime4KCPP::AC> getACUP();
    Anime4KCPP::Codec getCodec(const QString& codec);

    void logToTextBrowser(const QString& info);

private slots:
    void solt_done_renewState(int row, double pro, quint64 time);
    void solt_setError_renewState(int row);
    void solt_showError_renewState(QString err);
    void solt_allDone_remindUser(quint64 totalTime);
    void solt_logInfo_renewTextBrowser(QString info);
    void solt_updateProgress_updateCurrentTaskProgress(double v, double elpsed, double remaining);

private slots:
    void on_actionQuit_triggered();

    void on_pushButtonPickFiles_clicked();

    void on_pushButtonOutputPathPick_clicked();

    void on_pushButtonWebVideo_clicked();

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

    void on_actionSimplifiedChinese_triggered();

    void on_actionTraditionalChinese_triggered();

    void on_actionJapanese_triggered();

    void on_actionFrench_triggered();

    void on_actionEnglish_triggered();

    void on_actionSet_FFmpeg_path_triggered();

    void on_actionSet_FFmpeg_path_hovered();

    void on_actionDefault_triggered();

    void on_actionFusion_triggered();

    void on_actionFusion_dark_triggered();

    void on_pushButtonClearText_clicked();

    void on_spinBoxFontSize_valueChanged(int arg1);

    void on_fontComboBox_currentFontChanged(const QFont& f);

    void on_pushButtonCopyText_clicked();

    void on_pushButtonPreviewOriginal_clicked();

    void on_pushButtonPreviewOnlyResize_clicked();

    void on_pushButtonPickFolder_clicked();

    void on_checkBoxGPUMode_stateChanged(int arg1);

    void on_actionList_GPUs_triggered();

    void on_actionBenchmark_triggered();

    void on_pushButtonListGPUs_clicked();

    void on_pushButtonOutputPathOpen_clicked();

    void on_pushButtonReleaseGPU_clicked();

    void on_checkBoxACNet_stateChanged(int arg1);

    void on_pushButtonStop_clicked();

    void on_pushButtonPause_clicked();

    void on_pushButtonContinue_clicked();

    void on_comboBoxGPGPU_currentIndexChanged(int idx);
private:
    Ui::MainWindow* ui;
    QTranslator* translator;
    QStandardItemModel* tableModel;
    QSettings* config;

    bool foundFFmpegFlag;
    QString ffmpegPath;
    int totalTaskCount;
    Language currLanguage;
    Style currStyle;

    GPUMode GPUState;

    Anime4KCPP::ACInitializer initializer;

    QHash<QString, Language> languageSelector;
    QHash<QString, Anime4KCPP::Codec> codecSelector;

    std::atomic<bool> stopProcessing;
    std::atomic<ProcessingState> processingState;
};

#endif // !ANIME4KCPP_GUI_MAINWINDOW_H
