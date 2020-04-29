#include "mainwindow.h"
#include "./ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    //inital translator
    translator = new QTranslator(this);
    //inital textBrowser
    ui->fontComboBox->setFont(QFont("Consolas"));
    ui->fontComboBox->setCurrentFont(ui->fontComboBox->font());
    ui->spinBoxFontSize->setRange(9,30);
    ui->spinBoxFontSize->setValue(9);
    initTextBrowser();
    //accept drops
    this->setAcceptDrops(true);
    //inital tableView
    tableModel = new QStandardItemModel(this);
    tableModel->setColumnCount(5);
    tableModel->setHorizontalHeaderLabels({"Input file",
                                           "Output file",
                                           "Full path",
                                           "Output path",
                                           "State"});
    ui->tableViewProcessingList->setModel(tableModel);
    //inital processBar
    ui->progressBarProcessingList->reset();
    ui->progressBarProcessingList->setRange(0, 100);
    ui->progressBarProcessingList->setEnabled(false);
    //inital arguments
    ui->spinBoxThreads->setMinimum(1);
    ui->doubleSpinBoxPushColorStrength->setRange(0.0,1.0);
    ui->doubleSpinBoxPushGradientStrength->setRange(0.0,1.0);
    ui->doubleSpinBoxZoomFactor->setRange(1.0,10.0);
    //inital time and count
    totalTaskCount = totalTime = imageCount = videoCount = 0;
    //inital ffmpeg
    ffmpeg = checkFFmpeg();
    //inital config
    config = new QSettings("settings.ini", QSettings::IniFormat, this);
    readConfig(config);
    //inital Anime4KCoreForCPU
    mainAnime4kCPU = new Anime4K;
    mainAnime4kGPU = nullptr;
    //inital GPU
    GPU = GPUMODE_UNINITIALZED;
    //Register
    qRegisterMetaType<std::string>("std::string");
}

MainWindow::~MainWindow()
{
    delete ui;
    releaseMainAnime4K();
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    if (!ui->actionQuit_confirmation->isChecked())
    {
        writeConfig(config);
        event->accept();
        return;
    }

    if (QMessageBox::Yes == QMessageBox::warning(this, tr("Confirm"),
                                                 tr("Do you really want to exit?"),
                                                 QMessageBox::Yes|QMessageBox::No, QMessageBox::No))
    {
        writeConfig(config);
        event->accept();
        return;
    }

    event->ignore();
}

void MainWindow::dragEnterEvent(QDragEnterEvent *event)
{
    event->acceptProposedAction();
}

void MainWindow::dropEvent(QDropEvent *event)
{
    QList<QUrl> urls = event->mimeData()->urls();
    if(urls.isEmpty())
        return;

    QStringList files;

    bool flag=false;
    for (QUrl &url:urls)
    {
        QString file = url.toLocalFile();
        if(file.contains(QRegExp("[^\\x00-\\xff]")))
        {
            flag = true;
            continue;
        }
        if (!files.contains(file))
            files.append(file);
    }

    QStandardItem *inputFile;
    QStandardItem *outputFile;
    QStandardItem *inputPath;
    QStandardItem *outputPath;
    QStandardItem *state;

    for(QString &file:files)
    {
        QFileInfo fileInfo(file);

        if (fileType(fileInfo)==ERROR_TYPE){
            errorHandler(TYPE_NOT_ADD);
            continue;
        }

        inputFile = new QStandardItem(fileInfo.fileName());
        if(fileType(fileInfo)==VIDEO)
            outputFile = new QStandardItem(getOutputPrefix()+fileInfo.baseName()+".mp4");
        else
            outputFile = new QStandardItem(getOutputPrefix()+fileInfo.fileName());
        inputPath = new QStandardItem(fileInfo.filePath());
        state = new QStandardItem(tr("ready"));
        if (ui->lineEditOutputPath->text().isEmpty())
            outputPath = new QStandardItem(QDir::currentPath());
        else
            outputPath = new QStandardItem(ui->lineEditOutputPath->text());
        tableModel->appendRow({inputFile,outputFile,inputPath,outputPath,state});

        totalTaskCount++;
    }

    ui->labelTotalTaskCount->setText(QString("Total: %1 ").arg(totalTaskCount));

    if (flag)
    {
        errorHandler(INPUT_NONASCII);
    }

}

void MainWindow::readConfig(const QSettings *conf)
{
    QString language = conf->value("/GUI/language","en").toString();
    bool quitConfirmatiom = conf->value("/GUI/quitConfirmatiom","en").toBool();

    QString imageSuffix = conf->value("/Suffix/image","png:jpg:jpeg:bmp").toString();
    QString videoSuffix = conf->value("/Suffix/video","mp4:mkv:avi:m4v:flv:3gp:wmv:mov").toString();
    QString outputPath = conf->value("/Output/path",QApplication::applicationDirPath()+"/output").toString();
    QString outputPrefix = conf->value("/Output/perfix","output_anime4kcpp_").toString();

    int passes = conf->value("/Arguments/passes",2).toInt();
    int pushColorCount = conf->value("/Arguments/pushColorCount",2).toInt();
    double pushColorStrength = conf->value("/Arguments/pushColorStrength",0.3).toDouble();
    double pushGradientStrength = conf->value("/Arguments/pushGradientStrength",1.0).toDouble();
    double zoomFactor = conf->value("/Arguments/zoomFactor",2.0).toDouble();
    unsigned int threads = conf->value("/Arguments/threads",std::thread::hardware_concurrency()).toUInt();
    bool fastMode = conf->value("/Arguments/fastMode",false).toBool();

    bool enablePreprocessing = conf->value("/Preprocessing/enable",true).toBool();
    bool preMedian = conf->value("/Preprocessing/MedianBlur",false).toBool();
    bool preMean = conf->value("/Preprocessing/MeanBlur",false).toBool();
    bool preCAS = conf->value("/Preprocessing/CASSharpening",true).toBool();
    bool preGaussianWeak = conf->value("/Preprocessing/GaussianBlurWeak",false).toBool();
    bool preGaussian = conf->value("/Preprocessing/GaussianBlur",false).toBool();
    bool preBilateral = conf->value("/Preprocessing/BilateralFilter",false).toBool();
    bool preBilateralFaster = conf->value("/Preprocessing/BilateralFilterFaster",false).toBool();

    bool enablePostprocessing = conf->value("/Postprocessing/enable",true).toBool();
    bool postMedian = conf->value("/Postprocessing/MedianBlur",false).toBool();
    bool postMean = conf->value("/Postprocessing/MeanBlur",false).toBool();
    bool postCAS = conf->value("/Postprocessing/CASSharpening",false).toBool();
    bool postGaussianWeak = conf->value("/Postprocessing/GaussianBlurWeak",true).toBool();
    bool postGaussian = conf->value("/Postprocessing/GaussianBlur",false).toBool();
    bool postBilateral = conf->value("/Postprocessing/BilateralFilter",true).toBool();
    bool postBilateralFaster = conf->value("/Postprocessing/BilateralFilterFaster",false).toBool();

    //GUI options
    //set language
    switch (getLanguage(language))
    {
    case en:
        on_actionEnglish_triggered();
        break;
    case zh_cn:
        on_actionChinese_triggered();
        break;
    }
    ui->actionQuit_confirmation->setChecked(quitConfirmatiom);
    //suffix
    ui->lineEditImageSuffix->setText(imageSuffix);
    ui->lineEditVideoSuffix->setText(videoSuffix);
    //output
    ui->lineEditOutputPath->setText(outputPath);
    ui->lineEditOutputPrefix->setText(outputPrefix);
    //arguments
    ui->spinBoxPasses->setValue(passes);
    ui->spinBoxPushColorCount->setValue(pushColorCount);
    ui->doubleSpinBoxPushColorStrength->setValue(pushColorStrength);
    ui->doubleSpinBoxPushGradientStrength->setValue(pushGradientStrength);
    ui->doubleSpinBoxZoomFactor->setValue(zoomFactor);
    ui->spinBoxThreads->setValue(threads);
    ui->checkBoxFastMode->setChecked(fastMode);
    //preprocessing
    ui->checkBoxEnablePreprocessing->setChecked(enablePreprocessing);
    ui->checkBoxPreMedian->setChecked(preMedian);
    ui->checkBoxPreMean->setChecked(preMean);
    ui->checkBoxPreCAS->setChecked(preCAS);
    ui->checkBoxPreGaussianWeak->setChecked(preGaussianWeak);
    ui->checkBoxPreGaussian->setChecked(preGaussian);
    ui->checkBoxPreBilateral->setChecked(preBilateral);
    ui->checkBoxPreBilateralFaster->setChecked(preBilateralFaster);
    //postprocessing
    ui->checkBoxEnablePostprocessing->setChecked(enablePostprocessing);
    ui->checkBoxPostMedian->setChecked(postMedian);
    ui->checkBoxPostMean->setChecked(postMean);
    ui->checkBoxPostCAS->setChecked(postCAS);
    ui->checkBoxPostGaussianWeak->setChecked(postGaussianWeak);
    ui->checkBoxPostGaussian->setChecked(postGaussian);
    ui->checkBoxPostBilateral->setChecked(postBilateral);
    ui->checkBoxPostBilateralFaster->setChecked(postBilateralFaster);
}

void MainWindow::writeConfig(QSettings *conf)
{
    QString language = getLanguage(currLanguage);
    bool quitConfirmatiom = ui->actionQuit_confirmation->isChecked();

    QString imageSuffix = ui->lineEditImageSuffix->text();
    QString videoSuffix = ui->lineEditVideoSuffix->text();
    QString outputPath = ui->lineEditOutputPath->text();
    QString outputPrefix = ui->lineEditOutputPrefix->text();

    int passes = ui->spinBoxPasses->value();
    int pushColorCount = ui->spinBoxPushColorCount->value();
    double pushColorStrength = ui->doubleSpinBoxPushColorStrength->value();
    double pushGradientStrength = ui->doubleSpinBoxPushGradientStrength->value();
    double zoomFactor = ui->doubleSpinBoxZoomFactor->value();
    unsigned int threads = ui->spinBoxThreads->value();
    bool fastMode = ui->checkBoxFastMode->isChecked();

    bool enablePreprocessing = ui->checkBoxEnablePreprocessing->isChecked();
    bool preMedian = ui->checkBoxPreMedian->isChecked();
    bool preMean = ui->checkBoxPreMean->isChecked();
    bool preCAS = ui->checkBoxPreCAS->isChecked();
    bool preGaussianWeak = ui->checkBoxPreGaussianWeak->isChecked();
    bool preGaussian = ui->checkBoxPreGaussian->isChecked();
    bool preBilateral = ui->checkBoxPreBilateral->isChecked();
    bool preBilateralFaster = ui->checkBoxPreBilateralFaster->isChecked();

    bool enablePostprocessing = ui->checkBoxEnablePostprocessing->isChecked();
    bool postMedian = ui->checkBoxPostMedian->isChecked();
    bool postMean = ui->checkBoxPostMean->isChecked();
    bool postCAS = ui->checkBoxPostCAS->isChecked();
    bool postGaussianWeak = ui->checkBoxPostGaussianWeak->isChecked();
    bool postGaussian = ui->checkBoxPostGaussian->isChecked();
    bool postBilateral = ui->checkBoxPostBilateral->isChecked();
    bool postBilateralFaster = ui->checkBoxPostBilateralFaster->isChecked();

    conf->setValue("/GUI/language",language);
    conf->setValue("/GUI/quitConfirmatiom",quitConfirmatiom);

    conf->setValue("/Suffix/image",imageSuffix);
    conf->setValue("/Suffix/video",videoSuffix);
    conf->setValue("/Output/path",outputPath);
    conf->setValue("/Output/perfix",outputPrefix);

    conf->setValue("/Arguments/passes",passes);
    conf->setValue("/Arguments/pushColorCount",pushColorCount);
    conf->setValue("/Arguments/pushColorStrength",pushColorStrength);
    conf->setValue("/Arguments/pushGradientStrength",pushGradientStrength);
    conf->setValue("/Arguments/zoomFactor",zoomFactor);
    conf->setValue("/Arguments/threads",threads);
    conf->setValue("/Arguments/fastMode",fastMode);

    conf->setValue("/Preprocessing/enable",enablePreprocessing);
    conf->setValue("/Preprocessing/MedianBlur",preMedian);
    conf->setValue("/Preprocessing/MeanBlur",preMean);
    conf->setValue("/Preprocessing/CASSharpening",preCAS);
    conf->setValue("/Preprocessing/GaussianBlurWeak",preGaussianWeak);
    conf->setValue("/Preprocessing/GaussianBlur",preGaussian);
    conf->setValue("/Preprocessing/BilateralFilter",preBilateral);
    conf->setValue("/Preprocessing/BilateralFilterFaster",preBilateralFaster);

    conf->setValue("/Postprocessing/enable",enablePostprocessing);
    conf->setValue("/Postprocessing/MedianBlur",postMedian);
    conf->setValue("/Postprocessing/MeanBlur",postMean);
    conf->setValue("/Postprocessing/CASSharpening",postCAS);
    conf->setValue("/Postprocessing/GaussianBlurWeak",postGaussianWeak);
    conf->setValue("/Postprocessing/GaussianBlur",postGaussian);
    conf->setValue("/Postprocessing/BilateralFilter",postBilateral);
    conf->setValue("/Postprocessing/BilateralFilterFaster",postBilateralFaster);
}

Language MainWindow::getLanguage(const QString &lang)
{
    QMap<QString,Language> selector;
    selector["eh"]=en;
    selector["zh_cn"]=zh_cn;
    return selector[lang];
}

QString MainWindow::getLanguage(const Language lang)
{
    switch (lang)
    {
    case en:
        return "en";
    case zh_cn:
        return "zh_cn";
    }
    return "unknown";
}

void MainWindow::errorHandler(const ErrorType err)
{
    switch (err)
    {
    case INPUT_NONASCII:
        QMessageBox::information(this,
                                 tr("Error"),
                                 tr("Only ASCII encoding is supported"),
                                 QMessageBox::Ok);
        break;
    case PROCESSING_LIST_EMPTY:
        QMessageBox::information(this,
                                 tr("Error"),
                                 tr("Processing list empty"),
                                 QMessageBox::Ok);
        break;
    case FILE_NOT_EXIST:
        QMessageBox::information(this,
                                 tr("Error"),
                                 tr("File does not exists"),
                                 QMessageBox::Ok);
        break;
    case TYPE_NOT_IMAGE:
        QMessageBox::information(this,
                                 tr("Error"),
                                 tr("File type error, only image support"),
                                 QMessageBox::Ok);
        break;
    case TYPE_NOT_ADD:
        QMessageBox::information(this,
                                 tr("Error"),
                                 tr("File type error, you can add it manually"),
                                 QMessageBox::Ok);
        break;
    }
}

void MainWindow::initTextBrowser()
{
    ui->textBrowserInfoOut->setText(
                "----------------------------------------------\n"
                "        Welcome to use Anime4KCPP GUI         \n"
                "----------------------------------------------\n"+
        QString("         Anime4K GUI v%1                 \n"
                "         Anime4K Core v%2                \n"
                "----------------------------------------------\n").arg(VERSION, CORE_VERSION)
                );
    ui->textBrowserInfoOut->moveCursor(QTextCursor::End);
}

bool MainWindow::checkFFmpeg()
{
    if (!QProcess::execute("ffmpeg -version"))
    {
        ui->textBrowserInfoOut->insertPlainText(
                    "----------------------------------------------\n"
                    "               ffmpeg check OK                \n"
                    "----------------------------------------------\n"
                    );
        ui->textBrowserInfoOut->moveCursor(QTextCursor::End);
        return true;
    }
    QMessageBox::warning(this, tr("Warning"), tr("FFmpeg did not fount"), QMessageBox::Ok);
    ui->textBrowserInfoOut->insertPlainText(
                "----------------------------------------------\n"
                "             ffmpeg check failed              \n"
                "----------------------------------------------\n"
                );
    ui->textBrowserInfoOut->moveCursor(QTextCursor::End);
    return  false;
}

QString MainWindow::formatSuffixList(const QString &&type, QString str)
{
    return type+"( *."+str.replace(QRegExp(":")," *.")+");;";
}

void MainWindow::initAnime4K(Anime4K *&anime4K)
{
    int passes = ui->spinBoxPasses->value();
    int pushColorCount = ui->spinBoxPushColorCount->value();
    double pushColorStrength = ui->doubleSpinBoxPushColorStrength->value();
    double pushGradientStrength = ui->doubleSpinBoxPushGradientStrength->value();
    double zoomFactor = ui->doubleSpinBoxZoomFactor->value();
    bool fastMode = ui->checkBoxFastMode->isChecked();
    bool videoMode = false;
    bool preprocessing = ui->checkBoxEnablePreprocessing->isChecked();
    bool postprocessing = ui->checkBoxEnablePostprocessing->isChecked();
    unsigned int threads = ui->spinBoxThreads->value();
    uint8_t prefilters=0;
    if (preprocessing)
    {
        if (ui->checkBoxPreMedian->isChecked())
            prefilters|=1;
        if (ui->checkBoxPreMean->isChecked())
            prefilters|=2;
        if (ui->checkBoxPreCAS->isChecked())
            prefilters|=4;
        if (ui->checkBoxPreGaussianWeak->isChecked())
            prefilters|=8;
        if (ui->checkBoxPreGaussian->isChecked())
            prefilters|=16;
        if (ui->checkBoxPreBilateral->isChecked())
            prefilters|=32;
        if (ui->checkBoxPreBilateralFaster->isChecked())
            prefilters|=64;
    }
    uint8_t postfilters=0;
    if (postprocessing)
    {
        if (ui->checkBoxPostMedian->isChecked())
            postfilters|=1;
        if (ui->checkBoxPostMean->isChecked())
            postfilters|=2;
        if (ui->checkBoxPostCAS->isChecked())
            postfilters|=4;
        if (ui->checkBoxPostGaussianWeak->isChecked())
            postfilters|=8;
        if (ui->checkBoxPostGaussian->isChecked())
            postfilters|=16;
        if (ui->checkBoxPostBilateral->isChecked())
            postfilters|=32;
        if (ui->checkBoxPostBilateralFaster->isChecked())
            postfilters|=64;
    }
    if(ui->checkBoxGPUMode->isChecked())
        anime4K = mainAnime4kGPU;
    else
        anime4K = mainAnime4kCPU;
    anime4K->setArguments(passes,
                          pushColorCount,
                          pushColorStrength,
                          pushGradientStrength,
                          zoomFactor,
                          fastMode,
                          videoMode,
                          preprocessing,
                          postprocessing,
                          prefilters,
                          postfilters,
                          threads);
}

inline void MainWindow::releaseMainAnime4K()
{
    delete mainAnime4kCPU;
    if(mainAnime4kGPU!=nullptr)
        delete mainAnime4kGPU;
}

FileType MainWindow::fileType(const QFileInfo &file)
{
    QString imageSuffix = ui->lineEditImageSuffix->text();
    QString videoSuffix = ui->lineEditVideoSuffix->text();
    if (imageSuffix.contains(file.suffix(), Qt::CaseInsensitive))
        return IMAGE;
    if (videoSuffix.contains(file.suffix(), Qt::CaseInsensitive))
        return VIDEO;
    return ERROR_TYPE;
}

QString MainWindow::getOutputPrefix()
{
    QString prefix = ui->lineEditOutputPrefix->text();
    if (prefix.isEmpty())
        return "output_anime4kcpp_";
    if (prefix.contains(QRegExp("[^\\x00-\\xff]")))
    {
        errorHandler(INPUT_NONASCII);
        return "output_anime4kcpp_";
    }
    return ui->lineEditOutputPrefix->text();
}

void MainWindow::solt_done_renewState(int row, double pro, quint64 time)
{
    tableModel->setData(tableModel->index(row, 4), tr("done"), Qt::DisplayRole);
    ui->progressBarProcessingList->setValue(pro*100);
    ui->textBrowserInfoOut->insertPlainText(QString("processing time: %1 s\ndone\n").arg(time/1000.0));
    ui->textBrowserInfoOut->moveCursor(QTextCursor::End);
    totalTime += time;
}

void MainWindow::solt_error_renewState(int row, QString err)
{
    tableModel->setData(tableModel->index(row, 4), tr("error"), Qt::DisplayRole);
    QMessageBox::information(this,
                             tr("error"),
                             err,
                             QMessageBox::Ok);
}

void MainWindow::solt_allDone_remindUser()
{
    QMessageBox::information(this,
                             tr("Notice"),
                             QString("All tasks done\nTotal processing time: %1 s").arg(totalTime/1000.0),
                             QMessageBox::Ok);
    totalTime = 0;
    ui->tableViewProcessingList->setEnabled(true);
    ui->progressBarProcessingList->setEnabled(false);
    ui->progressBarProcessingList->reset();
    ui->pushButtonInputPath->setEnabled(true);
    ui->pushButtonDelete->setEnabled(true);
    ui->pushButtonClear->setEnabled(true);
    ui->pushButtonStart->setEnabled(true);
    on_pushButtonClear_clicked();
}

void MainWindow::solt_showInfo_renewTextBrowser(std::string info)
{
    ui->textBrowserInfoOut->insertPlainText(QString::fromStdString(info));
    ui->textBrowserInfoOut->moveCursor(QTextCursor::End);
}

void MainWindow::on_actionQuit_triggered()
{
    this->close();
}

void MainWindow::on_pushButtonInputPath_clicked()
{
    QStringList files = QFileDialog::getOpenFileNames(this, tr("pick files"), "./",
                                                      formatSuffixList(tr("image"),ui->lineEditImageSuffix->text())+
                                                      formatSuffixList(tr("video"),ui->lineEditVideoSuffix->text()));
    files.removeDuplicates();

    QStandardItem *inputFile;
    QStandardItem *outputFile;
    QStandardItem *inputPath;
    QStandardItem *outputPath;
    QStandardItem *state;


    bool flag=false;
    for(QString &file:files)
    {
        if(file.contains(QRegExp("[^\\x00-\\xff]")))
        {
            flag = true;
            continue;
        }

        QFileInfo fileInfo(file);

        if (fileType(fileInfo)==ERROR_TYPE){
            errorHandler(TYPE_NOT_ADD);
            continue;
        }

        inputFile = new QStandardItem(fileInfo.fileName());
        if(fileType(fileInfo)==VIDEO)
            outputFile = new QStandardItem(getOutputPrefix()+fileInfo.baseName()+".mp4");
        else
            outputFile = new QStandardItem(getOutputPrefix()+fileInfo.fileName());
        inputPath = new QStandardItem(fileInfo.filePath());
        state = new QStandardItem(tr("ready"));
        if (ui->lineEditOutputPath->text().isEmpty())
            outputPath = new QStandardItem(QDir::currentPath());
        else
            outputPath = new QStandardItem(ui->lineEditOutputPath->text());
        tableModel->appendRow({inputFile,outputFile,inputPath,outputPath,state});

        totalTaskCount++;
    }

    ui->labelTotalTaskCount->setText(QString("Total: %1 ").arg(totalTaskCount));

    if (flag)
    {
        errorHandler(INPUT_NONASCII);
    }


}

void MainWindow::on_pushButtonOutputPath_clicked()
{
    ui->lineEditOutputPath->setText(QFileDialog::getExistingDirectory(this,tr("output directory"),"./"));
}

void MainWindow::on_pushButtonClear_clicked()
{
    tableModel->removeRows(0, tableModel->rowCount());

    totalTaskCount = 0;
    ui->labelTotalTaskCount->setText("Total: 0 ");
}

void MainWindow::on_pushButtonDelete_clicked()
{
    tableModel->removeRow(ui->tableViewProcessingList->currentIndex().row());

    totalTaskCount--;
    ui->labelTotalTaskCount->setText(QString("Total: %1 ").arg(totalTaskCount));
}

void MainWindow::on_radioButtonFast_clicked()
{
    ui->spinBoxPasses->setValue(1);
    ui->spinBoxPushColorCount->setValue(2);
    ui->spinBoxThreads->setValue(std::thread::hardware_concurrency());
    ui->doubleSpinBoxPushColorStrength->setValue(0.3);
    ui->doubleSpinBoxPushGradientStrength->setValue(1.0);
    ui->doubleSpinBoxZoomFactor->setValue(2.0);
    ui->checkBoxFastMode->setChecked(true);
    ui->checkBoxEnablePreprocessing->setChecked(false);
    ui->checkBoxEnablePostprocessing->setChecked(false);
}

void MainWindow::on_radioButtonBalance_clicked()
{
    ui->spinBoxPasses->setValue(2);
    ui->spinBoxPushColorCount->setValue(2);
    ui->spinBoxThreads->setValue(std::thread::hardware_concurrency());
    ui->doubleSpinBoxPushColorStrength->setValue(0.3);
    ui->doubleSpinBoxPushGradientStrength->setValue(1.0);
    ui->doubleSpinBoxZoomFactor->setValue(2.0);
    ui->checkBoxFastMode->setChecked(false);
    ui->checkBoxEnablePreprocessing->setChecked(false);
    ui->checkBoxEnablePostprocessing->setChecked(false);
}

void MainWindow::on_radioButtonQuality_clicked()
{
    ui->spinBoxPasses->setValue(2);
    ui->spinBoxPushColorCount->setValue(2);
    ui->spinBoxThreads->setValue(std::thread::hardware_concurrency());
    ui->doubleSpinBoxPushColorStrength->setValue(0.3);
    ui->doubleSpinBoxPushGradientStrength->setValue(1.0);
    ui->doubleSpinBoxZoomFactor->setValue(2.0);
    ui->checkBoxFastMode->setChecked(false);
    //preprocessing
    ui->checkBoxEnablePreprocessing->setChecked(true);
    ui->checkBoxPreMedian->setChecked(false);
    ui->checkBoxPreMean->setChecked(false);
    ui->checkBoxPreCAS->setChecked(true);
    ui->checkBoxPreGaussianWeak->setChecked(false);
    ui->checkBoxPreGaussian->setChecked(false);
    ui->checkBoxPreBilateral->setChecked(false);
    ui->checkBoxPreBilateralFaster->setChecked(false);
    //postprocessing
    ui->checkBoxEnablePostprocessing->setChecked(true);
    ui->checkBoxPostMedian->setChecked(false);
    ui->checkBoxPostMean->setChecked(false);
    ui->checkBoxPostCAS->setChecked(false);
    ui->checkBoxPostGaussianWeak->setChecked(true);
    ui->checkBoxPostGaussian->setChecked(false);
    ui->checkBoxPostBilateral->setChecked(true);
    ui->checkBoxPostBilateralFaster->setChecked(false);
}

void MainWindow::on_checkBoxEnablePreprocessing_stateChanged(int arg1)
{
    if (arg1==Qt::CheckState::Checked)
    {
        ui->checkBoxPreCAS->setEnabled(true);
        ui->checkBoxPreMean->setEnabled(true);
        ui->checkBoxPreMedian->setEnabled(true);
        ui->checkBoxPreGaussianWeak->setEnabled(true);
        ui->checkBoxPreGaussian->setEnabled(true);
        ui->checkBoxPreBilateral->setEnabled(true);
        ui->checkBoxPreBilateralFaster->setEnabled(true);
    }
    else
    {
        ui->checkBoxPreCAS->setEnabled(false);
        ui->checkBoxPreMean->setEnabled(false);
        ui->checkBoxPreMedian->setEnabled(false);
        ui->checkBoxPreGaussianWeak->setEnabled(false);
        ui->checkBoxPreGaussian->setEnabled(false);
        ui->checkBoxPreBilateral->setEnabled(false);
        ui->checkBoxPreBilateralFaster->setEnabled(false);
    }
}

void MainWindow::on_checkBoxEnablePostprocessing_stateChanged(int arg1)
{
    if (arg1==Qt::CheckState::Checked)
    {
        ui->checkBoxPostCAS->setEnabled(true);
        ui->checkBoxPostMean->setEnabled(true);
        ui->checkBoxPostMedian->setEnabled(true);
        ui->checkBoxPostGaussianWeak->setEnabled(true);
        ui->checkBoxPostGaussian->setEnabled(true);
        ui->checkBoxPostBilateral->setEnabled(true);
        ui->checkBoxPostBilateralFaster->setEnabled(true);
    }
    else
    {
        ui->checkBoxPostCAS->setEnabled(false);
        ui->checkBoxPostMean->setEnabled(false);
        ui->checkBoxPostMedian->setEnabled(false);
        ui->checkBoxPostGaussianWeak->setEnabled(false);
        ui->checkBoxPostGaussian->setEnabled(false);
        ui->checkBoxPostBilateral->setEnabled(false);
        ui->checkBoxPostBilateralFaster->setEnabled(false);
    }
}

void MainWindow::on_pushButtonPreview_clicked()
{
    QFileInfo previewFile(ui->lineEditPreview->text());
    if (!previewFile.exists())
    {
        errorHandler(FILE_NOT_EXIST);
        return;
    }

    ui->pushButtonPreview->setEnabled(false);

    Anime4K *anime4k;
    initAnime4K(anime4k);
    switch (fileType(previewFile))
    {
    case IMAGE:
        anime4k->setVideoMode(false);
        anime4k->loadImage(previewFile.filePath().toStdString());
        anime4k->process();
        anime4k->showImage();
        break;
    case VIDEO:
        errorHandler(TYPE_NOT_IMAGE);
        break;
    case ERROR_TYPE:
        errorHandler(TYPE_NOT_ADD);
        break;
    }

    ui->pushButtonPreview->setEnabled(true);
}

void MainWindow::on_pushButtonPreviewPick_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("pick files"), "./",
                                                    formatSuffixList(tr("image"),ui->lineEditImageSuffix->text())+
                                                    formatSuffixList(tr("video"),ui->lineEditVideoSuffix->text()));
    if (fileName.contains(QRegExp("[^\\x00-\\xff]")))
    {
        errorHandler(INPUT_NONASCII);
        return;
    }
    ui->lineEditPreview->setText(fileName);
}

void MainWindow::on_pushButtonStart_clicked()
{
    if(ui->checkBoxGPUMode->isChecked() && (ui->checkBoxEnablePreprocessing->isChecked() || ui->checkBoxEnablePostprocessing->isChecked()))
    {
        if (QMessageBox::Yes == QMessageBox::information(this,
                                 tr("Notice"),
                                 tr("You are using GPU acceleration but still enabled"
                                    "preprocessing or postprocessing, which is not GPU acceletation yet, "
                                    "and may slow down processing for GPU (usually still faster than CPU), close them?"),
                                 QMessageBox::Yes | QMessageBox::No,
                                 QMessageBox::Yes))
        {
            ui->checkBoxEnablePreprocessing->setChecked(false);
            ui->checkBoxEnablePostprocessing->setChecked(false);
        }
    }

    int rows = tableModel->rowCount();
    if(!rows)
    {
        errorHandler(PROCESSING_LIST_EMPTY);
        return;
    }

    ui->pushButtonStart->setEnabled(false);
    ui->pushButtonClear->setEnabled(false);
    ui->pushButtonDelete->setEnabled(false);
    ui->pushButtonInputPath->setEnabled(false);
    ui->progressBarProcessingList->setEnabled(true);
    ui->tableViewProcessingList->setEnabled(false);

    QtConcurrent::run([this, rows](){
        //QList<<input_path,outpt_path>,row>
        QList<QPair<QPair<QString,QString>,int>> images;
        QList<QPair<QPair<QString,QString>,int>> videos;
        //read info
        {
            QDir outputPathMaker;
            QString outputPath;
            QString outputFileName;
            for(int i=0;i<rows;i++)
            {
                QFileInfo fileInfo(tableModel->index(i,2).data().toString());
                outputPathMaker.setPath(tableModel->index(i,3).data().toString());
                outputPath = outputPathMaker.absolutePath();
                outputFileName =  tableModel->index(i,1).data().toString();

                outputPathMaker.mkpath(outputPath);

                if (fileType(fileInfo)==IMAGE)
                {
                    images<<QPair<QPair<QString,QString>,int>(QPair<QString,QString>(fileInfo.filePath(),
                                                                                     outputPath+"/"+
                                                                                     outputFileName),i);
                    imageCount++;
                }
                else
                {
                    videos<<QPair<QPair<QString,QString>,int>(QPair<QString,QString>(fileInfo.filePath(),
                                                                                     outputPath+"/"+
                                                                                     outputFileName),i);
                    videoCount++;
                }
            }
        }

        double total = imageCount + videoCount;

        Communicator cm;
        connect(&cm,SIGNAL(done(int, double, quint64)),this,SLOT(solt_done_renewState(int, double, quint64)));
        connect(&cm,SIGNAL(error(int, QString)),this,SLOT(solt_error_renewState(int, QString)));
        connect(&cm,SIGNAL(showInfo(std::string)),this,SLOT(solt_showInfo_renewTextBrowser(std::string)));
        connect(&cm,SIGNAL(allDone()),this,SLOT(solt_allDone_remindUser()));

        Anime4K *anime4k;
        initAnime4K(anime4k);
        emit cm.showInfo(anime4k->getFiltersInfo());

        std::chrono::steady_clock::time_point startTime,endTime;

        if (imageCount)
        {
            anime4k->setVideoMode(false);
            for (QPair<QPair<QString,QString>,int> const &image: images)
            {
                try
                {
                    anime4k->loadImage(image.first.first.toStdString());
                    emit cm.showInfo(anime4k->getInfo()+"processing...\n");
                    startTime = std::chrono::steady_clock::now();
                    anime4k->process();
                    endTime = std::chrono::steady_clock::now();
                    anime4k->saveImage(image.first.second.toStdString());
                }
                catch (const char* err)
                {
                    emit cm.error(image.second,QString(err));
                }

                imageCount--;

                emit cm.done(image.second, 1.0-((imageCount + videoCount) / total),
                             std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count());

            }
        }
        if (videoCount)
        {
            anime4k->setVideoMode(true);
            for (QPair<QPair<QString,QString>,int> const &video: videos)
            {
                try
                {
                    anime4k->loadVideo(video.first.first.toStdString());
                    anime4k->setVideoSaveInfo(video.first.second.toStdString()+"_tmp_out.mp4");
                    emit cm.showInfo(anime4k->getInfo()+"processing...\n");
                    startTime = std::chrono::steady_clock::now();
                    anime4k->process();
                    endTime = std::chrono::steady_clock::now();
                    anime4k->saveVideo();
                }
                catch (const char* err)
                {
                    emit cm.error(video.second,QString(err));
                }

                if(ffmpeg)
                {
                    if(!QProcess::execute("ffmpeg -i \"" + video.first.second + "_tmp_out.mp4""\" -i \"" + video.first.first + "\" -c copy -map 0 -map 1:1 -y \"" + video.first.second + "\""))
                    {
#ifdef _WIN32
                        const char* command = ("del /q \"" + QDir::toNativeSeparators(video.first.second + "_tmp_out.mp4\"")).toLatin1();
#elif defined(__linux)
                        const char* command = ("rm \"" + QDir::toNativeSeparators(video.first.second + "_tmp_out.mp4\"")).toLatin1();
#endif // CURRENT SYSTEM
                        system(command);
                    }
                }

                videoCount--;

                emit cm.done(video.second, 1.0-(videoCount / total),
                             std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count());

            }
        }

        emit cm.allDone();
    });

}

void MainWindow::on_actionAbout_triggered()
{
    QMessageBox::information(this,
                             tr("About"),
                             QString("Anime4KCPP GUI\n\n"
                                     "Anime4K GUI v%1\n"
                                     "Anime4K Core v%2\n\n"
                                     "Copyright (c) 2020 TianZerL").arg(VERSION, CORE_VERSION),
                             QMessageBox::Ok);
}

void MainWindow::on_tabWidgetMain_tabBarClicked(int index)
{
    if (index == 1)
        ui->radioButtonCustom->setChecked(true);
}

void MainWindow::on_actionChinese_triggered()
{
    translator->load("./language/Anime4KCPP_GUI_zh_CN.qm");
    qApp->installTranslator(translator);
    ui->retranslateUi(this);
    currLanguage=zh_cn;
}

void MainWindow::on_actionEnglish_triggered()
{
    qApp->removeTranslator(translator);
    ui->retranslateUi(this);
    currLanguage=en;
}

void MainWindow::on_pushButtonClearText_clicked()
{
    ui->textBrowserInfoOut->clear();
    initTextBrowser();
}

void MainWindow::on_spinBoxFontSize_valueChanged(int value)
{
    ui->textBrowserInfoOut->setFont(QFont(ui->fontComboBox->font().family(),value));
}

void MainWindow::on_fontComboBox_currentFontChanged(const QFont &font)
{
    ui->textBrowserInfoOut->setFont(QFont(font.family(),ui->spinBoxFontSize));
}

void MainWindow::on_pushButtonCopyText_clicked()
{
   QApplication::clipboard()->setText(ui->textBrowserInfoOut->toPlainText());
   QMessageBox::information(this,
                            tr("Notice"),
                            tr("Log has been copied to the clipboard"),
                            QMessageBox::Ok);
}

void MainWindow::on_pushButtonPreviewOrgin_clicked()
{
    QString filePath =  ui->lineEditPreview->text();
    if (filePath.isEmpty())
    {
        errorHandler(FILE_NOT_EXIST);
        return;
    }
    QPixmap orginImage(filePath);
    QWidget *orginImageWidget = new QWidget(this,Qt::Window);
    orginImageWidget->setAttribute(Qt::WA_DeleteOnClose);
    QLabel *orginImageLable = new QLabel(orginImageWidget);
    orginImageLable->setPixmap(orginImage);
    orginImageWidget->setWindowTitle("orgin image");
    orginImageWidget->setFixedSize(orginImage.size());
    orginImageWidget->show();
}

void MainWindow::on_pushButtonPreviewOnlyResize_clicked()
{
    QString filePath =  ui->lineEditPreview->text();
    if (filePath.isEmpty())
    {
        errorHandler(FILE_NOT_EXIST);
        return;
    }
    //read image by opencv for resizing by CUBIC
    double factor = ui->doubleSpinBoxZoomFactor->value();
    cv::Mat orgImg = cv::imread(filePath.toStdString(), cv::IMREAD_COLOR);
    cv::resize(orgImg, orgImg, cv::Size(0,0), factor,factor, cv::INTER_CUBIC);
    //convert to QImage
    QImage orginImage(orgImg.data, orgImg.cols, orgImg.rows, QImage::Format_BGR888);
    QPixmap resizedImage(QPixmap::fromImage(orginImage));
    //show
    QWidget *resizedImageWidget = new QWidget(this,Qt::Window);
    resizedImageWidget->setAttribute(Qt::WA_DeleteOnClose);
    QLabel *resizedImageLable = new QLabel(resizedImageWidget);
    resizedImageLable->setPixmap(resizedImage);
    resizedImageWidget->setWindowTitle("resized image");
    resizedImageWidget->setFixedSize(resizedImage.size());
    resizedImageWidget->show();
}

void MainWindow::on_pushButtonPickFolder_clicked()
{
    QString folderPath = QFileDialog::getExistingDirectory(this,tr("output directory"),"./");
    QDir folder(folderPath);
    QFileInfoList fileInfoList = folder.entryInfoList(QDir::Files);

    QStandardItem *inputFile;
    QStandardItem *outputFile;
    QStandardItem *inputPath;
    QStandardItem *outputPath;
    QStandardItem *state;

    for (QFileInfo &fileInfo : fileInfoList)
    {
        if (!fileInfo.fileName().contains(QRegExp("[^\\x00-\\xff]")) && (fileType(fileInfo) != ERROR_TYPE))
        {
            inputFile = new QStandardItem(fileInfo.fileName());
            if(fileType(fileInfo)==VIDEO)
                outputFile = new QStandardItem(getOutputPrefix()+fileInfo.baseName()+".mp4");
            else
                outputFile = new QStandardItem(getOutputPrefix()+fileInfo.fileName());
            inputPath = new QStandardItem(fileInfo.filePath());
            state = new QStandardItem(tr("ready"));
            if (ui->lineEditOutputPath->text().isEmpty())
                outputPath = new QStandardItem(QDir::currentPath());
            else
                outputPath = new QStandardItem(ui->lineEditOutputPath->text());
            tableModel->appendRow({inputFile,outputFile,inputPath,outputPath,state});

            totalTaskCount++;
        }
    }

    ui->labelTotalTaskCount->setText(QString("Total: %1 ").arg(totalTaskCount));

}

void MainWindow::on_checkBoxGPUMode_stateChanged(int state)
{
    if((state == Qt::Checked) && (GPU == GPUMODE_UNINITIALZED))
    {
        if(QMessageBox::Yes == QMessageBox::information(this,
                                 tr("Notice"),
                                 tr("You are trying to enable GPU acceleration, "
                                    "which is an experimental function, check and inital GPU?"),
                                 QMessageBox::Yes | QMessageBox::No,
                                 QMessageBox::No))
        {
            std::pair<bool,std::string> ret = Anime4KGPU::checkGPUSupport();
            if(!ret.first)
            {
                QMessageBox::warning(this,
                                     tr("Warning"),
                                     QString::fromStdString(ret.second),
                                     QMessageBox::Ok);
                GPU = GPUMODE_UNSUPPORT;
                ui->checkBoxGPUMode->setCheckState(Qt::Unchecked);
            }
            else
            {
                mainAnime4kGPU = new Anime4KGPU;
                GPU = GPUMODE_INITIALZED;
                QMessageBox::information(this,
                                     tr("Notice"),
                                     "Inital successful!\n" +
                                     QString::fromStdString(ret.second),
                                     QMessageBox::Ok);
                ui->textBrowserInfoOut->insertPlainText("GPU inital successful!\n" + QString::fromStdString(ret.second) + "\n");
                ui->textBrowserInfoOut->moveCursor(QTextCursor::End);
            }
        }
        else
        {
            ui->checkBoxGPUMode->setCheckState(Qt::Unchecked);
        }
    }
    else if((state == Qt::Checked) && (GPU == GPUMODE_UNSUPPORT))
    {
        QMessageBox::warning(this,
                             tr("Warning"),
                             tr("Unsupport GPU acceleration in this platform"),
                             QMessageBox::Ok);
        ui->checkBoxGPUMode->setCheckState(Qt::Unchecked);
    }
}
