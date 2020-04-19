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
    initTextBrowser();
    //accept drops
    this->setAcceptDrops(true);
    //set quality check
    ui->radioButtonQuality->click();
    //inital tableView
    tableModel = new QStandardItemModel(this);
    tableModel->setColumnCount(5);
    tableModel->setHorizontalHeaderLabels({tr("Input file"),
                                           tr("Output file"),
                                           tr("full path"),
                                           tr("output path"),
                                           tr("State")});
    ui->tableViewProcessingList->setModel(tableModel);
    //inital suffix processing
    ui->lineEditImageSuffix->setText("png:jpg:jpeg:bmp");
    ui->lineEditVideoSuffix->setText("mp4:mkv:avi:m4v:flv:3gp:wmv:mov");
    //inital processBar
    ui->progressBarProcessingList->reset();
    ui->progressBarProcessingList->setRange(0, 100);
    ui->progressBarProcessingList->setEnabled(false);
    //inital arguments
    ui->spinBoxThreads->setMinimum(1);
    ui->doubleSpinBoxPushColorStrength->setRange(0.0,1.0);
    ui->doubleSpinBoxPushGradientStrength->setRange(0.0,1.0);
    ui->doubleSpinBoxZoomFactor->setRange(1.0,10.0);
    //inital mutex
    mutex = new QMutex;
    //inital time and count
    totalTime = imageCount = videoCount = 0;
    //inital ffmpeg
    ffmpeg = checkFFmpeg();
    //Register
    qRegisterMetaType<std::string>("std::string");
}

MainWindow::~MainWindow()
{
    delete ui;
    delete mutex;
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    if (QMessageBox::Yes == QMessageBox::warning(this, tr("Confirm"),
                                                 tr("Do you really want to exit?"),
                                                 QMessageBox::Yes|QMessageBox::No, QMessageBox::No))
    {
        event->accept();
    }
    else
    {
        event->ignore();
    }
}

void MainWindow::dragEnterEvent(QDragEnterEvent *event)
{
    event->acceptProposedAction();
}

void MainWindow::dropEvent(QDropEvent *event)
{
    QString file = event->mimeData()->urls().first().toLocalFile();
    QFileInfo fileInfo(file);

    if (fileType(fileInfo)==ERROR){
        QMessageBox::information(this,
                                 tr("Error"),
                                 tr("File type error, you can add it manually"),
                                 QMessageBox::Ok);
        return;
    }

    QStandardItem *inputFile;
    QStandardItem *outputFile;
    QStandardItem *inputPath;
    QStandardItem *outputPath;
    QStandardItem *state;


    inputFile = new QStandardItem(fileInfo.fileName());
    if(fileType(fileInfo)==VIDEO)
        outputFile = new QStandardItem("output_anime4kcpp_"+fileInfo.baseName()+".mp4");
    else
        outputFile = new QStandardItem("output_anime4kcpp_"+fileInfo.fileName());
    inputPath = new QStandardItem(fileInfo.filePath());
    state = new QStandardItem(tr("ready"));
    if (ui->lineEditOutputPath->text().isEmpty())
        outputPath = new QStandardItem(QDir::currentPath());
    else
        outputPath = new QStandardItem(ui->lineEditOutputPath->text());

    tableModel->appendRow({inputFile,outputFile,inputPath,outputPath,state});
}

inline void MainWindow::initTextBrowser()
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

inline bool MainWindow::checkFFmpeg()
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
    anime4K = new Anime4K(passes,
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

void MainWindow::releaseAnime4K(Anime4K *&anime4K)
{
    delete anime4K;
}

FileType MainWindow::fileType(QFileInfo &file)
{
    QString imageSuffix = ui->lineEditImageSuffix->text();
    QString videoSuffix = ui->lineEditVideoSuffix->text();
    if (imageSuffix.contains(file.suffix(), Qt::CaseInsensitive))
        return IMAGE;
    if (videoSuffix.contains(file.suffix(), Qt::CaseInsensitive))
        return VIDEO;
    return ERROR;
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
    ui->pushButtonStart->setEnabled(true);
    ui->progressBarProcessingList->setEnabled(false);
    ui->progressBarProcessingList->reset();
    tableModel->clear();
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
    QStandardItem *inputFile;
    QStandardItem *outputFile;
    QStandardItem *inputPath;
    QStandardItem *outputPath;
    QStandardItem *state;
    for(QString &file:files)
    {
        QFileInfo fileInfo(file);

        if (fileType(fileInfo)==ERROR){
            QMessageBox::information(this,
                                     tr("Error"),
                                     tr("File type error, you can add it manually"),
                                     QMessageBox::Ok);
            continue;
        }

        inputFile = new QStandardItem(fileInfo.fileName());
        if(fileType(fileInfo)==VIDEO)
            outputFile = new QStandardItem("output_anime4kcpp_"+fileInfo.baseName()+".mp4");
        else
            outputFile = new QStandardItem("output_anime4kcpp_"+fileInfo.fileName());
        inputPath = new QStandardItem(fileInfo.filePath());
        state = new QStandardItem(tr("ready"));
        if (ui->lineEditOutputPath->text().isEmpty())
            outputPath = new QStandardItem(QDir::currentPath());
        else
            outputPath = new QStandardItem(ui->lineEditOutputPath->text());
        tableModel->appendRow({inputFile,outputFile,inputPath,outputPath,state});
    }

}

void MainWindow::on_pushButtonOutputPath_clicked()
{
    ui->lineEditOutputPath->setText(QFileDialog::getExistingDirectory(this,tr("output directory"),"./"));
}

void MainWindow::on_pushButtonClear_clicked()
{
    tableModel->clear();
}

void MainWindow::on_pushButtonDelete_clicked()
{
    tableModel->removeRow(ui->tableViewProcessingList->currentIndex().row());
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
    ui->checkBoxEnablePreprocessing->setChecked(true);
    ui->checkBoxPreCAS->setChecked(true);
    ui->checkBoxEnablePostprocessing->setChecked(true);
    ui->checkBoxPostGaussianWeak->setChecked(true);
    ui->checkBoxPostBilateral->setChecked(true);
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
        QMessageBox::information(this,
                                 tr("Error"),
                                 tr("File does not exists"),
                                 QMessageBox::Ok);
        return;
    }

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
        QMessageBox::information(this,
                                 tr("Error"),
                                 tr("File type error, only image support preview"),
                                 QMessageBox::Ok);
        break;
    case ERROR:
        QMessageBox::information(this,
                                 tr("Error"),
                                 tr("File type error, you can add it manually"),
                                 QMessageBox::Ok);
        break;
    }

    releaseAnime4K(anime4k);
}

void MainWindow::on_pushButtonPreviewPick_clicked()
{
    ui->lineEditPreview->setText(QFileDialog::getOpenFileName(this, tr("pick files"), "./",
                                                              formatSuffixList(tr("image"),ui->lineEditImageSuffix->text())+
                                                              formatSuffixList(tr("video"),ui->lineEditVideoSuffix->text()))
                                 );
}

void MainWindow::on_pushButtonStart_clicked()
{
    int rows = tableModel->rowCount();
    if(!rows)
    {
        QMessageBox::information(this,
                                 tr("Error"),
                                 tr("Processing list empty"),
                                 QMessageBox::Ok);
        return;
    }

    ui->pushButtonStart->setEnabled(false);
    ui->progressBarProcessingList->setEnabled(true);

    QtConcurrent::run([this, rows](){
        QList<QPair<QPair<QString,QString>,int>> images;
        QList<QPair<QPair<QString,QString>,int>> videos;

        for(int i=0;i<rows;i++)
        {
            QFileInfo fileInfo(tableModel->index(i,2).data().toString());
            if (fileType(fileInfo)==IMAGE)
            {
                images<<QPair<QPair<QString,QString>,int>(QPair<QString,QString>(fileInfo.filePath(),
                                                                                 tableModel->index(i,3).data().toString()+"/"+
                                                                                 tableModel->index(i,1).data().toString()),i);
                imageCount++;
            }
            else
            {
                videos<<QPair<QPair<QString,QString>,int>(QPair<QString,QString>(fileInfo.filePath(),
                                                                                 tableModel->index(i,3).data().toString()+"/"+
                                                                                 tableModel->index(i,1).data().toString()),i);
                videoCount++;
            }
        }

        double imageTotal = imageCount;
        double videoTotal = videoCount;

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

                emit cm.done(image.second, 1.0-((imageCount-1)/imageTotal),
                             std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count());

                {
                    QMutexLocker locker(mutex);
                    imageCount--;
                }

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
                    anime4k->setVideoSaveInfo("tmp_out.mp4");
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
                    if(!QProcess::execute("ffmpeg -i \"tmp_out.mp4\" -i \"" + video.first.first + "\" -c copy -map 0 -map 1:1 -y \"" + video.first.second + "\""))
                    {
#ifdef _WIN32
                        const char* command = "del /q tmp_out.mp4";
#elif defined(__linux)
                        const char* command = "rm tmp_out.mp4";
#endif // SYSTEM
                        system(command);
                    }
                }

                emit cm.done(video.second, 1.0-((videoCount-1)/videoTotal),
                             std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count());

                {
                    QMutexLocker locker(mutex);
                    videoCount--;
                }

            }
        }

        releaseAnime4K(anime4k);
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
}

void MainWindow::on_actionEnglish_triggered()
{
    qApp->removeTranslator(translator);
    ui->retranslateUi(this);
}
