#include <QDateTime>
#include <QDesktopServices>
#include <QFile>
#include <QFileDialog>
#include <QList>
#include <QMimeData>
#include <QMimeDatabase>
#include <QMessageBox>
#include <QStyleFactory>
#include <QSharedPointer>

#include "Config.hpp"
#include "Logger.hpp"
#include "Upscaler.hpp"
#include "MainWindow.hpp"
#include "ui_MainWindow.h"

MainWindow::MainWindow() : QMainWindow(nullptr), ui(std::make_unique<Ui::MainWindow>())
{
    ui->setupUi(this);
    init();
}

MainWindow::~MainWindow() noexcept = default;

void MainWindow::init()
{
    qApp->setStyle(QStyleFactory::create(gConfig.gui.styleName));
    qApp->setPalette(qApp->style()->standardPalette());

    QObject::connect(ui->action_exit, &QAction::triggered, this, &QMainWindow::close);

    for (auto&& style: QStyleFactory::keys())
    {
        auto action = new QAction{ style, ui->menu_settings_style };
        QObject::connect(action, &QAction::triggered, this, [=]() {
            gConfig.gui.styleName = action->text();
            qApp->setStyle(QStyleFactory::create(gConfig.gui.styleName));
            qApp->setPalette(qApp->style()->standardPalette());
        });
        ui->menu_settings_style->addAction(action);
    }

    ui->action_settings_exit_confirmation->setChecked(gConfig.gui.exitConfirmation);
    QObject::connect(ui->action_settings_exit_confirmation, &QAction::toggled, this,
        [](const bool checked) { gConfig.gui.exitConfirmation = checked; });

    ui->line_edit_suffix_image->setText(gConfig.io.imageSuffix);
    ui->line_edit_suffix_video->setText(gConfig.io.videoSuffix);
    QObject::connect(ui->line_edit_suffix_image, &QLineEdit::textChanged, this,
        [](const QString& value) { gConfig.io.imageSuffix = value; });
    QObject::connect(ui->line_edit_suffix_video, &QLineEdit::textChanged, this,
        [](const QString& value) { gConfig.io.videoSuffix = value; });

    ui->line_edit_prefix_image->setText(gConfig.io.imagePrefix);
    ui->line_edit_prefix_video->setText(gConfig.io.videoPrefix);
    QObject::connect(ui->line_edit_prefix_image, &QLineEdit::textChanged, this,
        [](const QString& value) { gConfig.io.imagePrefix = value; });
    QObject::connect(ui->line_edit_prefix_video, &QLineEdit::textChanged, this,
        [](const QString& value) { gConfig.io.videoPrefix = value; });

    ui->line_edit_output_path_image->setText(gConfig.io.imageOutputPath);
    ui->line_edit_output_path_video->setText(gConfig.io.videoOutputPath);
    QObject::connect(ui->line_edit_output_path_image, &QLineEdit::textChanged, this,
        [](const QString& value) { gConfig.io.imageOutputPath = value; });
    QObject::connect(ui->line_edit_output_path_video, &QLineEdit::textChanged, this,
        [](const QString& value) { gConfig.io.videoOutputPath = value; });
    QObject::connect(ui->push_button_output_path_image_select, &QPushButton::clicked, this, [=]() {
            auto path = MainWindow::selectDir();
            if (!path.isEmpty()) ui->line_edit_output_path_image->setText(path);
    });
    QObject::connect(ui->push_button_output_path_video_select, &QPushButton::clicked, this, [=]() {
            auto path = MainWindow::selectDir();
            if (!path.isEmpty()) ui->line_edit_output_path_video->setText(path);
    });
    QObject::connect(ui->push_button_output_path_image_open, &QPushButton::clicked, this,
        []() { MainWindow::openDir(gConfig.io.imageOutputPath); });
    QObject::connect(ui->push_button_output_path_video_open, &QPushButton::clicked, this,
        []() { MainWindow::openDir(gConfig.io.videoOutputPath); });

    ui->line_edit_label_decode_hints_decoder->setText(gConfig.video.decoder);
    ui->line_edit_label_decode_hints_format->setText(gConfig.video.format);
    QObject::connect(ui->line_edit_label_decode_hints_decoder, &QLineEdit::textChanged, this,
        [](const QString& value) { gConfig.video.decoder = value; });
    QObject::connect(ui->line_edit_label_decode_hints_format, &QLineEdit::textChanged, this,
        [](const QString& value) { gConfig.video.format = value; });

    ui->line_edit_label_encode_hints_encoder->setText(gConfig.video.encoder);
    ui->spin_box_encode_hints_bitrate->setValue(gConfig.video.bitrate);
    QObject::connect(ui->line_edit_label_encode_hints_encoder, &QLineEdit::textChanged, this,
        [](const QString& value) { gConfig.video.encoder = value; });
    QObject::connect(ui->spin_box_encode_hints_bitrate, qOverload<int>(&QSpinBox::valueChanged), this,
        [](const int value) { gConfig.video.bitrate = value; });

    ui->combo_box_processor->addItems(gConfig.upscaler.processorList);
    ui->combo_box_processor->setCurrentText(gConfig.upscaler.processor);
    ui->spin_box_device->setValue(gConfig.upscaler.device);
    ui->double_spin_box_factor->setValue(gConfig.upscaler.factor);
    ui->combo_box_model->addItems(gConfig.upscaler.modelList);
    ui->combo_box_model->setCurrentText(gConfig.upscaler.model);
    QObject::connect(ui->combo_box_processor, &QComboBox::textActivated, this,
        [](const QString& value) { gConfig.upscaler.processor = value; });
    QObject::connect(ui->spin_box_device, qOverload<int>(&QSpinBox::valueChanged), this,
        [](const int value) { gConfig.upscaler.device = value; });
    QObject::connect(ui->double_spin_box_factor, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
        [](const double value) { gConfig.upscaler.factor = value; });
    QObject::connect(ui->combo_box_model, &QComboBox::textActivated, this,
        [](const QString& value) { gConfig.upscaler.model = value; });

    taskListModel.setHorizontalHeaderLabels({ tr("type"), tr("name"), tr("path"), tr("status") });
    ui->table_view_task_list->setModel(&taskListModel);
    ui->table_view_task_list->setEditTriggers(QTableView::NoEditTriggers);
    ui->push_button_task_list_stop->setEnabled(false);
    QObject::connect(ui->push_button_task_list_add, &QPushButton::clicked, this, &MainWindow::on_action_add_triggered);
    QObject::connect(ui->push_button_task_list_clear, &QPushButton::clicked, this, [=]() {
        taskListModel.removeRows(0, taskListModel.rowCount());
    });
    QObject::connect(ui->push_button_task_list_start, &QPushButton::clicked, this, &MainWindow::startTasks);
    QObject::connect(ui->push_button_task_list_stop, &QPushButton::clicked, &gUpscaler, &Upscaler::stop);
    QObject::connect(&gUpscaler, &Upscaler::started, ui->push_button_task_list_start,
        [=]() { ui->push_button_task_list_start->setEnabled(false); });
    QObject::connect(&gUpscaler, &Upscaler::started, ui->push_button_task_list_stop,
        [=]() { ui->push_button_task_list_stop->setEnabled(true); });
    QObject::connect(&gUpscaler, &Upscaler::stopped, ui->push_button_task_list_start,
        [=]() { ui->push_button_task_list_start->setEnabled(true); });
    QObject::connect(&gUpscaler, &Upscaler::stopped, ui->push_button_task_list_stop,
        [=]() { ui->push_button_task_list_stop->setEnabled(false); });

    ui->progress_bar_task_list->reset();
    ui->progress_bar_video_task->reset();
    QObject::connect(&gUpscaler, &Upscaler::started, ui->progress_bar_task_list, [=]() {
        taskListModel.setHeaderData(0, Qt::Horizontal, 0, Qt::UserRole);
        ui->progress_bar_task_list->reset();
    });
    QObject::connect(ui->push_button_task_list_add, &QPushButton::clicked, ui->progress_bar_task_list, &QProgressBar::reset);
    QObject::connect(ui->push_button_task_list_clear, &QPushButton::clicked, ui->progress_bar_task_list, &QProgressBar::reset);
    QObject::connect(ui->push_button_task_list_clear, &QPushButton::clicked, ui->progress_bar_video_task, &QProgressBar::reset);
    QObject::connect(&gUpscaler, &Upscaler::progress, ui->progress_bar_video_task, &QProgressBar::setValue);

    ui->text_browser_log->setSource(QUrl::fromLocalFile(gLogger.logFilePath()));
    QObject::connect(&gLogger, &Logger::logged, ui->text_browser_log, &QTextBrowser::reload);

    gLogger.info() << "started";
    gLogger.info() << gUpscaler.info();
}
void MainWindow::addTask(const QFileInfo& fileInfo)
{
    auto mimeType = QMimeDatabase{}.mimeTypeForFile(fileInfo);
    auto taskData = QSharedPointer<TaskData>::create();
    taskData->type = mimeType.inherits("image/jpeg") || mimeType.inherits("image/png") || mimeType.inherits("image/bmp")? TaskData::TYPE_IMAGE : TaskData::TYPE_VIDEO;
    taskData->path.input = fileInfo.filePath();
    auto prefix = taskData->type == TaskData::TYPE_IMAGE ? gConfig.io.imagePrefix : gConfig.io.videoPrefix;
    auto suffix = taskData->type == TaskData::TYPE_IMAGE ? gConfig.io.imageSuffix : gConfig.io.videoSuffix;
    taskData->path.output = QDir{ taskData->type == TaskData::TYPE_IMAGE ? gConfig.io.imageOutputPath : gConfig.io.videoOutputPath }.filePath(prefix + fileInfo.completeBaseName() + suffix);

    taskListModel.appendRow({ new QStandardItem{taskData->type == TaskData::TYPE_IMAGE ? "image" : "video"}, new QStandardItem{fileInfo.fileName()}, new QStandardItem{fileInfo.absoluteFilePath()}, new QStandardItem{"ready"} });
    auto rowIdx = taskListModel.rowCount() - 1;
    taskListModel.setData(taskListModel.index(rowIdx, 0), QVariant::fromValue(taskData), Qt::UserRole);
    QObject::connect(taskData.data(), &TaskData::finished, &taskListModel, [=](const bool success) {
        auto status = taskListModel.item(rowIdx, 3);
        status->setText(success ? "successed" : "failed");
        status->setForeground(success ? QColorConstants::Green : QColorConstants::Red);
        auto count = taskListModel.headerData(0, Qt::Horizontal, Qt::UserRole).toInt() + 1;
        ui->progress_bar_task_list->setValue(100 * count / taskListModel.rowCount());
        taskListModel.setHeaderData(0, Qt::Horizontal, count, Qt::UserRole);
    });
}
void MainWindow::startTasks()
{
    QList<QSharedPointer<TaskData>> taskList{};
    for (int i = 0; i < taskListModel.rowCount(); i++) taskList << (taskListModel.data(taskListModel.index(i, 0), Qt::UserRole)).value<QSharedPointer<TaskData>>();
    gUpscaler.start(taskList);
}

void MainWindow::closeEvent(QCloseEvent* const event)
{
    if (!gConfig.gui.exitConfirmation || QMessageBox::Yes == QMessageBox::question(this, "Close Confirmation", "Exit?", QMessageBox::Yes | QMessageBox::No))
        event->accept();
    else
        event->ignore();
}
void MainWindow::dragEnterEvent(QDragEnterEvent* const event)
{
    if (event->mimeData()->hasUrls()) event->acceptProposedAction();
}
void MainWindow::dropEvent(QDropEvent* const event)
{
    QFileInfo fileInfo{};
    auto urls = event->mimeData()->urls();
    for (auto&& url : urls)
    {
        fileInfo.setFile(url.toLocalFile());
        if (fileInfo.isDir())
        {
            QDir dir {fileInfo.absoluteFilePath()};
            if (!dir.exists()) return;
            auto files = dir.entryInfoList(QDir::Files);
            for (auto&& file : files) addTask(file);
        }
        else addTask(fileInfo);
    }
    event->acceptProposedAction();
}

void MainWindow::on_action_add_triggered()
{
    QFileDialog fileDialog{};
    fileDialog.setFileMode(QFileDialog::ExistingFiles);
    fileDialog.setNameFilters({ "Image (*.jpg *.jpeg *.png *.bmp)", "Video (*.mp4 *.m4v *.mkv)", "Any files (*)" });
    if (fileDialog.exec())
    {
        auto urls = fileDialog.selectedUrls();
        for (auto&& url : urls) addTask(QFileInfo{ url.toLocalFile() });
    }
}
void MainWindow::on_action_list_devices_triggered()
{
    QMessageBox::information(this, "Devices", Upscaler::info());
}
void MainWindow::on_action_about_triggered()
{
    QMessageBox::about(this, "About",
        "Anime4KCPP: A high performance anime upscaler.\n\n"
        "Anime4KCPP GUI:\n"
        "  core version: " AC_CORE_VERSION_STR " (" AC_CORE_FEATURES ")\n"
#       ifdef AC_CLI_ENABLE_VIDEO
        "  video module version: " AC_VIDEO_VERSION_STR "\n"
#       endif
        "  build date: " AC_BUILD_DATE "\n"
        "  built by: " AC_COMPILER_ID " (v" AC_COMPILER_VERSION ")\n\n"
        "Copyright (c) by TianZerL the Anime4KCPP project 2020-" AC_BUILD_YEAR "\n\n"
        "https://github.com/TianZerL/Anime4KCPP\n"
    );
}

QString MainWindow::selectDir()
{
    QString path{};
    QFileDialog fileDialog{};
    fileDialog.setFileMode(QFileDialog::Directory);
    if (fileDialog.exec()) path = fileDialog.selectedFiles().first();
    return path;
}
void MainWindow::openDir(const QString& path)
{
    QDesktopServices::openUrl(QUrl::fromLocalFile(path));
}
