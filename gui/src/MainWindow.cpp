#include <cstddef>
#include <iterator>

#include <QColor>
#include <QCursor>
#include <QDesktopServices>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QGridLayout>
#include <QIODevice>
#include <QList>
#include <QMessageBox>
#include <QMimeData>
#include <QMimeDatabase>
#include <QSharedPointer>
#include <QSpacerItem>
#include <QStyleFactory>
#include <QTextBrowser>
#include <QVariant>
#include <QVBoxLayout>
#include <QWeakPointer>

#include "AC/Specs.hpp"
#include "AC/Util/Stopwatch.hpp"

#include "Config.hpp"
#include "Logger.hpp"
#include "MainWindow.hpp"
#include "ui_MainWindow.h"
#include "Upscaler.hpp"

MainWindow::MainWindow() : QMainWindow(nullptr), ui(std::make_unique<Ui::MainWindow>())
{
    ui->setupUi(this);
    init();
}

MainWindow::~MainWindow() noexcept = default;

void MainWindow::init()
{
    qApp->setStyle(QStyleFactory::create(gConfig.gui.styleName));

    QObject::connect(ui->action_exit, &QAction::triggered, this, &MainWindow::close);

    for (auto&& style: QStyleFactory::keys())
    {
        auto action = new QAction{ style, ui->menu_settings_style };
        QObject::connect(action, &QAction::triggered, this, [=]() {
            gConfig.gui.styleName = action->text();
            qApp->setStyle(QStyleFactory::create(gConfig.gui.styleName));
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

    ui->line_edit_codec_hints_decoder->setText(gConfig.video.decoder);
    ui->line_edit_codec_hints_format->setText(gConfig.video.format);
    ui->line_edit_codec_hints_encoder->setText(gConfig.video.encoder);
    ui->spin_box_codec_hints_bitrate->setValue(gConfig.video.bitrate);
    QObject::connect(ui->line_edit_codec_hints_decoder, &QLineEdit::textChanged, this,
        [](const QString& value) { gConfig.video.decoder = value; });
    QObject::connect(ui->line_edit_codec_hints_format, &QLineEdit::textChanged, this,
        [](const QString& value) { gConfig.video.format = value; });
    QObject::connect(ui->line_edit_codec_hints_encoder, &QLineEdit::textChanged, this,
        [](const QString& value) { gConfig.video.encoder = value; });
    QObject::connect(ui->spin_box_codec_hints_bitrate, qOverload<int>(&QSpinBox::valueChanged), this,
        [](const int value) { gConfig.video.bitrate = value; });

    ui->spin_box_device->setValue(gConfig.upscaler.device);
    ui->double_spin_box_factor->setMinimum(1.0);
    ui->double_spin_box_factor->setValue(gConfig.upscaler.factor);
    ui->combo_box_processor->addItems({ std::begin(ac::specs::ProcessorList), std::end(ac::specs::ProcessorList) });
    for (std::size_t i = 0; i < std::size(ac::specs::ProcessorDescriptionList); i++) ui->combo_box_processor->setItemData(i, QCoreApplication::translate("ExternI18N", ac::specs::ProcessorDescriptionList[i]), Qt::ToolTipRole);
    ui->combo_box_processor->setCurrentText(gConfig.upscaler.processor);
    ui->combo_box_model->setStyleSheet("combobox-popup: 0;");
    ui->combo_box_model->view()->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    ui->combo_box_model->addItems({ std::begin(ac::specs::ModelList), std::end(ac::specs::ModelList) });
    for (std::size_t i = 0; i < std::size(ac::specs::ModelDescriptionList); i++) ui->combo_box_model->setItemData(i, QCoreApplication::translate("ExternI18N", ac::specs::ModelDescriptionList[i]), Qt::ToolTipRole);
    ui->combo_box_model->setCurrentText(gConfig.upscaler.model);
    QObject::connect(ui->spin_box_device, qOverload<int>(&QSpinBox::valueChanged), this,
        [](const int value) { gConfig.upscaler.device = value; });
    QObject::connect(ui->double_spin_box_factor, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
        [](const double value) { gConfig.upscaler.factor = value; });
    QObject::connect(ui->combo_box_processor, &QComboBox::textActivated, this,
        [](const QString& value) { gConfig.upscaler.processor = value; });
    QObject::connect(ui->combo_box_model, &QComboBox::textActivated, this,
        [](const QString& value) { gConfig.upscaler.model = value; });

    taskListModel.setHorizontalHeaderLabels({ tr("type"), tr("status"), tr("name"), tr("output name"), tr("path") });
    ui->table_view_task_list->setModel(&taskListModel);
    ui->table_view_task_list->setEditTriggers(QTableView::NoEditTriggers);
    ui->table_view_task_list->setContextMenuPolicy(Qt::CustomContextMenu);
    QObject::connect(ui->table_view_task_list, &QTableView::doubleClicked, this, &MainWindow::openTaskFile);
    // right click menu
    auto taskListContextMenu = new QMenu{ui->table_view_task_list};
    auto taskListActionOpen = new QAction{tr("open"), taskListContextMenu};
    auto taskListActionRemove = new QAction{tr("remove"), taskListContextMenu};
    taskListContextMenu->addActions({ taskListActionOpen, taskListActionRemove });
    QObject::connect(taskListActionOpen, &QAction::triggered, taskListContextMenu, [=]() {
        auto index = ui->table_view_task_list->currentIndex();
        if (index.isValid()) openTaskFile(index);
    });
    QObject::connect(taskListActionRemove, &QAction::triggered, taskListContextMenu, [=]() {
        auto index = ui->table_view_task_list->currentIndex();
        if (index.isValid()) taskListModel.removeRow(index.row());
    });
    QObject::connect(ui->table_view_task_list, &QTableView::customContextMenuRequested, taskListContextMenu, [=](const QPoint &pos) {
        auto index = ui->table_view_task_list->indexAt(pos);
        if (index.isValid()) taskListContextMenu->popup(QCursor::pos());
    });

    ui->push_button_task_list_stop->setEnabled(false);
    QObject::connect(ui->push_button_task_list_add, &QPushButton::clicked, this, &MainWindow::on_action_add_triggered);
    QObject::connect(ui->push_button_task_list_clear, &QPushButton::clicked, this,
        [=]() { taskListModel.removeRows(0, taskListModel.rowCount()); });
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
    QObject::connect(ui->push_button_task_list_add, &QPushButton::clicked, ui->progress_bar_video_task, &QProgressBar::reset);
    QObject::connect(ui->push_button_task_list_clear, &QPushButton::clicked, ui->progress_bar_task_list, &QProgressBar::reset);
    QObject::connect(ui->push_button_task_list_clear, &QPushButton::clicked, ui->progress_bar_video_task, &QProgressBar::reset);
    QObject::connect(&gUpscaler, &Upscaler::progress, ui->progress_bar_video_task, [this, stopwatchVideoTask = QSharedPointer<ac::util::Stopwatch>::create()](const int value) {
        ui->progress_bar_video_task->setValue(value);

        if (value == 0) stopwatchVideoTask->reset();
        else
        {
            double elapsed = stopwatchVideoTask->elapsed();
            double remaining = elapsed / (value / 100.0) - elapsed;
            ac::util::Stopwatch::FormatBuffer elapsedBuffer{}, remainingBuffer{};
            ui->progress_bar_video_task->setFormat(QString{ "%p% (%1 < %2)" }.arg(ac::util::Stopwatch::formatDuration(elapsedBuffer, elapsed), ac::util::Stopwatch::formatDuration(remainingBuffer, remaining)));
        }
    });

    QObject::connect(&gLogger, &Logger::logged, ui->text_browser_log, &QTextBrowser::append);

    setAcceptDrops(true);

    gLogger.info() << "Anime4KCPP GUI v" AC_CORE_VERSION_STR " started";
    gLogger.info() << '\n' << gUpscaler.listProcessorInfo();
}

void MainWindow::addTask(const QFileInfo& fileInfo)
{
    auto mimeType = QMimeDatabase{}.mimeTypeForFile(fileInfo);
    auto taskData = QSharedPointer<TaskData>::create();
    taskData->type = mimeType.inherits("image/jpeg") || mimeType.inherits("image/png") || mimeType.inherits("image/bmp")? TaskData::TYPE_IMAGE : TaskData::TYPE_VIDEO;
    taskData->path.input = fileInfo.filePath();

    auto itemType       = new QStandardItem{ taskData->type == TaskData::TYPE_IMAGE ? tr("image") : tr("video") };
    auto itemStatus     = new QStandardItem{ tr("ready") };
    auto itemName       = new QStandardItem{ fileInfo.fileName() };
    auto itemOutputName = new QStandardItem{};
    auto itemPath       = new QStandardItem{ fileInfo.absoluteFilePath() };
    itemType->setData(QVariant::fromValue(taskData), Qt::UserRole);

    taskListModel.appendRow({ itemType, itemStatus, itemName, itemOutputName, itemPath, });
    QObject::connect(taskData.data(), &TaskData::finished, &taskListModel, [=](const bool success) {
        itemStatus->setText(success ? tr("succeeded") : tr("failed"));
        itemStatus->setForeground(success ? QColorConstants::Green : QColorConstants::Red);
        auto count = taskListModel.headerData(0, Qt::Horizontal, Qt::UserRole).toInt() + 1;
        ui->progress_bar_task_list->setValue(100 * count / taskListModel.rowCount());
        taskListModel.setHeaderData(0, Qt::Horizontal, count, Qt::UserRole);
    });

    auto updateOutputInfo = [taskDataWeakPointer = QWeakPointer<TaskData>(taskData), baseName = fileInfo.completeBaseName(), itemOutputName, itemPath]() {
        if (auto taskData = taskDataWeakPointer.toStrongRef())
        {
            auto& prefix = taskData->type == TaskData::TYPE_IMAGE ? gConfig.io.imagePrefix : gConfig.io.videoPrefix;
            auto& suffix = taskData->type == TaskData::TYPE_IMAGE ? gConfig.io.imageSuffix : gConfig.io.videoSuffix;
            auto outputName = prefix + baseName + suffix;
            taskData->path.output = QDir{ taskData->type == TaskData::TYPE_IMAGE ? gConfig.io.imageOutputPath : gConfig.io.videoOutputPath }.filePath(outputName);

            itemOutputName->setText(outputName);
            itemPath->setData(QString{ "%1: %3\n%2: %4" }.arg(tr("input"), tr("output"), taskData->path.input, taskData->path.output), Qt::ToolTipRole);
        }
    };

    updateOutputInfo();

    if (taskData->type == TaskData::TYPE_IMAGE)
    {
        QObject::connect(ui->line_edit_prefix_image, &QLineEdit::textChanged, taskData.data(), updateOutputInfo);
        QObject::connect(ui->line_edit_suffix_image, &QLineEdit::textChanged, taskData.data(), updateOutputInfo);
        QObject::connect(ui->line_edit_output_path_image, &QLineEdit::textChanged, taskData.data(), updateOutputInfo);
    }
    else
    {
        QObject::connect(ui->line_edit_prefix_video, &QLineEdit::textChanged, taskData.data(), updateOutputInfo);
        QObject::connect(ui->line_edit_suffix_video, &QLineEdit::textChanged, taskData.data(), updateOutputInfo);
        QObject::connect(ui->line_edit_output_path_video, &QLineEdit::textChanged, taskData.data(), updateOutputInfo);
    }
}
void MainWindow::startTasks()
{
    QList<QSharedPointer<TaskData>> taskList{};
    for (int i = 0; i < taskListModel.rowCount(); i++) taskList << (taskListModel.data(taskListModel.index(i, 0), Qt::UserRole)).value<QSharedPointer<TaskData>>();
    gUpscaler.start(taskList);
}
void MainWindow::openTaskFile(const QModelIndex& index)
{
    auto variant = taskListModel.data(taskListModel.index(index.row(), 0), Qt::UserRole);
    if (variant.canConvert<QSharedPointer<TaskData>>())
    {
        auto data = variant.value<QSharedPointer<TaskData>>();
        if (index.column() == 3 && QFileInfo::exists(data->path.output))
            QDesktopServices::openUrl(QUrl::fromLocalFile(variant.value<QSharedPointer<TaskData>>()->path.output));
        else
            QDesktopServices::openUrl(QUrl::fromLocalFile(variant.value<QSharedPointer<TaskData>>()->path.input));
    }
}

void MainWindow::closeEvent(QCloseEvent* const event)
{
    if (!gConfig.gui.exitConfirmation || QMessageBox::Yes == QMessageBox::question(this, tr("Exit confirmation"), tr("Exit") + '?', QMessageBox::Yes | QMessageBox::No))
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
    fileDialog.setNameFilters({ "Media (*.jpg *.jpeg *.png *.bmp *.mp4 *.m4v *.mkv *.webm)", "Image (*.jpg *.jpeg *.png *.bmp)", "Video (*.mp4 *.m4v *.mkv *.webm)", "Any files (*)" });
    if (fileDialog.exec())
    {
        auto urls = fileDialog.selectedUrls();
        for (auto&& url : urls) addTask(QFileInfo{ url.toLocalFile() });
    }
}
void MainWindow::on_action_list_devices_triggered()
{
    auto devicesMessageBox = new QMessageBox{ this };
    devicesMessageBox->setAttribute(Qt::WA_DeleteOnClose);
    devicesMessageBox->setWindowTitle(tr("Devices"));
    devicesMessageBox->setWindowModality(Qt::NonModal);
    devicesMessageBox->setText(gUpscaler.listProcessorInfo());
    if (auto layout = qobject_cast<QGridLayout*>(devicesMessageBox->layout()))
        layout->addItem(new QSpacerItem{ 250, 0, QSizePolicy::Minimum, QSizePolicy::Expanding }, layout->rowCount(), 0, 1, layout->columnCount());
    devicesMessageBox->show();
}
void MainWindow::on_action_license_triggered()
{
    QFile licenseAC{ ":/license/ac" };
    if (licenseAC.open(QIODevice::Text | QIODevice::ReadOnly))
    {
        auto licenseDialog = new QDialog{ this };
        licenseDialog->setAttribute(Qt::WA_DeleteOnClose);
        licenseDialog->setWindowTitle(tr("License"));
        licenseDialog->resize(600, 300);

        auto verticalLayout = new QVBoxLayout{ licenseDialog };
        auto textBrowser = new QTextBrowser{ licenseDialog };
        auto buttonBox = new QDialogButtonBox{ licenseDialog };
        buttonBox->setOrientation(Qt::Orientation::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::StandardButton::Ok);
        QObject::connect(buttonBox, &QDialogButtonBox::accepted, licenseDialog, &QDialog::accept);

        verticalLayout->addWidget(textBrowser);
        verticalLayout->addWidget(buttonBox);
        textBrowser->setText(
            QString{
                "<p style='white-space: pre-wrap;'>"
                "<h3>Anime4KCPP</h3>"
                "%1"
                "</p>"
            }.arg(QString{ licenseAC.readAll() })
        );
        licenseDialog->setLayout(verticalLayout);
        licenseDialog->show();
    }
}
void MainWindow::on_action_about_triggered()
{
    QMessageBox::about(this, tr("About"),
        QString{
            "<p style='white-space: pre-wrap;'>"
            "%1\n\n"
            "Anime4KCPP GUI:\n"
            "  %2: " AC_CORE_VERSION_STR " (" AC_CORE_FEATURES ")\n"
            "  %3: "
#           ifdef AC_CLI_ENABLE_VIDEO
                AC_VIDEO_VERSION_STR "\n"
#           else
                "${DISABLED}\n"
#           endif
            "  %4: " AC_BUILD_DATE "\n"
            "  %5: " AC_COMPILER_ID " (v" AC_COMPILER_VERSION ")\n\n"
            "%6 (c) 2020-" AC_BUILD_YEAR " the Anime4KCPP project\n\n"
            "<a href='https://github.com/TianZerL/Anime4KCPP'>https://github.com/TianZerL/Anime4KCPP</a>\n"
            "</p>"
        }.arg(
            tr("Anime4KCPP: A high performance anime upscaler"),
            tr("core version"),
            tr("video module"),
            tr("build date"),
            tr("toolchain"),
            tr("Copyright")
        ).replace("${DISABLED}", tr("disabled"))
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
