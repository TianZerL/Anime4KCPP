#ifndef AC_GUI_MAIN_WINDOW_HPP
#define AC_GUI_MAIN_WINDOW_HPP

#include <memory>

#include <QCloseEvent>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileInfo>
#include <QMainWindow>
#include <QModelIndex>
#include <QStandardItemModel>
#include <QString>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow();
    ~MainWindow() noexcept override;

private:
    void init();
    void addTask(const QFileInfo& fileInfo);
    void startTasks();
    void openTaskFile(const QModelIndex& index);

private:
    void closeEvent(QCloseEvent* event) override;
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dropEvent(QDropEvent* event) override;

private slots:
    void on_action_add_triggered();
    void on_action_list_devices_triggered();
    void on_action_license_triggered();
    void on_action_about_triggered();

private:
    static QString selectDir();
    static void openDir(const QString& path);

private:
    QStandardItemModel taskListModel;

private:
    const std::unique_ptr<Ui::MainWindow> ui;
};

#endif
