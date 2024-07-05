#include <QApplication>
#include <QLocale>
#include <QTranslator>

#include "MainWindow.hpp"

int main(int argc, char* argv[])
{
    QApplication app{ argc, argv };
    QApplication::setWindowIcon(QIcon{ ":/icon/logo" });

    QTranslator translator{};
    if (translator.load(QLocale{}, "ac_gui", "_", ":/i18n")) QApplication::installTranslator(&translator);

    MainWindow mainWindow{};
    mainWindow.show();
    return app.exec();
}
