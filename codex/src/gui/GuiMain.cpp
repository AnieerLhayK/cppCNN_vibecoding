#include "gui/AppController.h"

#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickStyle>

int main(int argc, char* argv[]) {
    QGuiApplication application(argc, argv);
    application.setApplicationName(QStringLiteral("cppCNN Traffic Sign Studio"));
    application.setOrganizationName(QStringLiteral("cppCNN Vibecoding"));
    QQuickStyle::setStyle(QStringLiteral("Basic"));

    AppController controller;
    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty(QStringLiteral("appController"), &controller);
    engine.loadFromModule(QStringLiteral("CppCNNGui"), QStringLiteral("Main"));
    if (engine.rootObjects().isEmpty()) {
        return EXIT_FAILURE;
    }
    return application.exec();
}
