#include "gui/AppController.h"

AppController::AppController(QObject* parent)
    : QObject(parent) {
}

QString AppController::statusText() const {
    return QStringLiteral("Qt interface initialized");
}
