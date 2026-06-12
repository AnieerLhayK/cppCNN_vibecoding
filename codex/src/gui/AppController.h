#pragma once

#include <QObject>

class AppController final : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString statusText READ statusText CONSTANT)

public:
    explicit AppController(QObject* parent = nullptr);

    [[nodiscard]] QString statusText() const;
};
