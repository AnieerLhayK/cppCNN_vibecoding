#pragma once

#include <QString>
#include <QStringList>
#include <QVariantList>

#include <cstddef>

class ResourceLocator {
public:
    [[nodiscard]] static QString findDefaultModel();
    [[nodiscard]] static QString findLabels();
    [[nodiscard]] static QString findLabelsForModel(const QString& modelPath);
    [[nodiscard]] static QStringList readLabels(
        const QString& path,
        std::size_t classCount);
    [[nodiscard]] static QVariantList findDemoImages();

private:
    [[nodiscard]] static QString firstExistingFile(const QStringList& candidates);
};
