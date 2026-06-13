#include "gui/ResourceLocator.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QTextStream>
#include <QUrl>

QString ResourceLocator::findDefaultModel() {
    const QString applicationDir = QCoreApplication::applicationDirPath();
    const QString currentDir = QDir::currentPath();
    return firstExistingFile({
        QDir(applicationDir).filePath(QStringLiteral("models/gtsrb_subset10.bin")),
        QDir(currentDir).filePath(QStringLiteral("codex/models/gtsrb_subset10.bin")),
        QDir(currentDir).filePath(QStringLiteral("models/gtsrb_subset10.bin")),
    });
}

QString ResourceLocator::findLabels() {
    const QString applicationDir = QCoreApplication::applicationDirPath();
    const QString currentDir = QDir::currentPath();
    return firstExistingFile({
        QDir(applicationDir).filePath(QStringLiteral("labels.txt")),
        QDir(currentDir).filePath(QStringLiteral("codex/datasets/GTSRB_subset/labels.txt")),
        QDir(currentDir).filePath(QStringLiteral("datasets/GTSRB_subset/labels.txt")),
        QDir(currentDir).filePath(QStringLiteral("codex/assets/labels.txt")),
        QDir(currentDir).filePath(QStringLiteral("assets/labels.txt")),
    });
}

QStringList ResourceLocator::readLabels(
    const QString& path,
    const std::size_t classCount) {
    QStringList labels;
    QFile file(path);
    if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream input(&file);
        while (!input.atEnd() && labels.size() < static_cast<qsizetype>(classCount)) {
            const QString line = input.readLine().trimmed();
            if (!line.isEmpty()) {
                labels.push_back(line);
            }
        }
    }
    while (labels.size() < static_cast<qsizetype>(classCount)) {
        labels.push_back(QStringLiteral("Class %1").arg(labels.size()));
    }
    return labels;
}

QVariantList ResourceLocator::findDemoImages() {
    const QString applicationDir = QCoreApplication::applicationDirPath();
    const QString currentDir = QDir::currentPath();
    const QStringList directories = {
        QDir(applicationDir).filePath(QStringLiteral("demo_images")),
        QDir(currentDir).filePath(QStringLiteral("codex/Release/demo_images")),
        QDir(currentDir).filePath(QStringLiteral("Release/demo_images")),
    };

    QString directoryPath;
    for (const QString& candidate : directories) {
        if (QDir(candidate).exists()) {
            directoryPath = candidate;
            break;
        }
    }
    if (directoryPath.isEmpty()) {
        return {};
    }

    QVariantList results;
    const QFileInfoList files = QDir(directoryPath).entryInfoList(
        {QStringLiteral("*.ppm"), QStringLiteral("*.png"), QStringLiteral("*.jpg"),
         QStringLiteral("*.jpeg"), QStringLiteral("*.bmp")},
        QDir::Files,
        QDir::Name);
    for (const QFileInfo& file : files) {
        QVariantMap item;
        QString displayName =
            file.completeBaseName().replace(QLatin1Char('_'), QLatin1Char(' '));
        displayName.remove(QRegularExpression(QStringLiteral("^\\d+\\s+")));
        item.insert(
            QStringLiteral("name"),
            displayName);
        item.insert(QStringLiteral("url"), QUrl::fromLocalFile(file.absoluteFilePath()));
        results.push_back(item);
    }
    return results;
}

QString ResourceLocator::firstExistingFile(const QStringList& candidates) {
    for (const QString& candidate : candidates) {
        const QFileInfo file(candidate);
        if (file.isFile()) {
            return file.absoluteFilePath();
        }
    }
    return {};
}
