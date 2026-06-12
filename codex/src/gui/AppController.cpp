#include "gui/AppController.h"

#include "gui/ImageBridge.h"

#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <QTextStream>
#include <QtConcurrent>

#include <algorithm>
#include <exception>
#include <filesystem>
#include <numeric>
#include <utility>

namespace {

QString displayLabel(const QString& label, const int classId) {
    return label.isEmpty()
        ? QStringLiteral("Class %1").arg(classId)
        : label;
}

}  // namespace

AppController::AppController(QObject* parent)
    : QObject(parent) {
    connect(&watcher_, &QFutureWatcher<PredictionPayload>::finished, this, &AppController::finishPrediction);
    loadDemoImages();
    discoverResources();
}

AppController::~AppController() {
    watcher_.waitForFinished();
}

bool AppController::modelLoaded() const {
    return network_ != nullptr;
}

bool AppController::imageLoaded() const {
    return imageLoaded_;
}

bool AppController::busy() const {
    return busy_;
}

int AppController::classCount() const {
    return static_cast<int>(modelInfo_.classCount);
}

QString AppController::modelStatus() const {
    return modelStatus_;
}

QString AppController::modelPath() const {
    return modelPath_;
}

QString AppController::labelsPath() const {
    return labelsPath_;
}

QString AppController::modelDetails() const {
    return modelDetails_;
}

QUrl AppController::imageUrl() const {
    return imageUrl_;
}

QString AppController::imagePath() const {
    return imagePath_;
}

QString AppController::predictionLabel() const {
    return predictionLabel_;
}

double AppController::confidence() const {
    return confidence_;
}

qint64 AppController::inferenceMilliseconds() const {
    return inferenceMilliseconds_;
}

QVariantList AppController::topResults() const {
    return topResults_;
}

QVariantList AppController::demoImages() const {
    return demoImages_;
}

QString AppController::statusText() const {
    return statusText_;
}

QString AppController::errorText() const {
    return errorText_;
}

void AppController::loadImage(const QUrl& url) {
    if (busy_) {
        return;
    }

    const QString path = localPath(url);
    QImageReader reader(path);
    reader.setAutoTransform(true);
    const QImage image = reader.read();
    if (image.isNull()) {
        setStatus(
            QStringLiteral("Image load failed"),
            QStringLiteral("Could not read this image: %1").arg(reader.errorString()));
        return;
    }

    try {
        inputTensor_ = ImageBridge::toTensor(image);
    } catch (const std::exception& error) {
        setStatus(QStringLiteral("Image preprocessing failed"), QString::fromUtf8(error.what()));
        return;
    }

    imagePath_ = QDir::toNativeSeparators(QFileInfo(path).absoluteFilePath());
    imageUrl_ = QUrl::fromLocalFile(QFileInfo(path).absoluteFilePath());
    imageLoaded_ = true;
    clearPrediction();
    emit imageChanged();
    setStatus(QStringLiteral("Image ready: %1").arg(QFileInfo(path).fileName()));
}

void AppController::clearImage() {
    if (busy_) {
        return;
    }
    inputTensor_ = {};
    imageUrl_ = {};
    imagePath_.clear();
    imageLoaded_ = false;
    clearPrediction();
    emit imageChanged();
    setStatus(QStringLiteral("Choose or drop an image to begin."));
}

void AppController::loadModel(const QUrl& url) {
    if (busy_) {
        return;
    }
    const QString path = localPath(url);
    if (path.isEmpty()) {
        return;
    }
    applyModel(path);
}

void AppController::predict() {
    if (busy_) {
        return;
    }
    if (!network_) {
        setStatus(
            QStringLiteral("Model required"),
            QStringLiteral("Load gtsrb_subset10.bin before running recognition."));
        return;
    }
    if (!imageLoaded_ || inputTensor_.empty()) {
        setStatus(
            QStringLiteral("Image required"),
            QStringLiteral("Choose a supported image before running recognition."));
        return;
    }

    busy_ = true;
    clearPrediction();
    predictionLabel_ = QStringLiteral("Analyzing...");
    emit busyChanged();
    emit predictionChanged();
    setStatus(QStringLiteral("CNN inference is running in the background..."));

    const auto network = network_;
    const cppcnn::Tensor input = inputTensor_;
    const std::vector<QString> labels = labels_;
    watcher_.setFuture(QtConcurrent::run([network, input, labels]() {
        PredictionPayload payload;
        try {
            QElapsedTimer timer;
            timer.start();
            const cppcnn::Tensor probabilities = network->forward(input);
            payload.elapsedMilliseconds = timer.elapsed();

            std::vector<std::size_t> indices(probabilities.size());
            std::iota(indices.begin(), indices.end(), 0U);
            const std::size_t resultCount = std::min<std::size_t>(3, indices.size());
            std::partial_sort(
                indices.begin(),
                indices.begin() + static_cast<std::ptrdiff_t>(resultCount),
                indices.end(),
                [&probabilities](const std::size_t left, const std::size_t right) {
                    return probabilities[left] > probabilities[right];
                });

            for (std::size_t rank = 0; rank < resultCount; ++rank) {
                const auto classId = indices[rank];
                PredictionResult result;
                result.classId = static_cast<int>(classId);
                result.label = displayLabel(
                    classId < labels.size() ? labels[classId] : QString(),
                    result.classId);
                result.confidence = probabilities[classId];
                payload.results.push_back(std::move(result));
            }
        } catch (const std::exception& error) {
            payload.error = QString::fromUtf8(error.what());
        }
        return payload;
    }));
}

void AppController::discoverResources() {
    const QString applicationDir = QCoreApplication::applicationDirPath();
    const QString currentDir = QDir::currentPath();
    const QString model = firstExistingFile({
        QDir(applicationDir).filePath(QStringLiteral("models/gtsrb_subset10.bin")),
        QDir(currentDir).filePath(QStringLiteral("codex/models/gtsrb_subset10.bin")),
        QDir(currentDir).filePath(QStringLiteral("models/gtsrb_subset10.bin")),
    });

    if (model.isEmpty()) {
        modelStatus_ = QStringLiteral("Model missing");
        modelDetails_ = QStringLiteral("Select a compatible .bin model to enable recognition.");
        setStatus(
            QStringLiteral("Interface ready; model not found."),
            QStringLiteral("Expected models/gtsrb_subset10.bin beside the application or under codex/models."));
        emit modelChanged();
        return;
    }
    applyModel(model);
}

void AppController::applyModel(const QString& path) {
    try {
        const auto nativePath = std::filesystem::path(path.toStdWString());
        const cppcnn::ModelInfo info = cppcnn::CNN::inspectModel(nativePath);
        auto network = std::make_shared<cppcnn::CNN>(info.classCount);
        network->loadModel(nativePath);

        network_ = std::move(network);
        modelInfo_ = info;
        modelPath_ = QDir::toNativeSeparators(QFileInfo(path).absoluteFilePath());
        modelStatus_ = QStringLiteral("Model ready");
        modelDetails_ = QStringLiteral(
            "Format v%1 · LeNet · %2 trainable layers · %3 parameters")
            .arg(info.version)
            .arg(info.trainableLayerCount)
            .arg(info.parameterCount);
        loadLabels();
        clearPrediction();
        emit modelChanged();
        setStatus(QStringLiteral("Model loaded: %1 classes").arg(info.classCount));
    } catch (const std::exception& error) {
        network_.reset();
        modelInfo_ = {};
        modelPath_.clear();
        labels_.clear();
        modelStatus_ = QStringLiteral("Invalid model");
        modelDetails_ = QStringLiteral("The selected model could not be validated.");
        clearPrediction();
        emit modelChanged();
        setStatus(QStringLiteral("Model load failed"), QString::fromUtf8(error.what()));
    }
}

void AppController::loadLabels() {
    const QString applicationDir = QCoreApplication::applicationDirPath();
    const QString currentDir = QDir::currentPath();
    labelsPath_ = firstExistingFile({
        QDir(applicationDir).filePath(QStringLiteral("labels.txt")),
        QDir(currentDir).filePath(QStringLiteral("codex/datasets/GTSRB_subset/labels.txt")),
        QDir(currentDir).filePath(QStringLiteral("datasets/GTSRB_subset/labels.txt")),
        QDir(currentDir).filePath(QStringLiteral("codex/assets/labels.txt")),
        QDir(currentDir).filePath(QStringLiteral("assets/labels.txt")),
    });

    const QStringList loaded = readLabels(labelsPath_, modelInfo_.classCount);
    labels_.assign(loaded.begin(), loaded.end());
}

void AppController::loadDemoImages() {
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
        return;
    }

    const QDir directory(directoryPath);
    const QFileInfoList files = directory.entryInfoList(
        {QStringLiteral("*.ppm"), QStringLiteral("*.png"), QStringLiteral("*.jpg"),
         QStringLiteral("*.jpeg"), QStringLiteral("*.bmp")},
        QDir::Files,
        QDir::Name);
    for (const QFileInfo& file : files) {
        QVariantMap item;
        item.insert(QStringLiteral("name"), file.completeBaseName().replace(QLatin1Char('_'), QLatin1Char(' ')));
        item.insert(QStringLiteral("url"), QUrl::fromLocalFile(file.absoluteFilePath()));
        demoImages_.push_back(item);
    }
}

void AppController::clearPrediction() {
    predictionLabel_ = QStringLiteral("Awaiting prediction");
    confidence_ = 0.0;
    inferenceMilliseconds_ = 0;
    topResults_.clear();
    emit predictionChanged();
}

void AppController::setStatus(const QString& status, const QString& error) {
    statusText_ = status;
    errorText_ = error;
    emit statusChanged();
}

void AppController::finishPrediction() {
    const PredictionPayload payload = watcher_.result();
    busy_ = false;
    emit busyChanged();

    if (!payload.error.isEmpty() || payload.results.isEmpty()) {
        predictionLabel_ = QStringLiteral("Prediction failed");
        setStatus(
            QStringLiteral("CNN inference failed"),
            payload.error.isEmpty() ? QStringLiteral("The network returned no probabilities.") : payload.error);
        emit predictionChanged();
        return;
    }

    inferenceMilliseconds_ = payload.elapsedMilliseconds;
    predictionLabel_ = payload.results.front().label;
    confidence_ = payload.results.front().confidence;
    topResults_.clear();
    for (const PredictionResult& result : payload.results) {
        QVariantMap item;
        item.insert(QStringLiteral("classId"), result.classId);
        item.insert(QStringLiteral("label"), result.label);
        item.insert(QStringLiteral("confidence"), result.confidence);
        topResults_.push_back(item);
    }
    emit predictionChanged();
    setStatus(QStringLiteral("Recognition complete."));
}

QString AppController::localPath(const QUrl& url) {
    if (url.isLocalFile()) {
        return url.toLocalFile();
    }
    const QUrl parsed(url.toString());
    return parsed.isLocalFile() ? parsed.toLocalFile() : url.toString();
}

QString AppController::firstExistingFile(const QStringList& candidates) {
    for (const QString& candidate : candidates) {
        const QFileInfo file(candidate);
        if (file.isFile()) {
            return file.absoluteFilePath();
        }
    }
    return {};
}

QStringList AppController::readLabels(const QString& path, const std::size_t classCount) {
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
