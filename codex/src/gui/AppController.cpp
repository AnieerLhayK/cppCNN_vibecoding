#include "gui/AppController.h"

#include "gui/ImageBridge.h"
#include "gui/InferenceEngine.h"
#include "gui/ResourceLocator.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QImageReader>
#include <QtConcurrent>

#include <exception>
#include <filesystem>

AppController::AppController(QObject* parent)
    : QObject(parent) {
    connect(&watcher_, &QFutureWatcher<PredictionPayload>::finished, this, &AppController::finishPrediction);
    demoImages_ = ResourceLocator::findDemoImages();
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

QString AppController::applicationVersion() const {
    return QStringLiteral(CPPCNN_VERSION);
}

QUrl AppController::imageUrl() const {
    return imageUrl_;
}

QString AppController::imagePath() const {
    return imagePath_;
}

QString AppController::imageDetails() const {
    return imageDetails_;
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
    imageDetails_ = QStringLiteral("%1 x %2 | %3")
        .arg(image.width())
        .arg(image.height())
        .arg(QString::fromLatin1(reader.format()).toUpper());
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
    imageDetails_.clear();
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
    const QStringList labels = labels_;
    watcher_.setFuture(QtConcurrent::run(
        [network, input, labels]() {
            return InferenceEngine::predict(network, input, labels);
        }));
}

void AppController::discoverResources() {
    const QString model = ResourceLocator::findDefaultModel();

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
        const QString labelsPath = ResourceLocator::findLabels();
        const QStringList labels =
            ResourceLocator::readLabels(labelsPath, info.classCount);

        network_ = std::move(network);
        modelInfo_ = info;
        modelPath_ = QDir::toNativeSeparators(QFileInfo(path).absoluteFilePath());
        labelsPath_ = labelsPath;
        labels_ = labels;
        modelStatus_ = QStringLiteral("Model ready");
        modelDetails_ = QStringLiteral(
            "Format v%1 | LeNet | %2 trainable layers | %3 parameters")
            .arg(info.version)
            .arg(info.trainableLayerCount)
            .arg(info.parameterCount);
        clearPrediction();
        emit modelChanged();
        setStatus(QStringLiteral("Model loaded: %1 classes").arg(info.classCount));
    } catch (const std::exception& error) {
        if (!network_) {
            modelInfo_ = {};
            modelPath_.clear();
            labelsPath_.clear();
            labels_.clear();
            modelStatus_ = QStringLiteral("Invalid model");
            modelDetails_ = QStringLiteral("The selected model could not be validated.");
        }
        clearPrediction();
        emit modelChanged();
        setStatus(QStringLiteral("Model load failed"), QString::fromUtf8(error.what()));
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
