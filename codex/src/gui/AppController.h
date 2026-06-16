#pragma once

#include "cnn/CNN.h"
#include "cnn/Tensor.h"
#include "gui/PredictionResult.h"

#include <QFutureWatcher>
#include <QImage>
#include <QObject>
#include <QStringList>
#include <QUrl>
#include <QVariantList>

#include <memory>

class AppController final : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool modelLoaded READ modelLoaded NOTIFY modelChanged)
    Q_PROPERTY(bool imageLoaded READ imageLoaded NOTIFY imageChanged)
    Q_PROPERTY(bool busy READ busy NOTIFY busyChanged)
    Q_PROPERTY(int classCount READ classCount NOTIFY modelChanged)
    Q_PROPERTY(QString modelStatus READ modelStatus NOTIFY modelChanged)
    Q_PROPERTY(QString modelPath READ modelPath NOTIFY modelChanged)
    Q_PROPERTY(QString labelsPath READ labelsPath NOTIFY modelChanged)
    Q_PROPERTY(QString modelDetails READ modelDetails NOTIFY modelChanged)
    Q_PROPERTY(QString applicationVersion READ applicationVersion CONSTANT)
    Q_PROPERTY(QUrl imageUrl READ imageUrl NOTIFY imageChanged)
    Q_PROPERTY(QString imagePath READ imagePath NOTIFY imageChanged)
    Q_PROPERTY(QString imageDetails READ imageDetails NOTIFY imageChanged)
    Q_PROPERTY(QString predictionLabel READ predictionLabel NOTIFY predictionChanged)
    Q_PROPERTY(double confidence READ confidence NOTIFY predictionChanged)
    Q_PROPERTY(qint64 inferenceMilliseconds READ inferenceMilliseconds NOTIFY predictionChanged)
    Q_PROPERTY(QVariantList topResults READ topResults NOTIFY predictionChanged)
    Q_PROPERTY(QVariantList demoImages READ demoImages CONSTANT)
    Q_PROPERTY(QVariantList availableModels READ availableModels NOTIFY modelListChanged)
    Q_PROPERTY(int currentModelIndex READ currentModelIndex NOTIFY modelChanged)
    Q_PROPERTY(QString statusText READ statusText NOTIFY statusChanged)
    Q_PROPERTY(QString errorText READ errorText NOTIFY statusChanged)

public:
    explicit AppController(QObject* parent = nullptr);
    ~AppController() override;

    [[nodiscard]] bool modelLoaded() const;
    [[nodiscard]] bool imageLoaded() const;
    [[nodiscard]] bool busy() const;
    [[nodiscard]] int classCount() const;
    [[nodiscard]] QString modelStatus() const;
    [[nodiscard]] QString modelPath() const;
    [[nodiscard]] QString labelsPath() const;
    [[nodiscard]] QString modelDetails() const;
    [[nodiscard]] QString applicationVersion() const;
    [[nodiscard]] QUrl imageUrl() const;
    [[nodiscard]] QString imagePath() const;
    [[nodiscard]] QString imageDetails() const;
    [[nodiscard]] QString predictionLabel() const;
    [[nodiscard]] double confidence() const;
    [[nodiscard]] qint64 inferenceMilliseconds() const;
    [[nodiscard]] QVariantList topResults() const;
    [[nodiscard]] QVariantList demoImages() const;
    [[nodiscard]] QString statusText() const;
    [[nodiscard]] QString errorText() const;

    Q_INVOKABLE void loadImage(const QUrl& url);
    Q_INVOKABLE void clearImage();
    Q_INVOKABLE void loadModel(const QUrl& url);
    Q_INVOKABLE void predict();
    Q_INVOKABLE void selectModelByPath(const QString& path);
    Q_INVOKABLE void refreshModelList();

signals:
    void modelChanged();
    void modelListChanged();
    void imageChanged();
    void busyChanged();
    void predictionChanged();
    void statusChanged();

private:
    void discoverResources();
    void applyModel(const QString& path);
    void clearPrediction();
    void setStatus(const QString& status, const QString& error = {});
    void finishPrediction();
    [[nodiscard]] QVariantList availableModels() const;
    [[nodiscard]] int currentModelIndex() const;
    void scanAvailableModels();
    [[nodiscard]] static QString localPath(const QUrl& url);

    std::shared_ptr<cppcnn::CNN> network_;
    cppcnn::ModelInfo modelInfo_;
    cppcnn::Tensor inputTensor_;
    QStringList labels_;
    QFutureWatcher<PredictionPayload> watcher_;
    QUrl imageUrl_;
    QString imagePath_;
    QString imageDetails_;
    QString modelPath_;
    QString labelsPath_;
    QString modelStatus_ = QStringLiteral("Model not loaded");
    QString modelDetails_;
    QString predictionLabel_ = QStringLiteral("Awaiting prediction");
    QString statusText_ = QStringLiteral("Starting...");
    QString errorText_;
    QVariantList topResults_;
    QVariantList demoImages_;
    QVariantList availableModels_;
    double confidence_ = 0.0;
    qint64 inferenceMilliseconds_ = 0;
    bool imageLoaded_ = false;
    bool busy_ = false;
};
