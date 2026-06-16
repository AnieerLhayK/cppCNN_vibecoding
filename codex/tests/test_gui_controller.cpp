#include "cnn/CNN.h"
#include "gui/AppController.h"
#include "gui/ResourceLocator.h"
#include "image/ImageProcessor.h"

#include <QCoreApplication>
#include <QDir>
#include <QEventLoop>
#include <QImage>
#include <QTemporaryDir>
#include <QTimer>
#include <QUrl>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

void expect(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void testImageFormats(AppController& controller) {
    QTemporaryDir directory;
    expect(directory.isValid(), "Could not create the GUI image test directory.");

    QImage image(48, 36, QImage::Format_RGB32);
    image.fill(QColor(210, 35, 45));
    const QStringList formats = {
        QStringLiteral("ppm"),
        QStringLiteral("png"),
        QStringLiteral("jpg"),
        QStringLiteral("bmp"),
    };

    for (const QString& format : formats) {
        const QString path = directory.filePath(QStringLiteral("sample.") + format);
        expect(image.save(path), "Qt could not write a required test image format.");
        controller.loadImage(QUrl::fromLocalFile(path));
        expect(controller.imageLoaded(), "The GUI controller could not load a supported image.");
        expect(controller.errorText().isEmpty(), "A supported image produced an error state.");
    }
}

void testCompanionLabels() {
    QTemporaryDir directory;
    expect(directory.isValid(), "Could not create the companion-label test directory.");
    const QString modelPath = directory.filePath(QStringLiteral("semantic.bin"));
    const QString labelsPath = directory.filePath(QStringLiteral("semantic.labels.txt"));
    QFile labels(labelsPath);
    expect(
        labels.open(QIODevice::WriteOnly | QIODevice::Text),
        "Could not create companion labels.");
    labels.write("Yield\nStop\n");
    labels.close();

    expect(
        ResourceLocator::findLabelsForModel(modelPath) == labelsPath,
        "Model-specific companion labels were not selected.");
    const QStringList values = ResourceLocator::readLabels(labelsPath, 2);
    expect(
        values == QStringList({QStringLiteral("Yield"), QStringLiteral("Stop")}),
        "Companion labels were not read in model output order.");
}

void testModelAndPrediction(AppController& controller, const QString& sourceDirectory) {
    const QString modelPath =
        QDir(sourceDirectory).filePath(QStringLiteral("models/gtsrb_v2_subset10.bin"));
    if (!QFileInfo::exists(modelPath)) {
        std::cout << "GUI model inference checks skipped: local model is intentionally absent.\n";
        return;
    }

    expect(controller.modelLoaded(), "The GUI did not auto-discover the development model.");
    expect(controller.classCount() == 10, "The GUI model class count is incorrect.");
    expect(controller.applicationVersion() == QStringLiteral("2.0.0"), "The GUI version is incorrect.");
    expect(controller.demoImages().size() == 8, "The GUI demo class showcase is incomplete.");
    expect(
        controller.demoImages().front().toMap().value(QStringLiteral("name")).toString()
            == QStringLiteral("speed limit 30"),
        "The GUI demo label should omit its numeric sort prefix.");

    const QString demoPath = QDir(sourceDirectory).filePath(
        QStringLiteral("Release/demo_images/01_speed_limit_30.ppm"));
    expect(QFileInfo::exists(demoPath), "The GUI demo image is missing.");
    controller.loadImage(QUrl::fromLocalFile(demoPath));
    expect(controller.imageLoaded(), "The GUI could not load the PPM demo image.");

    const auto nativeModel = std::filesystem::path(modelPath.toStdWString());
    const auto modelInfo = cppcnn::CNN::inspectModel(nativeModel);
    cppcnn::CNN network(modelInfo.classCount, 42, modelInfo.architecture);
    network.loadModel(nativeModel);
    const auto input = cppcnn::ImageProcessor::loadAndPreprocess(
        std::filesystem::path(demoPath.toStdWString()));
    const int expectedClass = static_cast<int>(network.predict(input));

    QEventLoop loop;
    QTimer timeout;
    timeout.setSingleShot(true);
    QObject::connect(&timeout, &QTimer::timeout, &loop, &QEventLoop::quit);
    QObject::connect(&controller, &AppController::busyChanged, &loop, [&controller, &loop]() {
        if (!controller.busy()) {
            loop.quit();
        }
    });

    controller.predict();
    expect(controller.busy(), "Prediction was not started in the background.");
    timeout.start(15000);
    loop.exec();

    expect(!controller.busy(), "Background prediction did not finish.");
    expect(!controller.topResults().isEmpty(), "The GUI did not produce Top-3 results.");
    const int actualClass =
        controller.topResults().front().toMap().value(QStringLiteral("classId")).toInt();
    expect(actualClass == expectedClass, "GUI Top-1 differs from the CLI/core CNN result.");

    controller.loadModel(QUrl::fromLocalFile(
        QDir(sourceDirectory).filePath(QStringLiteral("models/missing-model.bin"))));
    expect(controller.modelLoaded(), "A failed model replacement must preserve the working model.");
    expect(!controller.errorText().isEmpty(), "A missing model must produce a visible error.");

    const QString semanticModelPath =
        QDir(sourceDirectory).filePath(QStringLiteral("models/gtsrb_v4_semantic10.bin"));
    if (QFileInfo::exists(semanticModelPath)) {
        controller.loadModel(QUrl::fromLocalFile(semanticModelPath));
        expect(
            controller.labelsPath().endsWith(QStringLiteral("gtsrb_v4_semantic10.labels.txt")),
            "The semantic model did not load its companion labels.");
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    QCoreApplication application(argc, argv);
    try {
        const QString sourceDirectory = QStringLiteral(CPPCNN_SOURCE_DIR);
        const QDir source(sourceDirectory);
        expect(
            QDir::setCurrent(QFileInfo(source.absolutePath()).dir().absolutePath()),
            "Could not set the GUI test working directory.");

        AppController controller;
        testImageFormats(controller);
        testCompanionLabels();
        testModelAndPrediction(controller, sourceDirectory);
        std::cout << "All GUI controller tests passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "GUI test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
