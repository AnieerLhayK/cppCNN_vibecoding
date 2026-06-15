#include "cnn/Activation.h"
#include "cnn/Augmenter.h"
#include "cnn/CNN.h"
#include "cnn/ConvLayer.h"
#include "cnn/DropoutLayer.h"
#include "cnn/FCLayer.h"
#include "cnn/FlattenLayer.h"
#include "cnn/Loss.h"
#include "cnn/LRScheduler.h"
#include "cnn/PoolingLayer.h"
#include "cnn/Tensor.h"
#include "cnn/Trainer.h"
#include "data/DataLoader.h"
#include "image/ImageProcessor.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {

void expect(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expectNear(const float actual, const float expected, const float tolerance, const char* message) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

void testTensor() {
    cppcnn::Tensor tensor(3, 2, 4, 0.25F);
    expect(tensor.channels() == 3, "Tensor channel count is incorrect.");
    expect(tensor.height() == 2, "Tensor height is incorrect.");
    expect(tensor.width() == 4, "Tensor width is incorrect.");
    expect(tensor.size() == 24, "Tensor element count is incorrect.");

    tensor.at(2, 1, 3) = 0.75F;
    expectNear(tensor.at(2, 1, 3), 0.75F, 1.0e-6F, "Tensor indexing failed.");

    bool threw = false;
    try {
        static_cast<void>(tensor.at(3, 0, 0));
    } catch (const std::out_of_range&) {
        threw = true;
    }
    expect(threw, "Tensor should reject an invalid channel index.");
}

void testImagePreprocessing() {
    cppcnn::RawImage image;
    image.width = 2;
    image.height = 2;
    image.channels = 3;
    image.pixels = {
        255, 0, 0, 0, 255, 0,
        0, 0, 255, 255, 255, 255,
    };

    const auto resized = cppcnn::ImageProcessor::resize(image, 4, 4);
    expect(resized.width == 4 && resized.height == 4, "Image resize dimensions are incorrect.");

    const auto tensor = cppcnn::ImageProcessor::normalize(resized);
    expect(tensor.channels() == 3, "Normalized image channel count is incorrect.");
    expect(tensor.height() == 4 && tensor.width() == 4, "Normalized image dimensions are incorrect.");
    for (const float value : tensor.values()) {
        expect(value >= 0.0F && value <= 1.0F, "Normalized pixel is outside [0, 1].");
    }
    expectNear(tensor.at(0, 0, 0), 1.0F, 1.0e-6F, "RGB channel ordering is incorrect.");
}

void testLayerDimensions() {
    cppcnn::Tensor input(3, 32, 32, 0.5F);
    cppcnn::ConvLayer convolution(3, 6, 5, 1, 0, 7);
    cppcnn::ReLULayer relu;
    cppcnn::MaxPoolingLayer pooling(2, 2);
    cppcnn::FlattenLayer flatten;
    cppcnn::FCLayer fullyConnected(6 * 14 * 14, 10, 8);
    cppcnn::SoftmaxLayer softmax;

    const auto convolved = convolution.forward(input);
    expect(
        convolved.channels() == 6 && convolved.height() == 28 && convolved.width() == 28,
        "Convolution output dimensions are incorrect.");
    const auto activated = relu.forward(convolved);
    const auto pooled = pooling.forward(activated);
    expect(
        pooled.channels() == 6 && pooled.height() == 14 && pooled.width() == 14,
        "Pooling output dimensions are incorrect.");
    const auto flattened = flatten.forward(pooled);
    expect(flattened.size() == 6 * 14 * 14, "Flatten output size is incorrect.");
    const auto logits = fullyConnected.forward(flattened);
    const auto probabilities = softmax.forward(logits);
    expect(probabilities.size() == 10, "Softmax output size is incorrect.");
    expectNear(
        std::accumulate(probabilities.values().begin(), probabilities.values().end(), 0.0F),
        1.0F,
        1.0e-5F,
        "Softmax probabilities do not sum to one.");

    const auto lossGradient = cppcnn::CrossEntropyLoss::gradient(probabilities, 3);
    const auto logitsGradient = softmax.backward(lossGradient);
    const auto flattenedGradient = fullyConnected.backward(logitsGradient);
    const auto pooledGradient = flatten.backward(flattenedGradient);
    const auto activatedGradient = pooling.backward(pooledGradient);
    const auto convolvedGradient = relu.backward(activatedGradient);
    const auto inputGradient = convolution.backward(convolvedGradient);
    expect(
        inputGradient.channels() == 3
            && inputGradient.height() == 32
            && inputGradient.width() == 32,
        "Backward dimension flow is incorrect.");
}

void testLeNetForward() {
    cppcnn::CNN network(10, 9, cppcnn::CNNArchitecture::LeNet);
    cppcnn::Tensor input(3, 32, 32, 0.25F);
    const auto probabilities = network.forward(input);
    expect(probabilities.size() == 10, "CNN output class count is incorrect.");
    expectNear(
        std::accumulate(probabilities.values().begin(), probabilities.values().end(), 0.0F),
        1.0F,
        1.0e-5F,
        "CNN output probabilities do not sum to one.");
}

void testEnhancedForward() {
    cppcnn::CNN network(43, 9, cppcnn::CNNArchitecture::Enhanced);
    cppcnn::Tensor input(3, 32, 32, 0.25F);
    const auto probabilities = network.forward(input);
    expect(probabilities.size() == 43, "Enhanced CNN output class count is incorrect.");
    expectNear(
        std::accumulate(probabilities.values().begin(), probabilities.values().end(), 0.0F),
        1.0F,
        1.0e-5F,
        "Enhanced CNN output probabilities do not sum to one.");

    // Test training mode toggle
    network.setTraining(false);
    const auto evalProb = network.forward(input);
    expectNear(
        std::accumulate(evalProb.values().begin(), evalProb.values().end(), 0.0F),
        1.0F,
        1.0e-5F,
        "Eval mode CNN probabilities do not sum to one.");
}

void testDropoutLayer() {
    cppcnn::DropoutLayer dropout(0.5F, 12345);
    cppcnn::Tensor input(3, 4, 4, 1.0F);

    // Training mode: should drop some values (scale = 1/(1-0.5) = 2.0)
    dropout.train(true);
    const auto trainOutput = dropout.forward(input);
    float sumTrain = 0.0F;
    float zeroCount = 0.0F;
    for (const auto& v : trainOutput.values()) {
        sumTrain += v;
        if (v == 0.0F) zeroCount += 1.0F;
    }
    expect(sumTrain > 0.0F, "Dropout training output should have some values.");
    expect(zeroCount > 0.0F, "Dropout training should drop some values.");
    expect(zeroCount < static_cast<float>(trainOutput.size()), "Dropout should keep some values.");

    // Eval mode: pass through
    dropout.train(false);
    const auto evalOutput = dropout.forward(input);
    float sumEval = 0.0F;
    for (const auto& v : evalOutput.values()) sumEval += v;
    expectNear(sumEval, static_cast<float>(input.size()), 1.0e-4F, "Dropout eval mode should pass through.");
}

void testLRScheduler() {
    cppcnn::LRScheduler::Config config;
    config.type = cppcnn::LRScheduler::Type::Step;
    config.initialLR = 0.01F;
    config.decayFactor = 0.1F;
    config.stepSize = 30;

    cppcnn::LRScheduler scheduler(config);
    expectNear(scheduler.getLR(0), 0.01F, 1.0e-6F, "Initial LR is incorrect.");
    expectNear(scheduler.getLR(29), 0.01F, 1.0e-6F, "LR before step is incorrect.");
    expectNear(scheduler.getLR(30), 0.001F, 1.0e-6F, "LR after step is incorrect.");
    expectNear(scheduler.getLR(60), 0.0001F, 1.0e-7F, "LR after second step is incorrect.");
}

void testAugmenter() {
    cppcnn::Tensor input(3, 8, 8, 0.5F);
    cppcnn::AugmentConfig config;
    config.enableRotation = true;
    config.enableTranslation = true;
    config.enableBrightness = true;
    config.enableContrast = true;
    config.enableNoise = true;

    cppcnn::Augmenter augmenter(config, 42);
    const auto augmented = augmenter.augment(input);

    expect(augmented.channels() == input.channels(), "Augmented channel count unchanged.");
    expect(augmented.height() == input.height(), "Augmented height unchanged.");
    expect(augmented.width() == input.width(), "Augmented width unchanged.");

    // Values should be in [0, 1]
    for (const auto& val : augmented.values()) {
        expect(val >= 0.0F && val <= 1.0F, "Augmented value outside [0, 1].");
    }
}

void testPerClassAccuracy() {
    cppcnn::PerClassAccuracy pca(5);

    // Mock probabilities
    cppcnn::Tensor probs(1, 1, 5, 0.1F);
    probs[0] = 0.6F; probs[1] = 0.1F; probs[2] = 0.1F; probs[3] = 0.1F; probs[4] = 0.1F;

    pca.add(probs, 0);  // correct
    pca.add(probs, 1);  // wrong (predicted 0)
    pca.add(probs, 0);  // correct

    expectNear(pca.classAccuracy(0), 1.0F, 1.0e-6F, "Class 0 accuracy.");
    expectNear(pca.classAccuracy(1), 0.0F, 1.0e-6F, "Class 1 accuracy.");
    expectNear(pca.meanAccuracy(), 0.5F, 1.0e-6F, "Mean accuracy (only 2 classes with samples).");
}

void testConfusionMatrix() {
    cppcnn::ConfusionMatrix cm(3);
    cm.add(0, 0);  // correct
    cm.add(1, 0);  // predicted 1, actual 0
    cm.add(2, 2);  // correct
    cm.add(1, 1);  // correct

    expect(cm.value(0, 0) == 1, "TP for class 0.");
    expect(cm.value(1, 0) == 1, "FP: predicted 1 for class 0.");
    expect(cm.value(1, 1) == 1, "TP for class 1.");
    expect(cm.value(2, 2) == 1, "TP for class 2.");
}

void testTrackSplit() {
    // Create a mock dataset with track IDs
    const auto root =
        std::filesystem::temp_directory_path() / "cppcnn_track_split_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "00000");

    // Create 6 test images with 2 tracks (00000 and 00001)
    for (int track = 0; track < 2; ++track) {
        for (int frame = 0; frame < 3; ++frame) {
            const std::string filename = std::to_string(track) + "_0000" + std::to_string(frame) + ".ppm";
            std::ofstream output(root / "00000" / filename);
            output << "P3\n2 2\n255\n0 0 0 0 0 0 0 0 0 0 0 0\n";
            output.close();
        }
    }

    cppcnn::DataLoaderOptions options;
    options.classLimit = 1;
    cppcnn::DataLoader loader(options);
    const auto dataset = loader.loadDirectory(root);

    expect(dataset.size() == 6, "Track test dataset should have 6 samples.");

    // Check track IDs
    int track0Count = 0, track1Count = 0;
    for (const auto& sample : dataset.samples) {
        if (sample.trackId == 0) ++track0Count;
        else if (sample.trackId == 1) ++track1Count;
    }
    expect(track0Count == 3, "Track 0 should have 3 samples.");
    expect(track1Count == 3, "Track 1 should have 3 samples.");

    // Test split
    const auto split = cppcnn::DataLoader::splitByTrack(dataset, 0.5F, 42);
    expect(!split.train.samples.empty(), "Split train should not be empty.");
    expect(!split.validation.samples.empty(), "Split val should not be empty.");

    // Verify no track is in both splits
    for (const auto& trainSample : split.train.samples) {
        for (const auto& valSample : split.validation.samples) {
            expect(trainSample.trackId != valSample.trackId
                   || trainSample.originalClassId != valSample.originalClassId,
                   "No track should appear in both train and val.");
        }
    }

    std::filesystem::remove_all(root);
}

void writeTestPpm(const std::filesystem::path& path, const int red, const int green, const int blue) {
    std::ofstream output(path);
    output << "P3\n2 2\n255\n";
    for (int index = 0; index < 4; ++index) {
        output << red << ' ' << green << ' ' << blue << '\n';
    }
}

void testDataLoader() {
    const auto root =
        std::filesystem::temp_directory_path() / "cppcnn_dataloader_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "00000");
    std::filesystem::create_directories(root / "00001");
    std::filesystem::create_directories(root / "00002");

    writeTestPpm(root / "00000" / "a.ppm", 255, 0, 0);
    writeTestPpm(root / "00000" / "b.ppm", 200, 0, 0);
    writeTestPpm(root / "00001" / "a.ppm", 0, 255, 0);
    writeTestPpm(root / "00001" / "b.ppm", 0, 200, 0);
    writeTestPpm(root / "00002" / "ignored.ppm", 0, 0, 255);

    cppcnn::DataLoaderOptions options;
    options.classLimit = 2;
    options.samplesPerClass = 1;
    cppcnn::DataLoader loader(options);
    const auto dataset = loader.loadDirectory(root);

    expect(dataset.classCount() == 2, "DataLoader class limit was not applied.");
    expect(dataset.size() == 2, "DataLoader per-class sample limit was not applied.");
    expect(dataset.classIds[0] == 0 && dataset.classIds[1] == 1, "Class IDs are incorrect.");
    expect(dataset.samples[0].label == 0, "Continuous label mapping is incorrect.");

    const auto tensor = loader.loadTensor(dataset.samples[0]);
    expect(
        tensor.channels() == 3 && tensor.height() == 32 && tensor.width() == 32,
        "DataLoader preprocessing dimensions are incorrect.");

    std::filesystem::remove_all(root);
}

void testOfficialGtsrbLayout() {
    const auto root =
        std::filesystem::temp_directory_path() / "cppcnn_official_gtsrb_test";
    std::filesystem::remove_all(root);

    const auto trainingImages =
        root / "GTSRB" / "Final_Training" / "Images";
    const auto testImages =
        root / "GTSRB" / "Final_Test" / "Images";
    std::filesystem::create_directories(trainingImages / "00000");
    std::filesystem::create_directories(trainingImages / "00001");
    std::filesystem::create_directories(testImages);

    writeTestPpm(trainingImages / "00000" / "train0.ppm", 255, 0, 0);
    writeTestPpm(trainingImages / "00001" / "train1.ppm", 0, 255, 0);
    writeTestPpm(testImages / "00000.ppm", 255, 0, 0);
    writeTestPpm(testImages / "00001.ppm", 0, 255, 0);

    std::ofstream csv(root / "GT-final_test.csv");
    csv << "Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId\n";
    csv << "00000.ppm;2;2;0;0;1;1;0\n";
    csv << "00001.ppm;2;2;0;0;1;1;1\n";
    csv.close();

    cppcnn::DataLoaderOptions options;
    options.classLimit = 2;
    cppcnn::DataLoader loader(options);
    const auto trainingSet =
        loader.loadDataset(root, cppcnn::DatasetSplit::Training);
    const auto testSet =
        loader.loadDataset(root, cppcnn::DatasetSplit::Test);

    expect(trainingSet.size() == 2, "Official GTSRB training layout was not detected.");
    expect(testSet.size() == 2, "Official GTSRB test CSV was not loaded.");
    expect(
        testSet.samples[0].originalClassId == 0
            && testSet.samples[1].originalClassId == 1,
        "Official GTSRB test labels are incorrect.");

    std::filesystem::remove_all(root);
}

void testModelPersistenceAndUpdate() {
    const auto modelPath =
        std::filesystem::temp_directory_path() / "cppcnn_model_test.bin";
    std::filesystem::remove(modelPath);

    cppcnn::CNN source(3, 21, cppcnn::CNNArchitecture::LeNet);
    cppcnn::Tensor input(3, 32, 32, 0.2F);
    const auto beforeSave = source.forward(input);
    source.saveModel(modelPath);
    expect(cppcnn::CNN::modelClassCount(modelPath) == 3, "Stored model class count is incorrect.");
    const auto modelInfo = cppcnn::CNN::inspectModel(modelPath);
    expect(modelInfo.version == 1, "Stored model version is incorrect.");
    expect(modelInfo.classCount == 3, "Inspected model class count is incorrect.");
    expect(modelInfo.trainableLayerCount == 4, "Inspected trainable layer count is incorrect.");
    expect(modelInfo.parameterCount > 50000, "Inspected model parameter count is too small.");

    cppcnn::CNN restored(3, 999, cppcnn::CNNArchitecture::LeNet);
    restored.loadModel(modelPath);
    const auto afterLoad = restored.forward(input);
    for (std::size_t index = 0; index < beforeSave.size(); ++index) {
        expectNear(
            afterLoad[index],
            beforeSave[index],
            1.0e-6F,
            "Loaded model prediction differs from the saved model.");
    }

    const float initialLoss = cppcnn::CrossEntropyLoss::value(afterLoad, 1);
    restored.backward(cppcnn::CrossEntropyLoss::gradient(afterLoad, 1));
    restored.update(0.01F);
    const auto afterUpdate = restored.forward(input);
    const float updatedLoss = cppcnn::CrossEntropyLoss::value(afterUpdate, 1);
    expect(std::isfinite(updatedLoss), "Training update produced a non-finite loss.");
    expect(updatedLoss <= initialLoss, "A single training step did not reduce sample loss.");

    std::filesystem::remove(modelPath);
}

void testMomentumUpdate() {
    cppcnn::CNN network(3, 21, cppcnn::CNNArchitecture::LeNet);
    cppcnn::Tensor input(3, 32, 32, 0.2F);
    const auto before = network.forward(input);
    network.backward(cppcnn::CrossEntropyLoss::gradient(before, 0));
    network.updateWithMomentum(0.01F, 1.0F, 0.0001F, 0.9F);
    const auto after = network.forward(input);
    const float beforeLoss = cppcnn::CrossEntropyLoss::value(before, 0);
    const float afterLoss = cppcnn::CrossEntropyLoss::value(after, 0);
    expect(std::isfinite(afterLoss), "Momentum update produced a non-finite loss.");
    expect(afterLoss <= beforeLoss, "Momentum update did not reduce sample loss.");
}

}  // namespace

int main() {
    try {
        testTensor();
        testImagePreprocessing();
        testLayerDimensions();
        testLeNetForward();
        testEnhancedForward();
        testDropoutLayer();
        testLRScheduler();
        testAugmenter();
        testPerClassAccuracy();
        testConfusionMatrix();
        testTrackSplit();
        testDataLoader();
        testOfficialGtsrbLayout();
        testModelPersistenceAndUpdate();
        testMomentumUpdate();
        std::cout << "All basic tests passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

