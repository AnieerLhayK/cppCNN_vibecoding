#include "cnn/Trainer.h"

#include "cnn/Augmenter.h"
#include "cnn/Loss.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cppcnn {

namespace {

void validateDataset(const CNN& network, const Dataset& dataset) {
    if (dataset.empty()) {
        throw std::invalid_argument("Training or evaluation dataset is empty.");
    }
    if (dataset.classCount() != network.classCount()) {
        throw std::invalid_argument(
            "Dataset class count does not match the CNN output class count.");
    }
}

Tensor loadSample(
    const DataLoader& loader,
    const DataSample& sample) {
    try {
        return loader.loadTensor(sample);
    } catch (const std::exception& error) {
        throw std::runtime_error(
            "Failed to load dataset image '" + sample.imagePath.string() + "': " + error.what());
    }
}

// Build class-balanced order: oversample minority classes
std::vector<std::size_t> buildBalancedOrder(
    const Dataset& dataset,
    const std::uint32_t seed) {
    // Count samples per class
    std::vector<std::size_t> classCounts(dataset.classCount(), 0);
    for (const auto& sample : dataset.samples) {
        if (sample.label < dataset.classCount()) {
            ++classCounts[sample.label];
        }
    }

    // Find max count
    std::size_t maxCount = 0;
    for (const auto& count : classCounts) {
        maxCount = std::max(maxCount, count);
    }

    // Build balanced indices
    std::vector<std::vector<std::size_t>> classIndices(dataset.classCount());
    for (std::size_t idx = 0; idx < dataset.samples.size(); ++idx) {
        classIndices[dataset.samples[idx].label].push_back(idx);
    }

    std::vector<std::size_t> balancedOrder;
    balancedOrder.reserve(maxCount * dataset.classCount());

    std::mt19937 generator(seed);
    for (std::size_t cls = 0; cls < dataset.classCount(); ++cls) {
        if (classIndices[cls].empty()) continue;
        // Repeat indices to match maxCount
        for (std::size_t i = 0; i < maxCount; ++i) {
            balancedOrder.push_back(classIndices[cls][i % classIndices[cls].size()]);
        }
        // Shuffle within class for randomness
        std::shuffle(
            balancedOrder.end() - static_cast<std::ptrdiff_t>(maxCount),
            balancedOrder.end(),
            generator);
    }

    // Final overall shuffle
    std::shuffle(balancedOrder.begin(), balancedOrder.end(), generator);
    return balancedOrder;
}

}  // namespace

TrainingReport Trainer::train(
    CNN& network,
    const Dataset& trainingSet,
    const DataLoader& loader,
    const TrainingOptions& options,
    std::ostream* progress) {
    validateDataset(network, trainingSet);
    if (options.epochs == 0 || options.batchSize == 0) {
        throw std::invalid_argument("Epoch count and batch size must be positive.");
    }
    if (options.learningRate <= 0.0F || options.weightDecay < 0.0F) {
        throw std::invalid_argument("Learning rate must be positive and weight decay non-negative.");
    }
    if (options.momentum < 0.0F || options.momentum >= 1.0F) {
        throw std::invalid_argument("Momentum must be in [0, 1).");
    }

    TrainingReport report;
    report.classCount = trainingSet.classCount();

    // Learning rate scheduler
    LRScheduler::Config lrConfig = options.lrScheduler;
    if (lrConfig.initialLR <= 0.0F) {
        lrConfig.initialLR = options.learningRate;
    }
    LRScheduler scheduler(lrConfig);

    // Data augmentation
    Augmenter augmenter(AugmentConfig{}, options.seed);

    // Validation set (track-based split)
    Dataset validationSet;
    TrainValSplit split;
    bool hasValidation = options.validationRatio > 0.0F;

    if (hasValidation) {
        if (options.useTrackSplit) {
            split = DataLoader::splitByTrack(trainingSet, options.validationRatio, options.seed);
        } else {
            // Simple random split
            std::vector<std::size_t> indices(trainingSet.samples.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::mt19937 gen(options.seed);
            std::shuffle(indices.begin(), indices.end(), gen);

            std::size_t valSize = static_cast<std::size_t>(trainingSet.samples.size() * options.validationRatio);
            for (std::size_t i = 0; i < trainingSet.samples.size(); ++i) {
                if (i < valSize) {
                    split.validation.samples.push_back(trainingSet.samples[indices[i]]);
                } else {
                    split.train.samples.push_back(trainingSet.samples[indices[i]]);
                }
            }
            split.train.classIds = trainingSet.classIds;
            split.validation.classIds = trainingSet.classIds;
        }
        validationSet = split.validation;
        // Use a local copy of the training set for the modified split
        // Since we can't modify the original, we work with split.train
    }

    // CSV history
    std::ofstream csvOutput;
    if (options.logCsvHistory && !options.csvHistoryPath.empty()) {
        csvOutput.open(options.csvHistoryPath);
        if (csvOutput) {
            writeCsvHeader(csvOutput);
        }
    }

    // Restore state for resume
    std::size_t startEpoch = 1;
    float bestValAccuracy = 0.0F;
    if (options.resume && !options.checkpointPath.empty()) {
        std::size_t savedEpoch = 0;
        TrainingOptions savedOpts;
        if (loadCheckpoint(const_cast<CNN&>(network), options.checkpointPath,
                           savedEpoch, bestValAccuracy, savedOpts)) {
            startEpoch = savedEpoch + 1;
            report.bestEpoch = savedEpoch;
            if (progress) {
                *progress << "Resumed from epoch " << savedEpoch
                         << " (best val acc: " << std::fixed << std::setprecision(2)
                         << bestValAccuracy * 100.0F << "%)\n";
            }
        }
    }

    std::vector<EpochMetrics> history;
    history.reserve(options.epochs);

    std::mt19937 generator(options.seed);
    std::vector<std::size_t> order;
    float currentLR = options.learningRate;

    const auto trainStart = std::chrono::steady_clock::now();
    bool earlyStopped = false;
    std::size_t earlyStopCounter = 0;

    for (std::size_t epoch = startEpoch; epoch <= options.epochs; ++epoch) {
        currentLR = scheduler.getLR(epoch - 1);
        const auto epochStart = std::chrono::steady_clock::now();

        // Build training order
        const Dataset& currentTrainingSet = hasValidation ? split.train : trainingSet;

        if (options.enableClassBalancing) {
            order = buildBalancedOrder(currentTrainingSet, options.seed + static_cast<std::uint32_t>(epoch));
        } else {
            order.resize(currentTrainingSet.size());
            std::iota(order.begin(), order.end(), 0);
            if (options.shuffle) {
                std::shuffle(order.begin(), order.end(), generator);
            }
        }

        network.setTraining(true);
        network.zeroGrad();
        float totalLoss = 0.0F;
        Accuracy accuracy;
        std::size_t currentBatchSize = 0;

        for (std::size_t position = 0; position < order.size(); ++position) {
            const DataSample& sample = currentTrainingSet.samples[order[position]];
            Tensor input = loadSample(loader, sample);

            // Apply augmentation during training
            if (options.enableAugmentation) {
                input = augmenter.augment(input);
            }

            const Tensor probabilities = network.forward(input);

            totalLoss += CrossEntropyLoss::value(probabilities, sample.label);
            accuracy.add(probabilities, sample.label);
            network.backward(CrossEntropyLoss::gradient(probabilities, sample.label));
            ++currentBatchSize;

            const bool batchFinished =
                currentBatchSize == options.batchSize || position + 1 == order.size();
            if (batchFinished) {
                const float gradientScale = 1.0F / static_cast<float>(currentBatchSize);
                if (options.useMomentum && options.momentum > 0.0F) {
                    network.updateWithMomentum(
                        currentLR, gradientScale, options.weightDecay, options.momentum);
                } else {
                    network.update(currentLR, gradientScale, options.weightDecay);
                }
                currentBatchSize = 0;
            }
        }

        const auto epochDuration = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - epochStart).count();

        EpochMetrics metrics;
        metrics.epoch = epoch;
        metrics.samples = order.size();
        metrics.averageLoss = totalLoss / static_cast<float>(order.size());
        metrics.accuracy = accuracy.value();
        metrics.learningRate = currentLR;
        metrics.durationSeconds = epochDuration;

        // Validation
        if (hasValidation) {
            std::vector<PerClassMetrics> perClass;
            ConfusionMatrix valConfusion(validationSet.classCount());
            const auto valMetrics = evaluateDetailed(
                network, validationSet, loader, perClass, valConfusion);

            metrics.hasValidation = true;
            metrics.validationSamples = valMetrics.samples;
            metrics.validationLoss = valMetrics.averageLoss;
            metrics.validationAccuracy = valMetrics.accuracy;
            metrics.meanClassAccuracy = 0.0F;
            metrics.minClassAccuracy = 0.0F;

            float meanSum = 0.0F;
            float minVal = 1.0F;
            std::size_t validCount = 0;
            for (const auto& pc : perClass) {
                if (pc.total > 0) {
                    meanSum += pc.accuracy;
                    minVal = std::min(minVal, pc.accuracy);
                    ++validCount;
                }
            }
            if (validCount > 0) {
                metrics.meanClassAccuracy = meanSum / static_cast<float>(validCount);
                metrics.minClassAccuracy = minVal;
            }

            // Update best validation accuracy and save checkpoint
            if (report.bestEpoch == 0
                || metrics.validationAccuracy > bestValAccuracy) {
                bestValAccuracy = metrics.validationAccuracy;
                report.bestEpoch = epoch;
                earlyStopCounter = 0;

                // Save best checkpoint
                if (!options.checkpointPath.empty() && options.saveBestOnly) {
                    saveCheckpoint(
                        network, options.checkpointPath, epoch, bestValAccuracy, options);
                }
            } else {
                ++earlyStopCounter;
            }

            // Update report confusion matrix from best epoch
            if (epoch == report.bestEpoch) {
                const auto& cmData = valConfusion.data();
                report.confusionMatrix = cmData;
            }

            // Early stopping
            if (options.earlyStoppingPatience > 0
                && earlyStopCounter >= options.earlyStoppingPatience) {
                earlyStopped = true;
                report.earlyStopEpoch = epoch;
                if (progress) {
                    *progress << "Early stopping triggered at epoch " << epoch << "\n";
                }
            }
        } else {
            // Save periodic checkpoint (not best-only)
            if (!options.checkpointPath.empty() && !options.saveBestOnly) {
                if (epoch % 5 == 0 || epoch == options.epochs) {
                    saveCheckpoint(network, options.checkpointPath, epoch, 0.0F, options);
                }
            }
        }

        history.push_back(metrics);

        // Progress output
        if (progress != nullptr) {
            *progress << "Epoch " << epoch << '/' << options.epochs
                     << " - loss: " << std::fixed << std::setprecision(4) << metrics.averageLoss
                     << " - acc: " << std::setprecision(2) << metrics.accuracy * 100.0F << "%"
                     << " - lr: " << std::setprecision(6) << currentLR
                     << " - " << std::setprecision(1) << epochDuration << "s";
            if (hasValidation) {
                *progress << " - val_loss: " << std::setprecision(4) << metrics.validationLoss
                         << " - val_acc: " << std::setprecision(2) << metrics.validationAccuracy * 100.0F << "%"
                         << " - mean_cls: " << std::setprecision(2) << metrics.meanClassAccuracy * 100.0F << "%"
                         << " - min_cls: " << std::setprecision(2) << metrics.minClassAccuracy * 100.0F << "%";
            }
            *progress << "\n";
        }

        // CSV row
        if (csvOutput) {
            writeCsvRow(csvOutput, metrics);
            csvOutput.flush();
        }

        scheduler.onEpochEnd();

        // Early stop
        if (earlyStopped) {
            break;
        }
    }

    report.totalDurationSeconds = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - trainStart).count();
    report.history = std::move(history);
    report.earlyStopped = earlyStopped;
    report.bestValidationAccuracy = bestValAccuracy;

    return report;
}

EpochMetrics Trainer::evaluate(
    CNN& network,
    const Dataset& dataset,
    const DataLoader& loader,
    const std::size_t sampleLimit) {
    validateDataset(network, dataset);
    const bool wasTraining = network.isTraining();
    network.setTraining(false);
    const std::size_t count =
        sampleLimit == 0 ? dataset.size() : std::min(sampleLimit, dataset.size());

    float totalLoss = 0.0F;
    Accuracy accuracy;
    for (std::size_t index = 0; index < count; ++index) {
        const DataSample& sample = dataset.samples[index];
        const Tensor probabilities = network.forward(loadSample(loader, sample));
        totalLoss += CrossEntropyLoss::value(probabilities, sample.label);
        accuracy.add(probabilities, sample.label);
    }

    EpochMetrics metrics;
    metrics.samples = count;
    metrics.averageLoss = totalLoss / static_cast<float>(count);
    metrics.accuracy = accuracy.value();
    network.setTraining(wasTraining);
    return metrics;
}

EpochMetrics Trainer::evaluateDetailed(
    CNN& network,
    const Dataset& dataset,
    const DataLoader& loader,
    std::vector<PerClassMetrics>& perClass,
    ConfusionMatrix& confusion,
    const std::size_t sampleLimit) {
    validateDataset(network, dataset);
    const bool wasTraining = network.isTraining();
    network.setTraining(false);
    const std::size_t count =
        sampleLimit == 0 ? dataset.size() : std::min(sampleLimit, dataset.size());

    // Initialize per-class tracking
    perClass.assign(dataset.classCount(), {});
    for (std::size_t c = 0; c < dataset.classCount(); ++c) {
        perClass[c].classIndex = c;
    }

    confusion = ConfusionMatrix(dataset.classCount());
    float totalLoss = 0.0F;
    Accuracy accuracy;

    for (std::size_t index = 0; index < count; ++index) {
        const DataSample& sample = dataset.samples[index];
        const Tensor probabilities = network.forward(loadSample(loader, sample));
        const std::size_t predicted = argmax(probabilities);

        totalLoss += CrossEntropyLoss::value(probabilities, sample.label);
        accuracy.add(probabilities, sample.label);
        confusion.add(predicted, sample.label);

        if (sample.label < perClass.size()) {
            if (predicted == sample.label) {
                ++perClass[sample.label].correct;
            }
            ++perClass[sample.label].total;
        }
    }

    for (auto& pc : perClass) {
        pc.accuracy = pc.total == 0
            ? 0.0F
            : static_cast<float>(pc.correct) / static_cast<float>(pc.total);
    }

    EpochMetrics metrics;
    metrics.samples = count;
    metrics.averageLoss = totalLoss / static_cast<float>(count);
    metrics.accuracy = accuracy.value();
    network.setTraining(wasTraining);
    return metrics;
}

void Trainer::saveCheckpoint(
    const CNN& network,
    const std::string& path,
    const std::size_t epoch,
    const float bestValAccuracy,
    const TrainingOptions& options) {
    // We save model and optimizer state in an extended format
    // First save the normal model
    network.saveModel(std::filesystem::path(path));

    // Then append optimizer state (velocity buffers) and metadata
    std::ofstream extra(path + ".opt", std::ios::binary);
    if (!extra) return;

    // Write epoch and best accuracy
    const auto epoch64 = static_cast<std::uint64_t>(epoch);
    extra.write(reinterpret_cast<const char*>(&epoch64), sizeof(epoch64));
    extra.write(reinterpret_cast<const char*>(&bestValAccuracy), sizeof(bestValAccuracy));

    // Write training options needed for resume
    const auto optFlags = static_cast<std::uint64_t>(
        (options.useMomentum ? 1ULL : 0ULL)
        | (options.enableAugmentation ? 2ULL : 0ULL)
        | (options.enableClassBalancing ? 4ULL : 0ULL));
    extra.write(reinterpret_cast<const char*>(&optFlags), sizeof(optFlags));

    // Collect optimizer state from all trainable layers
    std::vector<float> optBuffer;
    network.saveOptimizerState(optBuffer);

    const auto stateSize = static_cast<std::uint64_t>(optBuffer.size());
    extra.write(reinterpret_cast<const char*>(&stateSize), sizeof(stateSize));
    extra.write(reinterpret_cast<const char*>(optBuffer.data()),
                static_cast<std::streamsize>(optBuffer.size() * sizeof(float)));
}

bool Trainer::loadCheckpoint(
    CNN& network,
    const std::string& path,
    std::size_t& epoch,
    float& bestValAccuracy,
    TrainingOptions& options) {
    // Load the model weights
    try {
        network.loadModel(std::filesystem::path(path));
    } catch (...) {
        return false;
    }

    // Load optimizer state from companion file
    std::ifstream extra(path + ".opt", std::ios::binary);
    if (!extra) return false;

    std::uint64_t epoch64 = 0;
    extra.read(reinterpret_cast<char*>(&epoch64), sizeof(epoch64));
    epoch = static_cast<std::size_t>(epoch64);

    extra.read(reinterpret_cast<char*>(&bestValAccuracy), sizeof(bestValAccuracy));

    std::uint64_t optFlags = 0;
    extra.read(reinterpret_cast<char*>(&optFlags), sizeof(optFlags));

    std::uint64_t stateSize = 0;
    extra.read(reinterpret_cast<char*>(&stateSize), sizeof(stateSize));
    if (!extra || stateSize != network.optimizerStateSize()) {
        return false;
    }

    std::vector<float> optBuffer(stateSize);
    extra.read(reinterpret_cast<char*>(optBuffer.data()),
               static_cast<std::streamsize>(stateSize * sizeof(float)));
    if (!extra) {
        return false;
    }

    network.loadOptimizerState(optBuffer.data());
    options.useMomentum = (optFlags & 1ULL) != 0;
    options.enableAugmentation = (optFlags & 2ULL) != 0;
    options.enableClassBalancing = (optFlags & 4ULL) != 0;

    return true;
}

void Trainer::writeCsvHeader(std::ostream& output) {
    output << "epoch,samples,loss,accuracy,lr,duration_sec"
           << ",val_samples,val_loss,val_accuracy"
           << ",mean_class_acc,min_class_acc\n";
}

void Trainer::writeCsvRow(std::ostream& output, const EpochMetrics& metrics) {
    output << metrics.epoch
           << ',' << metrics.samples
           << ',' << std::fixed << std::setprecision(6) << metrics.averageLoss
           << ',' << std::setprecision(6) << metrics.accuracy
           << ',' << std::setprecision(8) << metrics.learningRate
           << ',' << std::setprecision(2) << metrics.durationSeconds;

    if (metrics.hasValidation) {
        output << ',' << metrics.validationSamples
               << ',' << std::setprecision(6) << metrics.validationLoss
               << ',' << std::setprecision(6) << metrics.validationAccuracy
               << ',' << std::setprecision(6) << metrics.meanClassAccuracy
               << ',' << std::setprecision(6) << metrics.minClassAccuracy;
    } else {
        output << ",,,,,";
    }
    output << '\n';
}

}  // namespace cppcnn
