#pragma once

#include "cnn/CNN.h"
#include "cnn/Loss.h"
#include "cnn/LRScheduler.h"
#include "data/DataLoader.h"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace cppcnn {

struct TrainingOptions {
    std::size_t epochs = 5;
    std::size_t batchSize = 16;
    float learningRate = 0.01F;
    float weightDecay = 0.0001F;
    float momentum = 0.9F;
    bool useMomentum = true;

    // Data augmentation
    bool enableAugmentation = false;

    // Class-balanced sampling
    bool enableClassBalancing = false;

    // Learning rate scheduling
    LRScheduler::Config lrScheduler;

    // Validation
    float validationRatio = 0.0F;       // 0 = no validation split
    bool useTrackSplit = true;          // use track-aware splitting

    // Early stopping
    std::size_t earlyStoppingPatience = 0;  // 0 = disabled

    // Checkpoint
    std::string checkpointPath;         // empty = no checkpoint saving
    bool saveBestOnly = true;
    bool resume = false;

    // Logging
    bool logCsvHistory = false;
    std::string csvHistoryPath;

    bool shuffle = true;
    std::uint32_t seed = 42;
};

struct PerClassMetrics {
    std::size_t classIndex = 0;
    std::size_t correct = 0;
    std::size_t total = 0;
    float accuracy = 0.0F;
};

struct EpochMetrics {
    std::size_t epoch = 0;
    std::size_t samples = 0;
    float averageLoss = 0.0F;
    float accuracy = 0.0F;
    float learningRate = 0.0F;
    float durationSeconds = 0.0F;

    // Validation (if validation set is used)
    bool hasValidation = false;
    std::size_t validationSamples = 0;
    float validationLoss = 0.0F;
    float validationAccuracy = 0.0F;

    // Per-class (if validation set is used)
    float meanClassAccuracy = 0.0F;
    float minClassAccuracy = 0.0F;
};

struct TrainingReport {
    std::vector<EpochMetrics> history;
    float bestValidationAccuracy = 0.0F;
    std::size_t bestEpoch = 0;
    float totalDurationSeconds = 0.0F;
    bool earlyStopped = false;
    std::size_t earlyStopEpoch = 0;

    // Confusion matrix (43x43 for GTSRB)
    std::vector<std::size_t> confusionMatrix;
    std::size_t classCount = 0;
};

class Trainer {
public:
    static TrainingReport train(
        CNN& network,
        const Dataset& trainingSet,
        const DataLoader& loader,
        const TrainingOptions& options,
        std::ostream* progress = nullptr);

    static EpochMetrics evaluate(
        CNN& network,
        const Dataset& dataset,
        const DataLoader& loader,
        std::size_t sampleLimit = 0);

    // Evaluate with per-class metrics
    static EpochMetrics evaluateDetailed(
        CNN& network,
        const Dataset& dataset,
        const DataLoader& loader,
        std::vector<PerClassMetrics>& perClass,
        ConfusionMatrix& confusion,
        std::size_t sampleLimit = 0);

    // Save checkpoint (model + optimizer state + epoch)
    static void saveCheckpoint(
        const CNN& network,
        const std::string& path,
        std::size_t epoch,
        float bestValAccuracy,
        const TrainingOptions& options);

    // Load checkpoint for resume
    static bool loadCheckpoint(
        CNN& network,
        const std::string& path,
        std::size_t& epoch,
        float& bestValAccuracy,
        TrainingOptions& options);

    // Write CSV header and row
    static void writeCsvHeader(std::ostream& output);
    static void writeCsvRow(std::ostream& output, const EpochMetrics& metrics);
};

}  // namespace cppcnn
