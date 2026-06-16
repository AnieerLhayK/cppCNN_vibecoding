#pragma once

#include "LibTorchModel.h"

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace cppcnn {

/// Metrics for a single training/validation epoch.
struct LibTorchEpochMetrics {
    std::size_t epoch = 0;
    float trainLoss = 0.0F;
    float trainAccuracy = 0.0F;
    float valLoss = 0.0F;
    float valAccuracy = 0.0F;
    float meanClassAccuracy = 0.0F;
    float minClassAccuracy = 0.0F;
    float durationSec = 0.0F;
    double currentLr = 0.0;
};

/// Summary of a complete training run.
struct LibTorchTrainingReport {
    std::vector<LibTorchEpochMetrics> history;
    float bestValAccuracy = 0.0F;
    std::size_t bestEpoch = 0;
    float totalDurationSec = 0.0F;
    std::size_t totalParams = 0;
};

/// Options for GPU training.
struct LibTorchTrainingOptions {
    std::size_t epochs = 10;
    std::size_t batchSize = 64;
    double learningRate = 0.01;
    double weightDecay = 0.0001;
    double momentum = 0.9;
    float validationRatio = 0.2F;
    bool enableAugmentation = false;
    bool enableClassBalancing = false;
    bool useCuda = true;
    bool verbose = true;
    std::uint32_t seed = 42;
    std::string checkpointPath;   ///< Path to save best model checkpoint (empty = skip)
    std::string csvPath;          ///< Path for CSV epoch history (empty = skip)
	// LR scheduler (StepLR)
	bool enableLrScheduler = false;
	double lrDecayFactor = 0.1;
	std::size_t lrStepSize = 30;
	double minLr = 1.0e-6;
	std::size_t warmupEpochs = 0;
};

// ---------------------------------------------------------------------------
// Image + label pair type used for training
// ---------------------------------------------------------------------------
struct TrainSample {
    std::string imagePath;
    int64_t label;
};

/// Train a model using LibTorch on GPU (or CPU).
/// The model should already be .to(device).
LibTorchTrainingReport trainLibTorchModel(
    LibTorchModule& model,
    torch::Device device,
    const std::vector<TrainSample>& trainSamples,
    const std::vector<TrainSample>& valSamples,
    int64_t numClasses,
    const LibTorchTrainingOptions& opts);

/// Evaluate a model on a dataset.
/// @return {avgLoss, accuracy, meanClassAcc, minClassAcc}
std::tuple<float, float, float, float> evaluateLibTorchModel(
    LibTorchModule& model,
    torch::Device device,
    const std::vector<TrainSample>& samples,
    int64_t numClasses,
    std::size_t batchSize = 64);

}  // namespace cppcnn
