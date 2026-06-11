#include "cnn/Trainer.h"

#include "cnn/Loss.h"

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <random>
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

}  // namespace

std::vector<EpochMetrics> Trainer::train(
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

    std::vector<std::size_t> order(trainingSet.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 generator(options.seed);
    std::vector<EpochMetrics> history;
    history.reserve(options.epochs);

    for (std::size_t epoch = 1; epoch <= options.epochs; ++epoch) {
        if (options.shuffle) {
            std::shuffle(order.begin(), order.end(), generator);
        }

        network.zeroGrad();
        float totalLoss = 0.0F;
        Accuracy accuracy;
        std::size_t currentBatchSize = 0;

        for (std::size_t position = 0; position < order.size(); ++position) {
            const DataSample& sample = trainingSet.samples[order[position]];
            const Tensor input = loadSample(loader, sample);
            const Tensor probabilities = network.forward(input);

            totalLoss += CrossEntropyLoss::value(probabilities, sample.label);
            accuracy.add(probabilities, sample.label);
            network.backward(CrossEntropyLoss::gradient(probabilities, sample.label));
            ++currentBatchSize;

            const bool batchFinished =
                currentBatchSize == options.batchSize || position + 1 == order.size();
            if (batchFinished) {
                network.update(
                    options.learningRate,
                    1.0F / static_cast<float>(currentBatchSize),
                    options.weightDecay);
                currentBatchSize = 0;
            }
        }

        EpochMetrics metrics;
        metrics.epoch = epoch;
        metrics.samples = trainingSet.size();
        metrics.averageLoss = totalLoss / static_cast<float>(trainingSet.size());
        metrics.accuracy = accuracy.value();
        history.push_back(metrics);

        if (progress != nullptr) {
            *progress
                << "Epoch " << epoch << '/' << options.epochs
                << " - loss: " << std::fixed << std::setprecision(4) << metrics.averageLoss
                << " - accuracy: " << std::setprecision(2) << metrics.accuracy * 100.0F
                << "%\n";
        }
    }
    return history;
}

EpochMetrics Trainer::evaluate(
    CNN& network,
    const Dataset& dataset,
    const DataLoader& loader,
    const std::size_t sampleLimit) {
    validateDataset(network, dataset);
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
    return metrics;
}

}  // namespace cppcnn
