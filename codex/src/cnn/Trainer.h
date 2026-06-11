#pragma once

#include "cnn/CNN.h"
#include "data/DataLoader.h"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <vector>

namespace cppcnn {

struct TrainingOptions {
    std::size_t epochs = 5;
    std::size_t batchSize = 16;
    float learningRate = 0.01F;
    float weightDecay = 0.0001F;
    bool shuffle = true;
    std::uint32_t seed = 42;
};

struct EpochMetrics {
    std::size_t epoch = 0;
    std::size_t samples = 0;
    float averageLoss = 0.0F;
    float accuracy = 0.0F;
};

class Trainer {
public:
    static std::vector<EpochMetrics> train(
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
};

}  // namespace cppcnn
