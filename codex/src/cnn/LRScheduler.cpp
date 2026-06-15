#include "cnn/LRScheduler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cppcnn {

LRScheduler::LRScheduler(Config config)
    : config_(std::move(config)) {
    if (config_.initialLR <= 0.0F) {
        throw std::invalid_argument("Initial learning rate must be positive.");
    }
    if (config_.decayFactor <= 0.0F || config_.decayFactor > 1.0F) {
        throw std::invalid_argument("Decay factor must be in (0, 1].");
    }
    if (config_.stepSize == 0) {
        throw std::invalid_argument("Step size must be positive.");
    }
}

float LRScheduler::getLR(const std::size_t epoch) const {
    float lr = config_.initialLR;

    // Warmup: linearly increase from minLR to initialLR
    if (config_.warmupEpochs > 0 && epoch < config_.warmupEpochs) {
        const float progress = static_cast<float>(epoch + 1) / static_cast<float>(config_.warmupEpochs);
        return config_.minLR + (config_.initialLR - config_.minLR) * progress;
    }

    switch (config_.type) {
    case Type::Step:
    case Type::MultiStep: {
        const std::size_t effectiveEpoch = epoch - config_.warmupEpochs;
        const float decay = std::pow(config_.decayFactor,
            static_cast<float>(effectiveEpoch / config_.stepSize));
        lr *= decay;
        break;
    }
    case Type::Cosine: {
        const std::size_t effectiveEpoch = epoch - config_.warmupEpochs;
        const float cosine = 0.5F * (1.0F + std::cos(
            static_cast<float>(effectiveEpoch) * 3.14159265F /
            static_cast<float>(std::max(config_.stepSize, std::size_t(1)))));
        lr = config_.minLR + (config_.initialLR - config_.minLR) * cosine;
        break;
    }
    default:
        break;
    }

    return std::max(lr, config_.minLR);
}

void LRScheduler::onEpochEnd() {
    ++currentEpoch_;
}

const LRScheduler::Config& LRScheduler::config() const noexcept {
    return config_;
}

}  // namespace cppcnn
