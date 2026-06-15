#include "cnn/DropoutLayer.h"

#include <algorithm>
#include <random>
#include <stdexcept>

namespace cppcnn {

DropoutLayer::DropoutLayer(const float rate, const std::uint32_t seed)
    : rate_(rate), generator_(seed) {
    if (rate < 0.0F || rate >= 1.0F) {
        throw std::invalid_argument("Dropout rate must be in [0, 1).");
    }
    scale_ = 1.0F / (1.0F - rate_);
}

Tensor DropoutLayer::forward(const Tensor& input) {
    if (input.empty()) {
        throw std::invalid_argument("Dropout input cannot be empty.");
    }

    Tensor output(input.channels(), input.height(), input.width());
    lastSize_ = input.size();

    if (!training_) {
        // Inference: pass through unchanged
        std::copy(input.values().begin(), input.values().end(), output.values().begin());
        return output;
    }

    mask_.resize(input.size());
    std::bernoulli_distribution distribution(1.0F - rate_);

    for (std::size_t index = 0; index < input.size(); ++index) {
        if (distribution(generator_)) {
            output[index] = input[index] * scale_;
            mask_[index] = 1;
        } else {
            output[index] = 0.0F;
            mask_[index] = 0;
        }
    }
    return output;
}

Tensor DropoutLayer::backward(const Tensor& outputGradient) {
    if (outputGradient.size() != lastSize_) {
        throw std::invalid_argument("Dropout output gradient size is incorrect.");
    }

    Tensor inputGradient(
        outputGradient.channels(),
        outputGradient.height(),
        outputGradient.width());

    if (!training_) {
        std::copy(outputGradient.values().begin(), outputGradient.values().end(),
                  inputGradient.values().begin());
        return inputGradient;
    }

    for (std::size_t index = 0; index < outputGradient.size(); ++index) {
        inputGradient[index] = mask_[index] != 0U ? outputGradient[index] * scale_ : 0.0F;
    }
    return inputGradient;
}

void DropoutLayer::train(const bool enabled) {
    training_ = enabled;
}

std::string DropoutLayer::type() const {
    return "Dropout";
}

}  // namespace cppcnn
