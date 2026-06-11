#include "cnn/Activation.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace cppcnn {

Tensor ReLULayer::forward(const Tensor& input) {
    Tensor output = input;
    channels_ = input.channels();
    height_ = input.height();
    width_ = input.width();
    active_.resize(input.size());

    for (std::size_t index = 0; index < input.size(); ++index) {
        active_[index] = input[index] > 0.0F ? 1U : 0U;
        output[index] = std::max(0.0F, input[index]);
    }
    return output;
}

Tensor ReLULayer::backward(const Tensor& outputGradient) {
    if (outputGradient.size() != active_.size()) {
        throw std::invalid_argument("ReLU output gradient dimensions are incorrect.");
    }

    Tensor inputGradient(channels_, height_, width_);
    for (std::size_t index = 0; index < outputGradient.size(); ++index) {
        inputGradient[index] = active_[index] != 0U ? outputGradient[index] : 0.0F;
    }
    return inputGradient;
}

std::string ReLULayer::type() const {
    return "ReLU";
}

Tensor SoftmaxLayer::forward(const Tensor& input) {
    if (input.empty()) {
        throw std::invalid_argument("Softmax input cannot be empty.");
    }

    probabilities_.resize(input.channels(), input.height(), input.width());
    const float maximum = *std::max_element(input.values().begin(), input.values().end());
    float sum = 0.0F;
    for (std::size_t index = 0; index < input.size(); ++index) {
        probabilities_[index] = std::exp(input[index] - maximum);
        sum += probabilities_[index];
    }
    for (float& probability : probabilities_.values()) {
        probability /= sum;
    }
    return probabilities_;
}

Tensor SoftmaxLayer::backward(const Tensor& outputGradient) {
    if (outputGradient.size() != probabilities_.size()) {
        throw std::invalid_argument("Softmax output gradient dimensions are incorrect.");
    }

    const float weightedGradient = std::inner_product(
        outputGradient.values().begin(),
        outputGradient.values().end(),
        probabilities_.values().begin(),
        0.0F);

    Tensor inputGradient(
        probabilities_.channels(),
        probabilities_.height(),
        probabilities_.width());
    for (std::size_t index = 0; index < probabilities_.size(); ++index) {
        inputGradient[index] =
            probabilities_[index] * (outputGradient[index] - weightedGradient);
    }
    return inputGradient;
}

std::string SoftmaxLayer::type() const {
    return "Softmax";
}

}  // namespace cppcnn
