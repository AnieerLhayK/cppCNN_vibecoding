#include "cnn/FCLayer.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace cppcnn {

FCLayer::FCLayer(
    const std::size_t inputSize,
    const std::size_t outputSize,
    const std::uint32_t seed)
    : inputSize_(inputSize), outputSize_(outputSize) {
    if (inputSize == 0 || outputSize == 0) {
        throw std::invalid_argument("Fully connected layer dimensions must be positive.");
    }

    weights_.resize(inputSize_ * outputSize_);
    biases_.assign(outputSize_, 0.0F);
    weightGradients_.assign(weights_.size(), 0.0F);
    biasGradients_.assign(outputSize_, 0.0F);
    weightVelocity_.assign(weights_.size(), 0.0F);
    biasVelocity_.assign(outputSize_, 0.0F);

    std::mt19937 generator(seed);
    const float standardDeviation = std::sqrt(2.0F / static_cast<float>(inputSize_));
    std::normal_distribution<float> distribution(0.0F, standardDeviation);
    for (float& weight : weights_) {
        weight = distribution(generator);
    }
}

Tensor FCLayer::forward(const Tensor& input) {
    if (input.size() != inputSize_) {
        throw std::invalid_argument("Fully connected input size does not match the layer.");
    }
    cachedInput_ = input;

    Tensor output(1, 1, outputSize_);
    for (std::size_t outputIndex = 0; outputIndex < outputSize_; ++outputIndex) {
        float sum = biases_[outputIndex];
        const std::size_t rowOffset = outputIndex * inputSize_;
        for (std::size_t inputIndex = 0; inputIndex < inputSize_; ++inputIndex) {
            sum += weights_[rowOffset + inputIndex] * input[inputIndex];
        }
        output[outputIndex] = sum;
    }
    return output;
}

Tensor FCLayer::backward(const Tensor& outputGradient) {
    if (cachedInput_.empty()) {
        throw std::logic_error("Fully connected backward called before forward.");
    }
    if (outputGradient.size() != outputSize_) {
        throw std::invalid_argument("Fully connected output gradient size is incorrect.");
    }

    Tensor inputGradient(
        cachedInput_.channels(),
        cachedInput_.height(),
        cachedInput_.width());
    for (std::size_t outputIndex = 0; outputIndex < outputSize_; ++outputIndex) {
        const float gradient = outputGradient[outputIndex];
        biasGradients_[outputIndex] += gradient;
        const std::size_t rowOffset = outputIndex * inputSize_;
        for (std::size_t inputIndex = 0; inputIndex < inputSize_; ++inputIndex) {
            weightGradients_[rowOffset + inputIndex] += cachedInput_[inputIndex] * gradient;
            inputGradient[inputIndex] += weights_[rowOffset + inputIndex] * gradient;
        }
    }
    return inputGradient;
}

void FCLayer::zeroGrad() {
    std::fill(weightGradients_.begin(), weightGradients_.end(), 0.0F);
    std::fill(biasGradients_.begin(), biasGradients_.end(), 0.0F);
}

void FCLayer::update(
    const float learningRate,
    const float gradientScale,
    const float weightDecay) {
    for (std::size_t index = 0; index < weights_.size(); ++index) {
        weights_[index] -= learningRate
            * (weightGradients_[index] * gradientScale + weightDecay * weights_[index]);
    }
    for (std::size_t index = 0; index < biases_.size(); ++index) {
        biases_[index] -= learningRate * biasGradients_[index] * gradientScale;
    }
    zeroGrad();
}

void FCLayer::updateWithMomentum(
    const float learningRate,
    const float gradientScale,
    const float weightDecay,
    const float momentum) {
    for (std::size_t index = 0; index < weights_.size(); ++index) {
        const float gradient = weightGradients_[index] * gradientScale + weightDecay * weights_[index];
        weightVelocity_[index] = momentum * weightVelocity_[index] + learningRate * gradient;
        weights_[index] -= weightVelocity_[index];
    }
    for (std::size_t index = 0; index < biases_.size(); ++index) {
        biasVelocity_[index] = momentum * biasVelocity_[index] + learningRate * biasGradients_[index] * gradientScale;
        biases_[index] -= biasVelocity_[index];
    }
    zeroGrad();
}

void FCLayer::saveOptimizerState(std::vector<float>& buffer) const {
    buffer.insert(buffer.end(), weightVelocity_.begin(), weightVelocity_.end());
    buffer.insert(buffer.end(), biasVelocity_.begin(), biasVelocity_.end());
}

void FCLayer::loadOptimizerState(const float*& cursor) {
    std::copy(cursor, cursor + weightVelocity_.size(), weightVelocity_.begin());
    cursor += weightVelocity_.size();
    std::copy(cursor, cursor + biasVelocity_.size(), biasVelocity_.begin());
    cursor += biasVelocity_.size();
}

std::size_t FCLayer::optimizerStateSize() const noexcept {
    return weightVelocity_.size() + biasVelocity_.size();
}

std::string FCLayer::type() const {
    return "FC";
}

const std::vector<float>& FCLayer::weights() const noexcept {
    return weights_;
}

const std::vector<float>& FCLayer::biases() const noexcept {
    return biases_;
}

std::vector<float>& FCLayer::mutableWeights() noexcept {
    return weights_;
}

std::vector<float>& FCLayer::mutableBiases() noexcept {
    return biases_;
}

}  // namespace cppcnn
