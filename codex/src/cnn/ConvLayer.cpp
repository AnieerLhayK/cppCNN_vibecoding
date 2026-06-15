#include "cnn/ConvLayer.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace cppcnn {

ConvLayer::ConvLayer(
    const std::size_t inputChannels,
    const std::size_t outputChannels,
    const std::size_t kernelSize,
    const std::size_t stride,
    const std::size_t padding,
    const std::uint32_t seed)
    : inputChannels_(inputChannels),
      outputChannels_(outputChannels),
      kernelSize_(kernelSize),
      stride_(stride),
      padding_(padding) {
    if (inputChannels == 0 || outputChannels == 0 || kernelSize == 0 || stride == 0) {
        throw std::invalid_argument("Convolution dimensions and stride must be positive.");
    }

    const std::size_t weightCount =
        outputChannels_ * inputChannels_ * kernelSize_ * kernelSize_;
    weights_.resize(weightCount);
    biases_.assign(outputChannels_, 0.0F);
    weightGradients_.assign(weightCount, 0.0F);
    biasGradients_.assign(outputChannels_, 0.0F);
    weightVelocity_.assign(weightCount, 0.0F);
    biasVelocity_.assign(outputChannels_, 0.0F);

    std::mt19937 generator(seed);
    const float standardDeviation = std::sqrt(
        2.0F / static_cast<float>(inputChannels_ * kernelSize_ * kernelSize_));
    std::normal_distribution<float> distribution(0.0F, standardDeviation);
    for (float& weight : weights_) {
        weight = distribution(generator);
    }
}

Tensor ConvLayer::forward(const Tensor& input) {
    if (input.channels() != inputChannels_) {
        throw std::invalid_argument("Convolution input channel count does not match the layer.");
    }
    if (input.height() + 2 * padding_ < kernelSize_
        || input.width() + 2 * padding_ < kernelSize_) {
        throw std::invalid_argument("Convolution kernel is larger than the padded input.");
    }

    const std::size_t outputHeight =
        (input.height() + 2 * padding_ - kernelSize_) / stride_ + 1;
    const std::size_t outputWidth =
        (input.width() + 2 * padding_ - kernelSize_) / stride_ + 1;
    Tensor output(outputChannels_, outputHeight, outputWidth);
    cachedInput_ = input;

    for (std::size_t outputChannel = 0; outputChannel < outputChannels_; ++outputChannel) {
        for (std::size_t outputRow = 0; outputRow < outputHeight; ++outputRow) {
            for (std::size_t outputColumn = 0; outputColumn < outputWidth; ++outputColumn) {
                float sum = biases_[outputChannel];
                for (std::size_t inputChannel = 0; inputChannel < inputChannels_; ++inputChannel) {
                    for (std::size_t kernelRow = 0; kernelRow < kernelSize_; ++kernelRow) {
                        for (std::size_t kernelColumn = 0; kernelColumn < kernelSize_; ++kernelColumn) {
                            const auto inputRow =
                                static_cast<std::ptrdiff_t>(outputRow * stride_ + kernelRow)
                                - static_cast<std::ptrdiff_t>(padding_);
                            const auto inputColumn =
                                static_cast<std::ptrdiff_t>(outputColumn * stride_ + kernelColumn)
                                - static_cast<std::ptrdiff_t>(padding_);
                            if (inputRow < 0 || inputColumn < 0
                                || inputRow >= static_cast<std::ptrdiff_t>(input.height())
                                || inputColumn >= static_cast<std::ptrdiff_t>(input.width())) {
                                continue;
                            }
                            sum += input.at(
                                       inputChannel,
                                       static_cast<std::size_t>(inputRow),
                                       static_cast<std::size_t>(inputColumn))
                                * weights_[weightOffset(
                                    outputChannel,
                                    inputChannel,
                                    kernelRow,
                                    kernelColumn)];
                        }
                    }
                }
                output.at(outputChannel, outputRow, outputColumn) = sum;
            }
        }
    }
    return output;
}

Tensor ConvLayer::backward(const Tensor& outputGradient) {
    if (cachedInput_.empty()) {
        throw std::logic_error("Convolution backward called before forward.");
    }
    const std::size_t expectedHeight =
        (cachedInput_.height() + 2 * padding_ - kernelSize_) / stride_ + 1;
    const std::size_t expectedWidth =
        (cachedInput_.width() + 2 * padding_ - kernelSize_) / stride_ + 1;
    if (outputGradient.channels() != outputChannels_
        || outputGradient.height() != expectedHeight
        || outputGradient.width() != expectedWidth) {
        throw std::invalid_argument("Convolution output gradient dimensions are incorrect.");
    }

    Tensor inputGradient(
        cachedInput_.channels(),
        cachedInput_.height(),
        cachedInput_.width());

    for (std::size_t outputChannel = 0; outputChannel < outputChannels_; ++outputChannel) {
        for (std::size_t outputRow = 0; outputRow < expectedHeight; ++outputRow) {
            for (std::size_t outputColumn = 0; outputColumn < expectedWidth; ++outputColumn) {
                const float gradient = outputGradient.at(outputChannel, outputRow, outputColumn);
                biasGradients_[outputChannel] += gradient;

                for (std::size_t inputChannel = 0; inputChannel < inputChannels_; ++inputChannel) {
                    for (std::size_t kernelRow = 0; kernelRow < kernelSize_; ++kernelRow) {
                        for (std::size_t kernelColumn = 0; kernelColumn < kernelSize_; ++kernelColumn) {
                            const auto inputRow =
                                static_cast<std::ptrdiff_t>(outputRow * stride_ + kernelRow)
                                - static_cast<std::ptrdiff_t>(padding_);
                            const auto inputColumn =
                                static_cast<std::ptrdiff_t>(outputColumn * stride_ + kernelColumn)
                                - static_cast<std::ptrdiff_t>(padding_);
                            if (inputRow < 0 || inputColumn < 0
                                || inputRow >= static_cast<std::ptrdiff_t>(cachedInput_.height())
                                || inputColumn >= static_cast<std::ptrdiff_t>(cachedInput_.width())) {
                                continue;
                            }

                            const std::size_t weightIndex = weightOffset(
                                outputChannel,
                                inputChannel,
                                kernelRow,
                                kernelColumn);
                            weightGradients_[weightIndex] +=
                                cachedInput_.at(
                                    inputChannel,
                                    static_cast<std::size_t>(inputRow),
                                    static_cast<std::size_t>(inputColumn))
                                * gradient;
                            inputGradient.at(
                                inputChannel,
                                static_cast<std::size_t>(inputRow),
                                static_cast<std::size_t>(inputColumn)) +=
                                weights_[weightIndex] * gradient;
                        }
                    }
                }
            }
        }
    }
    return inputGradient;
}

void ConvLayer::zeroGrad() {
    std::fill(weightGradients_.begin(), weightGradients_.end(), 0.0F);
    std::fill(biasGradients_.begin(), biasGradients_.end(), 0.0F);
}

void ConvLayer::update(
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

void ConvLayer::updateWithMomentum(
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

void ConvLayer::saveOptimizerState(std::vector<float>& buffer) const {
    buffer.insert(buffer.end(), weightVelocity_.begin(), weightVelocity_.end());
    buffer.insert(buffer.end(), biasVelocity_.begin(), biasVelocity_.end());
}

void ConvLayer::loadOptimizerState(const float*& cursor) {
    std::copy(cursor, cursor + weightVelocity_.size(), weightVelocity_.begin());
    cursor += weightVelocity_.size();
    std::copy(cursor, cursor + biasVelocity_.size(), biasVelocity_.begin());
    cursor += biasVelocity_.size();
}

std::size_t ConvLayer::optimizerStateSize() const noexcept {
    return weightVelocity_.size() + biasVelocity_.size();
}

std::string ConvLayer::type() const {
    return "Conv";
}

const std::vector<float>& ConvLayer::weights() const noexcept {
    return weights_;
}

const std::vector<float>& ConvLayer::biases() const noexcept {
    return biases_;
}

std::vector<float>& ConvLayer::mutableWeights() noexcept {
    return weights_;
}

std::vector<float>& ConvLayer::mutableBiases() noexcept {
    return biases_;
}

std::size_t ConvLayer::weightOffset(
    const std::size_t outputChannel,
    const std::size_t inputChannel,
    const std::size_t kernelRow,
    const std::size_t kernelColumn) const {
    return ((outputChannel * inputChannels_ + inputChannel) * kernelSize_ + kernelRow)
        * kernelSize_
        + kernelColumn;
}

}  // namespace cppcnn
