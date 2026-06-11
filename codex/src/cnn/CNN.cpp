#include "cnn/CNN.h"

#include "cnn/Activation.h"
#include "cnn/ConvLayer.h"
#include "cnn/FCLayer.h"
#include "cnn/FlattenLayer.h"
#include "cnn/Loss.h"
#include "cnn/PoolingLayer.h"

#include <stdexcept>

namespace cppcnn {

CNN::CNN(const std::size_t classCount, const std::uint32_t seed)
    : classCount_(classCount) {
    if (classCount == 0) {
        throw std::invalid_argument("CNN class count must be positive.");
    }

    // LeNet-style flow for 3 x 32 x 32 input:
    // 6 x 28 x 28 -> 6 x 14 x 14 -> 16 x 10 x 10 -> 16 x 5 x 5.
    layers_.push_back(std::make_unique<ConvLayer>(3, 6, 5, 1, 0, seed));
    layers_.push_back(std::make_unique<ReLULayer>());
    layers_.push_back(std::make_unique<MaxPoolingLayer>(2, 2));
    layers_.push_back(std::make_unique<ConvLayer>(6, 16, 5, 1, 0, seed + 1));
    layers_.push_back(std::make_unique<ReLULayer>());
    layers_.push_back(std::make_unique<MaxPoolingLayer>(2, 2));
    layers_.push_back(std::make_unique<FlattenLayer>());
    layers_.push_back(std::make_unique<FCLayer>(16 * 5 * 5, 120, seed + 2));
    layers_.push_back(std::make_unique<ReLULayer>());
    layers_.push_back(std::make_unique<FCLayer>(120, classCount_, seed + 3));
    layers_.push_back(std::make_unique<SoftmaxLayer>());
}

Tensor CNN::forward(const Tensor& input) {
    if (input.channels() != 3 || input.height() != 32 || input.width() != 32) {
        throw std::invalid_argument("CNN expects a normalized 3 x 32 x 32 input tensor.");
    }

    Tensor output = input;
    for (const auto& layer : layers_) {
        output = layer->forward(output);
    }
    return output;
}

Tensor CNN::backward(const Tensor& outputGradient) {
    Tensor gradient = outputGradient;
    for (auto iterator = layers_.rbegin(); iterator != layers_.rend(); ++iterator) {
        gradient = (*iterator)->backward(gradient);
    }
    return gradient;
}

void CNN::zeroGrad() {
    for (const auto& layer : layers_) {
        layer->zeroGrad();
    }
}

void CNN::update(
    const float learningRate,
    const float gradientScale,
    const float weightDecay) {
    for (const auto& layer : layers_) {
        layer->update(learningRate, gradientScale, weightDecay);
    }
}

std::size_t CNN::predict(const Tensor& input, float* confidence) {
    const Tensor probabilities = forward(input);
    const std::size_t prediction = argmax(probabilities);
    if (confidence != nullptr) {
        *confidence = probabilities[prediction];
    }
    return prediction;
}

std::size_t CNN::classCount() const noexcept {
    return classCount_;
}

std::size_t CNN::layerCount() const noexcept {
    return layers_.size();
}

}  // namespace cppcnn
