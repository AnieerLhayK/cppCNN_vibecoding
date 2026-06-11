#include "cnn/FlattenLayer.h"

#include <stdexcept>

namespace cppcnn {

Tensor FlattenLayer::forward(const Tensor& input) {
    if (input.empty()) {
        throw std::invalid_argument("Flatten input cannot be empty.");
    }
    inputChannels_ = input.channels();
    inputHeight_ = input.height();
    inputWidth_ = input.width();

    Tensor output(1, 1, input.size());
    output.values() = input.values();
    return output;
}

Tensor FlattenLayer::backward(const Tensor& outputGradient) {
    if (outputGradient.size() != inputChannels_ * inputHeight_ * inputWidth_) {
        throw std::invalid_argument("Flatten output gradient dimensions are incorrect.");
    }

    Tensor inputGradient(inputChannels_, inputHeight_, inputWidth_);
    inputGradient.values() = outputGradient.values();
    return inputGradient;
}

std::string FlattenLayer::type() const {
    return "Flatten";
}

}  // namespace cppcnn
