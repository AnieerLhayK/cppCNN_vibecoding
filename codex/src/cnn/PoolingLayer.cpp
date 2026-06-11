#include "cnn/PoolingLayer.h"

#include <limits>
#include <stdexcept>

namespace cppcnn {

MaxPoolingLayer::MaxPoolingLayer(
    const std::size_t windowSize,
    const std::size_t stride)
    : windowSize_(windowSize), stride_(stride) {
    if (windowSize == 0 || stride == 0) {
        throw std::invalid_argument("Pooling window and stride must be positive.");
    }
}

Tensor MaxPoolingLayer::forward(const Tensor& input) {
    if (input.empty() || input.height() < windowSize_ || input.width() < windowSize_) {
        throw std::invalid_argument("Pooling input is empty or smaller than its window.");
    }

    inputChannels_ = input.channels();
    inputHeight_ = input.height();
    inputWidth_ = input.width();

    const std::size_t outputHeight = (input.height() - windowSize_) / stride_ + 1;
    const std::size_t outputWidth = (input.width() - windowSize_) / stride_ + 1;
    Tensor output(input.channels(), outputHeight, outputWidth);
    maximumIndices_.resize(output.size());

    std::size_t outputIndex = 0;
    for (std::size_t channel = 0; channel < input.channels(); ++channel) {
        for (std::size_t outputRow = 0; outputRow < outputHeight; ++outputRow) {
            for (std::size_t outputColumn = 0; outputColumn < outputWidth; ++outputColumn) {
                float maximum = -std::numeric_limits<float>::infinity();
                std::size_t maximumIndex = 0;
                for (std::size_t row = 0; row < windowSize_; ++row) {
                    for (std::size_t column = 0; column < windowSize_; ++column) {
                        const std::size_t inputRow = outputRow * stride_ + row;
                        const std::size_t inputColumn = outputColumn * stride_ + column;
                        const float value = input.at(channel, inputRow, inputColumn);
                        if (value > maximum) {
                            maximum = value;
                            maximumIndex =
                                (channel * input.height() + inputRow) * input.width() + inputColumn;
                        }
                    }
                }
                output.at(channel, outputRow, outputColumn) = maximum;
                maximumIndices_[outputIndex++] = maximumIndex;
            }
        }
    }
    return output;
}

Tensor MaxPoolingLayer::backward(const Tensor& outputGradient) {
    if (maximumIndices_.empty()) {
        throw std::logic_error("Pooling backward called before forward.");
    }
    if (outputGradient.size() != maximumIndices_.size()) {
        throw std::invalid_argument("Pooling output gradient dimensions are incorrect.");
    }

    Tensor inputGradient(inputChannels_, inputHeight_, inputWidth_);
    for (std::size_t index = 0; index < outputGradient.size(); ++index) {
        inputGradient[maximumIndices_[index]] += outputGradient[index];
    }
    return inputGradient;
}

std::string MaxPoolingLayer::type() const {
    return "MaxPool";
}

}  // namespace cppcnn
