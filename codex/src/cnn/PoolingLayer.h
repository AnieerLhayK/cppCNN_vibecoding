#pragma once

#include "cnn/Layer.h"

#include <cstddef>
#include <vector>

namespace cppcnn {

class MaxPoolingLayer final : public Layer {
public:
    explicit MaxPoolingLayer(std::size_t windowSize = 2, std::size_t stride = 2);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& outputGradient) override;

    [[nodiscard]] std::string type() const override;

private:
    std::size_t windowSize_;
    std::size_t stride_;
    std::size_t inputChannels_ = 0;
    std::size_t inputHeight_ = 0;
    std::size_t inputWidth_ = 0;
    std::vector<std::size_t> maximumIndices_;
};

}  // namespace cppcnn
