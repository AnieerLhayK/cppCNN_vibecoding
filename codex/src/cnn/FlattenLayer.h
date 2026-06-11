#pragma once

#include "cnn/Layer.h"

#include <cstddef>

namespace cppcnn {

class FlattenLayer final : public Layer {
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& outputGradient) override;

    [[nodiscard]] std::string type() const override;

private:
    std::size_t inputChannels_ = 0;
    std::size_t inputHeight_ = 0;
    std::size_t inputWidth_ = 0;
};

}  // namespace cppcnn
