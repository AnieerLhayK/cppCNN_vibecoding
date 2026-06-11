#pragma once

#include "cnn/Layer.h"

#include <cstdint>
#include <vector>

namespace cppcnn {

class ReLULayer final : public Layer {
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& outputGradient) override;

    [[nodiscard]] std::string type() const override;

private:
    std::vector<std::uint8_t> active_;
    std::size_t channels_ = 0;
    std::size_t height_ = 0;
    std::size_t width_ = 0;
};

class SoftmaxLayer final : public Layer {
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& outputGradient) override;

    [[nodiscard]] std::string type() const override;

private:
    Tensor probabilities_;
};

}  // namespace cppcnn
