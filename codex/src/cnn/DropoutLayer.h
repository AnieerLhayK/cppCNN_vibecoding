#pragma once

#include "cnn/Layer.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace cppcnn {

class DropoutLayer final : public Layer {
public:
    explicit DropoutLayer(float rate = 0.5F, std::uint32_t seed = 42);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& outputGradient) override;
    void train(bool enabled) override;

    [[nodiscard]] std::string type() const override;

private:
    float rate_;          // dropout probability
    float scale_;         // 1.0F / (1.0F - rate_)
    bool training_ = true;
    std::mt19937 generator_;
    std::vector<std::uint8_t> mask_;
    std::size_t lastSize_ = 0;
};

}  // namespace cppcnn
