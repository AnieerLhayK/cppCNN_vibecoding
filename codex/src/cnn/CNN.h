#pragma once

#include "cnn/Layer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace cppcnn {

class CNN {
public:
    explicit CNN(std::size_t classCount = 10, std::uint32_t seed = 42);

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& outputGradient);
    void zeroGrad();
    void update(float learningRate, float gradientScale = 1.0F, float weightDecay = 0.0F);

    [[nodiscard]] std::size_t predict(const Tensor& input, float* confidence = nullptr);
    [[nodiscard]] std::size_t classCount() const noexcept;
    [[nodiscard]] std::size_t layerCount() const noexcept;

private:
    std::size_t classCount_;
    std::vector<std::unique_ptr<Layer>> layers_;
};

}  // namespace cppcnn
