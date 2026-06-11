#pragma once

#include "cnn/Layer.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cppcnn {

class ConvLayer final : public Layer {
public:
    ConvLayer(
        std::size_t inputChannels,
        std::size_t outputChannels,
        std::size_t kernelSize,
        std::size_t stride = 1,
        std::size_t padding = 0,
        std::uint32_t seed = 1);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& outputGradient) override;
    void zeroGrad() override;
    void update(float learningRate, float gradientScale, float weightDecay) override;

    [[nodiscard]] std::string type() const override;
    [[nodiscard]] const std::vector<float>& weights() const noexcept;
    [[nodiscard]] const std::vector<float>& biases() const noexcept;
    [[nodiscard]] std::vector<float>& mutableWeights() noexcept;
    [[nodiscard]] std::vector<float>& mutableBiases() noexcept;

private:
    [[nodiscard]] std::size_t weightOffset(
        std::size_t outputChannel,
        std::size_t inputChannel,
        std::size_t kernelRow,
        std::size_t kernelColumn) const;

    std::size_t inputChannels_;
    std::size_t outputChannels_;
    std::size_t kernelSize_;
    std::size_t stride_;
    std::size_t padding_;
    std::vector<float> weights_;
    std::vector<float> biases_;
    std::vector<float> weightGradients_;
    std::vector<float> biasGradients_;
    Tensor cachedInput_;
};

}  // namespace cppcnn
