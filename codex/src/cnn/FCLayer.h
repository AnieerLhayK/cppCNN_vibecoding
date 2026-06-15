#pragma once

#include "cnn/Layer.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cppcnn {

class FCLayer final : public Layer {
public:
    FCLayer(std::size_t inputSize, std::size_t outputSize, std::uint32_t seed = 1);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& outputGradient) override;
    void zeroGrad() override;
    void update(float learningRate, float gradientScale, float weightDecay) override;
    void updateWithMomentum(
        float learningRate, float gradientScale,
        float weightDecay, float momentum) override;

    [[nodiscard]] std::string type() const override;
    [[nodiscard]] bool isTrainable() const noexcept override { return true; }

    void saveOptimizerState(std::vector<float>& buffer) const override;
    void loadOptimizerState(const float*& cursor) override;
    [[nodiscard]] std::size_t optimizerStateSize() const noexcept override;

    [[nodiscard]] const std::vector<float>& weights() const noexcept;
    [[nodiscard]] const std::vector<float>& biases() const noexcept;
    [[nodiscard]] std::vector<float>& mutableWeights() noexcept;
    [[nodiscard]] std::vector<float>& mutableBiases() noexcept;

private:
    std::size_t inputSize_;
    std::size_t outputSize_;
    std::vector<float> weights_;
    std::vector<float> biases_;
    std::vector<float> weightGradients_;
    std::vector<float> biasGradients_;
    std::vector<float> weightVelocity_;
    std::vector<float> biasVelocity_;
    Tensor cachedInput_;
};

}  // namespace cppcnn
