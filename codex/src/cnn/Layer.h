#pragma once

#include "cnn/Tensor.h"

#include <cstddef>
#include <string>
#include <vector>

namespace cppcnn {

class Layer {
public:
    virtual ~Layer() = default;

    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& outputGradient) = 0;

    virtual void zeroGrad() {}
    virtual void update(float learningRate, float gradientScale, float weightDecay) {
        static_cast<void>(learningRate);
        static_cast<void>(gradientScale);
        static_cast<void>(weightDecay);
    }

    // Momentum SGD support
    virtual void updateWithMomentum(
        float learningRate, float gradientScale,
        float weightDecay, float momentum) {
        static_cast<void>(learningRate);
        static_cast<void>(gradientScale);
        static_cast<void>(weightDecay);
        static_cast<void>(momentum);
    }

    [[nodiscard]] virtual std::string type() const = 0;

    // Training mode toggle (for dropout etc.)
    virtual void train(bool enabled) { static_cast<void>(enabled); }
    [[nodiscard]] virtual bool isTrainable() const noexcept { return false; }

    // Optimizer state serialization for resume
    virtual void saveOptimizerState(std::vector<float>& buffer) const {
        static_cast<void>(buffer);
    }
    virtual void loadOptimizerState(const float*& cursor) {
        static_cast<void>(cursor);
    }
    [[nodiscard]] virtual std::size_t optimizerStateSize() const noexcept { return 0; }
};

}  // namespace cppcnn
