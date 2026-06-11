#pragma once

#include "cnn/Tensor.h"

#include <string>

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

    [[nodiscard]] virtual std::string type() const = 0;
};

}  // namespace cppcnn
