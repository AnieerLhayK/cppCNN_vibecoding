#pragma once

#include "cnn/Tensor.h"

#include <cstddef>

namespace cppcnn {

class CrossEntropyLoss {
public:
    static float value(const Tensor& probabilities, std::size_t label);
    static Tensor gradient(const Tensor& probabilities, std::size_t label);
};

[[nodiscard]] std::size_t argmax(const Tensor& values);

class Accuracy {
public:
    void add(const Tensor& probabilities, std::size_t label);
    void reset() noexcept;

    [[nodiscard]] std::size_t correct() const noexcept;
    [[nodiscard]] std::size_t total() const noexcept;
    [[nodiscard]] float value() const noexcept;

private:
    std::size_t correct_ = 0;
    std::size_t total_ = 0;
};

}  // namespace cppcnn
