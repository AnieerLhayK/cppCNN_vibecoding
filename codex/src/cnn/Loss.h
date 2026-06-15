#pragma once

#include "cnn/Tensor.h"

#include <cstddef>
#include <vector>

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

// Per-class accuracy tracker
class PerClassAccuracy {
public:
    explicit PerClassAccuracy(std::size_t classCount);

    void add(const Tensor& probabilities, std::size_t label);
    void reset() noexcept;

    [[nodiscard]] std::size_t classCount() const noexcept;
    [[nodiscard]] float classAccuracy(std::size_t classIndex) const;
    [[nodiscard]] float meanAccuracy() const;
    [[nodiscard]] float minAccuracy() const;

private:
    std::size_t classCount_;
    std::vector<std::size_t> correct_;
    std::vector<std::size_t> total_;
};

// Confusion matrix
class ConfusionMatrix {
public:
    explicit ConfusionMatrix(std::size_t classCount);

    void add(std::size_t predicted, std::size_t actual);
    void reset() noexcept;

    [[nodiscard]] std::size_t classCount() const noexcept;
    [[nodiscard]] std::size_t value(std::size_t predicted, std::size_t actual) const;
    [[nodiscard]] const std::vector<std::size_t>& data() const noexcept;
    [[nodiscard]] std::size_t total() const noexcept;

private:
    std::size_t classCount_;
    std::vector<std::size_t> matrix_;  // row-major: [predicted * classCount + actual]
};

}  // namespace cppcnn
