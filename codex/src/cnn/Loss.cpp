#include "cnn/Loss.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cppcnn {

float CrossEntropyLoss::value(const Tensor& probabilities, const std::size_t label) {
    if (label >= probabilities.size()) {
        throw std::out_of_range("Class label is outside the probability vector.");
    }
    return -std::log(std::max(probabilities[label], 1.0e-7F));
}

Tensor CrossEntropyLoss::gradient(
    const Tensor& probabilities,
    const std::size_t label) {
    if (label >= probabilities.size()) {
        throw std::out_of_range("Class label is outside the probability vector.");
    }

    Tensor result(
        probabilities.channels(),
        probabilities.height(),
        probabilities.width());
    result[label] = -1.0F / std::max(probabilities[label], 1.0e-7F);
    return result;
}

std::size_t argmax(const Tensor& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot compute argmax of an empty tensor.");
    }
    return static_cast<std::size_t>(
        std::distance(
            values.values().begin(),
            std::max_element(values.values().begin(), values.values().end())));
}

void Accuracy::add(const Tensor& probabilities, const std::size_t label) {
    if (argmax(probabilities) == label) {
        ++correct_;
    }
    ++total_;
}

void Accuracy::reset() noexcept {
    correct_ = 0;
    total_ = 0;
}

std::size_t Accuracy::correct() const noexcept {
    return correct_;
}

std::size_t Accuracy::total() const noexcept {
    return total_;
}

float Accuracy::value() const noexcept {
    return total_ == 0
        ? 0.0F
        : static_cast<float>(correct_) / static_cast<float>(total_);
}

}  // namespace cppcnn
