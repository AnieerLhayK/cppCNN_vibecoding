#include "cnn/Loss.h"

#include <algorithm>
#include <cmath>
#include <limits>
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

// PerClassAccuracy
PerClassAccuracy::PerClassAccuracy(const std::size_t classCount)
    : classCount_(classCount) {
    if (classCount == 0) {
        throw std::invalid_argument("PerClassAccuracy class count must be positive.");
    }
    correct_.assign(classCount_, 0);
    total_.assign(classCount_, 0);
}

void PerClassAccuracy::add(const Tensor& probabilities, const std::size_t label) {
    if (label >= classCount_) {
        throw std::out_of_range("Label exceeds class count in PerClassAccuracy.");
    }
    if (argmax(probabilities) == label) {
        ++correct_[label];
    }
    ++total_[label];
}

void PerClassAccuracy::reset() noexcept {
    std::fill(correct_.begin(), correct_.end(), 0);
    std::fill(total_.begin(), total_.end(), 0);
}

std::size_t PerClassAccuracy::classCount() const noexcept {
    return classCount_;
}

float PerClassAccuracy::classAccuracy(const std::size_t classIndex) const {
    if (classIndex >= classCount_) {
        throw std::out_of_range("Class index out of range.");
    }
    return total_[classIndex] == 0
        ? 0.0F
        : static_cast<float>(correct_[classIndex]) / static_cast<float>(total_[classIndex]);
}

float PerClassAccuracy::meanAccuracy() const {
    float sum = 0.0F;
    std::size_t valid = 0;
    for (std::size_t index = 0; index < classCount_; ++index) {
        if (total_[index] > 0) {
            sum += classAccuracy(index);
            ++valid;
        }
    }
    return valid == 0 ? 0.0F : sum / static_cast<float>(valid);
}

float PerClassAccuracy::minAccuracy() const {
    float minVal = std::numeric_limits<float>::max();
    for (std::size_t index = 0; index < classCount_; ++index) {
        if (total_[index] > 0) {
            minVal = std::min(minVal, classAccuracy(index));
        }
    }
    return minVal == std::numeric_limits<float>::max() ? 0.0F : minVal;
}

// ConfusionMatrix
ConfusionMatrix::ConfusionMatrix(const std::size_t classCount)
    : classCount_(classCount) {
    if (classCount == 0) {
        throw std::invalid_argument("ConfusionMatrix class count must be positive.");
    }
    matrix_.assign(classCount_ * classCount_, 0);
}

void ConfusionMatrix::add(const std::size_t predicted, const std::size_t actual) {
    if (predicted >= classCount_ || actual >= classCount_) {
        throw std::out_of_range("Class index out of range in ConfusionMatrix.");
    }
    ++matrix_[predicted * classCount_ + actual];
}

void ConfusionMatrix::reset() noexcept {
    std::fill(matrix_.begin(), matrix_.end(), 0);
}

std::size_t ConfusionMatrix::classCount() const noexcept {
    return classCount_;
}

std::size_t ConfusionMatrix::value(const std::size_t predicted, const std::size_t actual) const {
    if (predicted >= classCount_ || actual >= classCount_) {
        throw std::out_of_range("Class index out of range in ConfusionMatrix.");
    }
    return matrix_[predicted * classCount_ + actual];
}

const std::vector<std::size_t>& ConfusionMatrix::data() const noexcept {
    return matrix_;
}

std::size_t ConfusionMatrix::total() const noexcept {
    std::size_t sum = 0;
    for (const auto& val : matrix_) {
        sum += val;
    }
    return sum;
}

}  // namespace cppcnn
