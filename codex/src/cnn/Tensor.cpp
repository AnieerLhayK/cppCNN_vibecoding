#include "cnn/Tensor.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace cppcnn {

namespace {

std::size_t checkedElementCount(
    const std::size_t channels,
    const std::size_t height,
    const std::size_t width) {
    if (channels == 0 || height == 0 || width == 0) {
        return 0;
    }

    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (channels > maximum / height || channels * height > maximum / width) {
        throw std::overflow_error("Tensor dimensions are too large.");
    }
    return channels * height * width;
}

}  // namespace

Tensor::Tensor(
    const std::size_t channels,
    const std::size_t height,
    const std::size_t width,
    const float value) {
    resize(channels, height, width, value);
}

void Tensor::resize(
    const std::size_t channels,
    const std::size_t height,
    const std::size_t width,
    const float value) {
    const auto count = checkedElementCount(channels, height, width);
    channels_ = count == 0 ? 0 : channels;
    height_ = count == 0 ? 0 : height;
    width_ = count == 0 ? 0 : width;
    values_.assign(count, value);
}

void Tensor::fill(const float value) {
    std::fill(values_.begin(), values_.end(), value);
}

bool Tensor::empty() const noexcept {
    return values_.empty();
}

std::size_t Tensor::channels() const noexcept {
    return channels_;
}

std::size_t Tensor::height() const noexcept {
    return height_;
}

std::size_t Tensor::width() const noexcept {
    return width_;
}

std::size_t Tensor::size() const noexcept {
    return values_.size();
}

float& Tensor::at(
    const std::size_t channel,
    const std::size_t row,
    const std::size_t column) {
    return values_.at(offset(channel, row, column));
}

const float& Tensor::at(
    const std::size_t channel,
    const std::size_t row,
    const std::size_t column) const {
    return values_.at(offset(channel, row, column));
}

float& Tensor::operator[](const std::size_t index) {
    return values_.at(index);
}

const float& Tensor::operator[](const std::size_t index) const {
    return values_.at(index);
}

std::vector<float>& Tensor::values() noexcept {
    return values_;
}

const std::vector<float>& Tensor::values() const noexcept {
    return values_;
}

std::size_t Tensor::offset(
    const std::size_t channel,
    const std::size_t row,
    const std::size_t column) const {
    if (channel >= channels_ || row >= height_ || column >= width_) {
        throw std::out_of_range("Tensor index is outside its dimensions.");
    }
    return (channel * height_ + row) * width_ + column;
}

}  // namespace cppcnn
