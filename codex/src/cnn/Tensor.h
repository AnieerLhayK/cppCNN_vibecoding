#pragma once

#include <cstddef>
#include <vector>

namespace cppcnn {

class Tensor {
public:
    Tensor() = default;
    Tensor(std::size_t channels, std::size_t height, std::size_t width, float value = 0.0F);

    void resize(std::size_t channels, std::size_t height, std::size_t width, float value = 0.0F);
    void fill(float value);

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::size_t channels() const noexcept;
    [[nodiscard]] std::size_t height() const noexcept;
    [[nodiscard]] std::size_t width() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;

    float& at(std::size_t channel, std::size_t row, std::size_t column);
    const float& at(std::size_t channel, std::size_t row, std::size_t column) const;

    float& operator[](std::size_t index);
    const float& operator[](std::size_t index) const;

    [[nodiscard]] std::vector<float>& values() noexcept;
    [[nodiscard]] const std::vector<float>& values() const noexcept;

private:
    [[nodiscard]] std::size_t offset(
        std::size_t channel,
        std::size_t row,
        std::size_t column) const;

    std::size_t channels_ = 0;
    std::size_t height_ = 0;
    std::size_t width_ = 0;
    std::vector<float> values_;
};

}  // namespace cppcnn
