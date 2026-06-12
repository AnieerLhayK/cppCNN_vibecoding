#pragma once

#include "cnn/Layer.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace cppcnn {

struct ModelInfo {
    std::uint32_t version = 0;
    std::size_t classCount = 0;
    std::size_t trainableLayerCount = 0;
    std::size_t parameterCount = 0;
};

class CNN {
public:
    explicit CNN(std::size_t classCount = 10, std::uint32_t seed = 42);

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& outputGradient);
    void zeroGrad();
    void update(float learningRate, float gradientScale = 1.0F, float weightDecay = 0.0F);

    [[nodiscard]] std::size_t predict(const Tensor& input, float* confidence = nullptr);
    void saveModel(const std::filesystem::path& path) const;
    void loadModel(const std::filesystem::path& path);
    [[nodiscard]] static std::size_t modelClassCount(const std::filesystem::path& path);
    [[nodiscard]] static ModelInfo inspectModel(const std::filesystem::path& path);

    [[nodiscard]] std::size_t classCount() const noexcept;
    [[nodiscard]] std::size_t layerCount() const noexcept;

private:
    std::size_t classCount_;
    std::vector<std::unique_ptr<Layer>> layers_;
};

}  // namespace cppcnn
