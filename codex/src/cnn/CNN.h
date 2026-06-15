#pragma once

#include "cnn/Layer.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace cppcnn {

enum class CNNArchitecture {
    LeNet,        // Original: 2xConv(6,16) + 2xFC(120,N)
    Enhanced,     // Wider/deeper: 3xConv(32,64,128) + 3xFC(512,256,N)
};

struct ModelInfo {
    std::uint32_t version = 0;
    std::size_t classCount = 0;
    std::size_t trainableLayerCount = 0;
    std::size_t parameterCount = 0;
    CNNArchitecture architecture = CNNArchitecture::LeNet;
};

class CNN {
public:
    explicit CNN(std::size_t classCount = 10, std::uint32_t seed = 42,
                 CNNArchitecture arch = CNNArchitecture::LeNet);

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& outputGradient);
    void zeroGrad();
    void update(float learningRate, float gradientScale = 1.0F, float weightDecay = 0.0F);
    void updateWithMomentum(float learningRate, float gradientScale = 1.0F,
                            float weightDecay = 0.0F, float momentum = 0.9F);

    // Training mode
    void setTraining(bool enabled);
    [[nodiscard]] bool isTraining() const noexcept { return training_; }

    [[nodiscard]] std::size_t predict(const Tensor& input, float* confidence = nullptr);
    void saveModel(const std::filesystem::path& path) const;
    void loadModel(const std::filesystem::path& path);
    [[nodiscard]] static std::size_t modelClassCount(const std::filesystem::path& path);
    [[nodiscard]] static ModelInfo inspectModel(const std::filesystem::path& path);

    [[nodiscard]] std::size_t classCount() const noexcept;
    [[nodiscard]] std::size_t layerCount() const noexcept;
    [[nodiscard]] CNNArchitecture architecture() const noexcept { return arch_; }

    // Optimizer state serialization for resume
    void saveOptimizerState(std::vector<float>& buffer) const;
    void loadOptimizerState(const float* buffer);
    [[nodiscard]] std::size_t optimizerStateSize() const noexcept;

private:
    std::size_t classCount_;
    bool training_ = true;
    CNNArchitecture arch_;
    std::vector<std::unique_ptr<Layer>> layers_;
};

}  // namespace cppcnn
