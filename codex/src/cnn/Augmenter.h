#pragma once

#include "cnn/Tensor.h"

#include <cstddef>
#include <cstdint>
#include <random>

namespace cppcnn {

struct AugmentConfig {
    bool enableRotation = true;
    float maxRotationDeg = 15.0F;

    bool enableTranslation = true;
    int maxTranslationPx = 4;

    bool enableScaling = true;
    float maxScaleChange = 0.10F;  // +/- 10%

    bool enableBrightness = true;
    float maxBrightnessChange = 0.20F;

    bool enableContrast = true;
    float maxContrastChange = 0.20F;

    bool enableNoise = true;
    float noiseSigma = 0.05F;

    bool enableBlur = false;
    bool enableCompression = false;
};

class Augmenter {
public:
    explicit Augmenter(AugmentConfig config = {}, std::uint32_t seed = 42);

    Tensor augment(const Tensor& input);

    [[nodiscard]] const AugmentConfig& config() const noexcept;

private:
    void applyRotation(Tensor& image, std::mt19937& generator);
    void applyTranslation(Tensor& image, std::mt19937& generator);
    void applyScaling(Tensor& image, std::mt19937& generator);
    void applyBrightness(Tensor& image, std::mt19937& generator);
    void applyContrast(Tensor& image, std::mt19937& generator);
    void applyNoise(Tensor& image, std::mt19937& generator);

    AugmentConfig config_;
    std::uint32_t seed_;
};

}  // namespace cppcnn
