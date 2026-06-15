#include "cnn/Augmenter.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace cppcnn {

namespace {

float randomFloat(std::mt19937& generator, const float min, const float max) {
    std::uniform_real_distribution<float> dist(min, max);
    return dist(generator);
}

// Bilinear interpolation at (row, col) in a single-channel image
float sampleChannel(const float* data, std::size_t height, std::size_t width,
                    float row, float col) {
    // Clamp to edge
    row = std::max(0.0F, std::min(row, static_cast<float>(height) - 1.001F));
    col = std::max(0.0F, std::min(col, static_cast<float>(width) - 1.001F));

    const std::size_t r0 = static_cast<std::size_t>(row);
    const std::size_t c0 = static_cast<std::size_t>(col);
    const std::size_t r1 = std::min(r0 + 1, height - 1);
    const std::size_t c1 = std::min(c0 + 1, width - 1);
    const float ry = row - static_cast<float>(r0);
    const float rx = col - static_cast<float>(c0);

    return data[r0 * width + c0] * (1.0F - rx) * (1.0F - ry)
         + data[r0 * width + c1] * rx * (1.0F - ry)
         + data[r1 * width + c0] * (1.0F - rx) * ry
         + data[r1 * width + c1] * rx * ry;
}

}  // namespace

Augmenter::Augmenter(AugmentConfig config, const std::uint32_t seed)
    : config_(std::move(config)), seed_(seed) {}

Tensor Augmenter::augment(const Tensor& input) {
    if (input.empty()) {
        throw std::invalid_argument("Augmenter input cannot be empty.");
    }
    Tensor result = input;
    std::mt19937 generator(seed_);

    if (config_.enableRotation) {
        applyRotation(result, generator);
    }
    if (config_.enableTranslation) {
        applyTranslation(result, generator);
    }
    if (config_.enableScaling) {
        applyScaling(result, generator);
    }
    if (config_.enableBrightness) {
        applyBrightness(result, generator);
    }
    if (config_.enableContrast) {
        applyContrast(result, generator);
    }
    if (config_.enableNoise) {
        applyNoise(result, generator);
    }

    // Clamp to [0, 1]
    for (float& val : result.values()) {
        val = std::max(0.0F, std::min(1.0F, val));
    }
    return result;
}

void Augmenter::applyRotation(Tensor& image, std::mt19937& generator) {
    const float angleDeg = randomFloat(generator, -config_.maxRotationDeg, config_.maxRotationDeg);
    if (std::fabs(angleDeg) < 0.5F) return;

    const float angleRad = angleDeg * 3.14159265F / 180.0F;
    const float cosA = std::cos(angleRad);
    const float sinA = std::sin(angleRad);
    const std::size_t H = image.height();
    const std::size_t W = image.width();
    const float cy = static_cast<float>(H) / 2.0F;
    const float cx = static_cast<float>(W) / 2.0F;

    Tensor result(image.channels(), H, W);
    for (std::size_t c = 0; c < image.channels(); ++c) {
        for (std::size_t row = 0; row < H; ++row) {
            for (std::size_t col = 0; col < W; ++col) {
                // Rotate around center
                const float sr = static_cast<float>(row) - cy;
                const float sc = static_cast<float>(col) - cx;
                const float srcRow = cosA * sr + sinA * sc + cy;
                const float srcCol = -sinA * sr + cosA * sc + cx;
                result.at(c, row, col) = sampleChannel(
                    &image.at(c, 0, 0), H, W, srcRow, srcCol);
            }
        }
    }
    image = std::move(result);
}

void Augmenter::applyTranslation(Tensor& image, std::mt19937& generator) {
    const int dy = static_cast<int>(std::round(
        randomFloat(generator, -static_cast<float>(config_.maxTranslationPx),
                    static_cast<float>(config_.maxTranslationPx))));
    const int dx = static_cast<int>(std::round(
        randomFloat(generator, -static_cast<float>(config_.maxTranslationPx),
                    static_cast<float>(config_.maxTranslationPx))));
    if (dy == 0 && dx == 0) return;

    const std::size_t H = image.height();
    const std::size_t W = image.width();
    Tensor result(image.channels(), H, W, 0.0F);

    for (std::size_t c = 0; c < image.channels(); ++c) {
        for (std::size_t row = 0; row < H; ++row) {
            const std::ptrdiff_t srcRow = static_cast<std::ptrdiff_t>(row) - dy;
            if (srcRow < 0 || srcRow >= static_cast<std::ptrdiff_t>(H)) continue;
            for (std::size_t col = 0; col < W; ++col) {
                const std::ptrdiff_t srcCol = static_cast<std::ptrdiff_t>(col) - dx;
                if (srcCol < 0 || srcCol >= static_cast<std::ptrdiff_t>(W)) continue;
                result.at(c, row, col) = image.at(
                    c, static_cast<std::size_t>(srcRow), static_cast<std::size_t>(srcCol));
            }
        }
    }
    image = std::move(result);
}

void Augmenter::applyScaling(Tensor& image, std::mt19937& generator) {
    const float scale = 1.0F + randomFloat(generator, -config_.maxScaleChange, config_.maxScaleChange);
    if (std::fabs(scale - 1.0F) < 0.01F) return;

    const std::size_t H = image.height();
    const std::size_t W = image.width();
    const float cy = static_cast<float>(H) / 2.0F;
    const float cx = static_cast<float>(W) / 2.0F;
    const float invScale = 1.0F / scale;

    Tensor result(image.channels(), H, W);
    for (std::size_t c = 0; c < image.channels(); ++c) {
        for (std::size_t row = 0; row < H; ++row) {
            for (std::size_t col = 0; col < W; ++col) {
                const float srcRow = (static_cast<float>(row) - cy) * invScale + cy;
                const float srcCol = (static_cast<float>(col) - cx) * invScale + cx;
                result.at(c, row, col) = sampleChannel(
                    &image.at(c, 0, 0), H, W, srcRow, srcCol);
            }
        }
    }
    image = std::move(result);
}

void Augmenter::applyBrightness(Tensor& image, std::mt19937& generator) {
    const float delta = randomFloat(generator, -config_.maxBrightnessChange, config_.maxBrightnessChange);
    if (std::fabs(delta) < 0.01F) return;

    for (float& val : image.values()) {
        val += delta;
    }
}

void Augmenter::applyContrast(Tensor& image, std::mt19937& generator) {
    const float factor = 1.0F + randomFloat(generator, -config_.maxContrastChange, config_.maxContrastChange);
    if (std::fabs(factor - 1.0F) < 0.01F) return;

    // Compute mean per channel
    const std::size_t H = image.height();
    const std::size_t W = image.width();
    for (std::size_t c = 0; c < image.channels(); ++c) {
        float mean = 0.0F;
        for (std::size_t row = 0; row < H; ++row) {
            for (std::size_t col = 0; col < W; ++col) {
                mean += image.at(c, row, col);
            }
        }
        mean /= static_cast<float>(H * W);

        for (std::size_t row = 0; row < H; ++row) {
            for (std::size_t col = 0; col < W; ++col) {
                float& val = image.at(c, row, col);
                val = mean + factor * (val - mean);
            }
        }
    }
}

void Augmenter::applyNoise(Tensor& image, std::mt19937& generator) {
    if (config_.noiseSigma <= 0.0F) return;
    std::normal_distribution<float> noise(0.0F, config_.noiseSigma);
    for (float& val : image.values()) {
        val += noise(generator);
    }
}

const AugmentConfig& Augmenter::config() const noexcept {
    return config_;
}

}  // namespace cppcnn
